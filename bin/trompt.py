# Feed-forward network[s].

import math
import statistics
from pathlib import Path
from typing import Any

import delu
import numpy as np
import rtdl_num_embeddings
import torch
import torch.nn as nn
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from typing_extensions import NotRequired, TypedDict

import lib
from lib import KWArgs, PartKey
import rtdl_revisiting_models

class ImportanceGetter(nn.Module): #Figure 3 part 1
    def __init__(self, P, C, d):
        super().__init__()
        self.colemb = nn.Parameter(torch.empty(C, d))
        self.pemb = nn.Parameter(torch.empty(P, d))
        torch.nn.init.normal_(self.colemb, std=0.01)
        torch.nn.init.normal_(self.pemb, std=0.01)
        self.C = C
        self.P = P
        self.d = d
        self.dense = nn.Linear(2 * self.d, self.d)
        self.laynorm1 = nn.LayerNorm(self.d)
        self.laynorm2 = nn.LayerNorm(self.d)
    def forward(self, O):
        eprompt = self.pemb.unsqueeze(0).repeat(O.shape[0], 1, 1)

        dense_out = self.dense(torch.cat((self.laynorm1(eprompt), O), dim=-1))
        dense_out = dense_out + eprompt + O
        ecolumn = self.laynorm2(self.colemb.unsqueeze(0).repeat(O.shape[0], 1, 1))

        return torch.softmax(dense_out @ ecolumn.transpose(1, 2), dim=-1)

class TromptEmbedding(nn.Module): # Figure 3 part 2
    def __init__(self, n_num_features, n_bin_features, cat_cardinalities, d):
        super().__init__()
        self.d = d
        self.m_num = rtdl_revisiting_models.LinearEmbeddings(n_bin_features + n_num_features, d) if (n_bin_features + n_num_features) else None
        self.m_cat = lib.deep.CategoricalEmbeddings1d(cat_cardinalities, d) if cat_cardinalities else None
        self.relu = nn.ReLU()
        self.laynorm1 = nn.LayerNorm(self.d)
        self.laynorm2 = nn.LayerNorm(self.d)

    def forward(self, x_num, x_bin, x_cat):
        if not(x_bin is None):
            xnc = torch.cat((x_num, x_bin), dim=-1)
        else:
            xnc = x_num
        if not(x_cat is None):
            return torch.cat((self.laynorm1(self.relu(self.m_num(xnc))), self.laynorm2(self.m_cat(x_cat))), dim=1)
        return self.laynorm1(self.relu(self.m_num(xnc)))

class Expander(nn.Module): #Figure 3 part 3
    def __init__(self, P):
        super().__init__()
        self.lin = nn.Linear(1, P)
        self.relu = nn.ReLU()
        self.gn = nn.GroupNorm(2, P)

    def forward(self, x):
        res = (self.relu(self.lin(x.unsqueeze(-1))))
        return x.unsqueeze(1) + self.gn(torch.permute(res, (0, 3, 1, 2)))

class TromptCell(nn.Module):
    def __init__(self, n_num_features, n_bin_features, cat_cardinalities, P, d):
        super().__init__()
        C = n_num_features + n_bin_features + len(cat_cardinalities)
        self.enc = TromptEmbedding(n_num_features, n_bin_features, cat_cardinalities, d)
        self.fe = ImportanceGetter(P, C, d)
        self.ex = Expander(P)

    def forward(self, x_num, x_bin, x_cat, O):
        x_res = self.ex(self.enc(x_num, x_bin, x_cat))
        M = self.fe(O)
        return (M.unsqueeze(-1) * x_res).sum(dim=2)

class TromptDownstream(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l1 = nn.Linear(d, 1)
        self.l2 = nn.Linear(d, d)
        self.relu = nn.ReLU()
        self.laynorm1 = nn.LayerNorm(d)
        self.lf = nn.Linear(d, 1)
    def forward(self, o):
        pw = torch.softmax(self.l1(o).squeeze(-1), dim=-1)
        xnew = (pw.unsqueeze(-1) * o).sum(dim=-2)
        return self.lf(self.laynorm1(self.relu(self.l2(xnew))))

class Model(nn.Module): #Trompt
    def __init__(self, n_num_features, n_bin_features, cat_cardinalities, P, d, n_cycles):
        super().__init__()
        self.tcell = TromptCell(n_num_features, n_bin_features, cat_cardinalities, P, d)
        self.tdown = TromptDownstream(d)
        self.init_rec = nn.Parameter(torch.empty(P, d))
        nn.init.normal_(self.init_rec, std=0.01)
        self.n_cycles = n_cycles
    def forward(self, x_num : None | Tensor = None, x_bin : None | Tensor = None, x_cat : None | Tensor = None):
        O = self.init_rec.unsqueeze(0).repeat(x_num.shape[0], 1, 1)
        outputs = []
        for i in range(self.n_cycles):
            O = self.tcell(x_num, x_bin, x_cat, O)
            outputs.append(self.tdown(O))
        return torch.stack(outputs, dim=1).squeeze(-1)



class Config(TypedDict):
    seed: int
    data: KWArgs
    bins: NotRequired[KWArgs]
    model: KWArgs
    optimizer: KWArgs
    n_lr_warmup_epochs: NotRequired[int]
    batch_size: int
    patience: int
    n_epochs: int
    gradient_clipping_norm: NotRequired[float]
    parameter_statistics: NotRequired[bool]


def main(
    config: Config, output: str | Path, *, force: bool = False
) -> None | lib.JSONDict:
    # >>> start
    assert set(config) >= Config.__required_keys__
    assert set(config) <= Config.__required_keys__ | Config.__optional_keys__
    if not lib.start(output, force=force):
        return None

    lib.show_config(config)  # type: ignore[code]
    output = Path(output)
    delu.random.seed(config['seed'])
    device = lib.get_device()
    report = lib.create_report(config)  # type: ignore[code]

    # >>> dataset
    dataset = lib.data.build_dataset(**config['data'])
    if dataset.task.is_regression:
        dataset.data['y'], regression_label_stats = lib.data.standardize_labels(
            dataset.data['y']
        )
    else:
        regression_label_stats = None
    dataset = dataset.to_torch(device)
    Y_train = dataset.data['y']['train'].to(
        torch.long if dataset.task.is_multiclass else torch.float
    )

    # >>> model
    if 'bins' in config:
        compute_bins_kwargs = (
            {
                'y': Y_train.to(
                    torch.long if dataset.task.is_classification else torch.float
                ),
                'regression': dataset.task.is_regression,
                'verbose': True,
            }
            if 'tree_kwargs' in config['bins']
            else {}
        )
        bin_edges = rtdl_num_embeddings.compute_bins(
            dataset['x_num']['train'], **config['bins'], **compute_bins_kwargs
        )
        logger.info(f'Bin counts: {[len(x) - 1 for x in bin_edges]}')
    else:
        bin_edges = None
    model = Model(
        n_num_features=dataset.n_num_features,
        n_bin_features=dataset.n_bin_features,
        cat_cardinalities=dataset.compute_cat_cardinalities(),

        **config['model'],

    )
    report['n_parameters'] = lib.deep.get_n_parameters(model)
    logger.info(f'n_parameters = {report["n_parameters"]}')
    report['prediction_type'] = 'labels' if dataset.task.is_regression else 'logits'
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # >>> training
    optimizer = lib.deep.make_optimizer(
        **config['optimizer'], params=lib.deep.make_parameter_groups(model)
    )
    loss_fn = lib.deep.get_loss_fn(dataset.task.type_)
    gradient_clipping_norm = config.get('gradient_clipping_norm')

    step = 0
    batch_size = config['batch_size']
    report['epoch_size'] = epoch_size = math.ceil(dataset.size('train') / batch_size)
    eval_batch_size = 32768
    chunk_size = None
    generator = torch.Generator(device).manual_seed(config['seed'])

    report['metrics'] = {'val': {'score': -math.inf}}
    if 'n_lr_warmup_epochs' in config:
        n_warmup_steps = min(10000, config['n_lr_warmup_epochs'] * epoch_size)
        n_warmup_steps = max(1, math.trunc(n_warmup_steps / epoch_size)) * epoch_size
        logger.info(f'{n_warmup_steps=}')
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=n_warmup_steps
        )
    else:
        lr_scheduler = None
    timer = delu.tools.Timer()
    early_stopping = delu.tools.EarlyStopping(config['patience'], mode='max')
    parameter_statistics = config.get('parameter_statistics', config['seed'] == 1)
    training_log = []
    writer = torch.utils.tensorboard.SummaryWriter(output)  # type: ignore[code]

    def apply_model(part: PartKey, idx: Tensor) -> Tensor:
        return model(
            **{
                key: dataset.data[key][part][idx]  # type: ignore[index]
                for key in ['x_num', 'x_bin', 'x_cat']
                if key in dataset  # type: ignore[index]
            }
        )

    @torch.inference_mode()
    def evaluate(
        parts: list[PartKey], eval_batch_size: int
    ) -> tuple[dict[PartKey, Any], dict[PartKey, np.ndarray], int]:
        model.eval()
        predictions: dict[PartKey, np.ndarray] = {}
        for part in parts:
            while eval_batch_size:
                try:
                    predictions[part] = (
                        torch.cat(
                            [
                                apply_model(part, idx).mean(-1)
                                for idx in torch.arange(
                                    len(dataset.data['y'][part]),
                                    device=device,
                                ).split(eval_batch_size)
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    logger.warning(f'eval_batch_size = {eval_batch_size}')
                else:
                    break
            if not eval_batch_size:
                RuntimeError('Not enough memory even for eval_batch_size=1')
        if regression_label_stats is not None:
            predictions = {
                k: v * regression_label_stats.std + regression_label_stats.mean
                for k, v in predictions.items()
            }
        metrics = (
            dataset.task.calculate_metrics(predictions, report['prediction_type'])
            if lib.are_valid_predictions(predictions)
            else {x: {'score': -999999.0} for x in predictions}
        )
        return metrics, predictions, eval_batch_size

    def save_checkpoint() -> None:
        lib.dump_checkpoint(
            output,
            {
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'generator': generator.get_state(),
                'random_state': delu.random.get_state(),
                'early_stopping': early_stopping,
                'report': report,
                'timer': timer,
                'training_log': training_log,
            }
            | (
                {}
                if lr_scheduler is None
                else {'lr_scheduler': lr_scheduler.state_dict()}
            ),
        )
        lib.dump_report(output, report)
        lib.backup_output(output)

    
    print()
    timer.run()
    K = model.n_cycles
    while config['n_epochs'] == -1 or step // epoch_size < config['n_epochs']:
        print(f'[...] {output} | {timer}')

        # >>>
        model.train()
        epoch_losses = []
        for batch_idx in tqdm(
            torch.randperm(
                len(dataset.data['y']['train']), generator=generator, device=device
            ).split(batch_size),
            desc=f'Epoch {step // epoch_size} Step {step}',
        ):
            loss, new_chunk_size = lib.deep.zero_grad_forward_backward(
                optimizer,
                lambda idx: loss_fn(apply_model('train', idx), Y_train[idx].unsqueeze(1).repeat(1, K)),
                batch_idx,
                chunk_size or batch_size,
            )

            if parameter_statistics and (
                step % epoch_size == 0  # The first batch of the epoch.
                or step // epoch_size == 0  # The first epoch.
            ):
                for k, v in lib.deep.compute_parameter_stats(model).items():
                    writer.add_scalars(k, v, step, timer.elapsed())
                    del k, v

            if gradient_clipping_norm is not None:
                nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), gradient_clipping_norm
                )
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
            step += 1
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or batch_size):
                chunk_size = new_chunk_size
                logger.warning(f'chunk_size = {chunk_size}')

        epoch_losses = torch.stack(epoch_losses).tolist()
        mean_loss = statistics.mean(epoch_losses)
        metrics, predictions, eval_batch_size = evaluate(
            ['val', 'test'], eval_batch_size
        )

        training_log.append(
            {'epoch-losses': epoch_losses, 'metrics': metrics, 'time': timer.elapsed()}
        )
        lib.print_metrics(mean_loss, metrics)
        writer.add_scalars('loss', {'train': mean_loss}, step, timer.elapsed())
        for part in metrics:
            writer.add_scalars(
                'score', {part: metrics[part]['score']}, step, timer.elapsed()
            )

        if metrics['val']['score'] > report['metrics']['val']['score']:
            print('🌸 New best epoch! 🌸')
            report['best_step'] = step
            report['metrics'] = metrics
            save_checkpoint()
            lib.dump_predictions(output, predictions)

        early_stopping.update(metrics['val']['score'])
        if early_stopping.should_stop() or not lib.are_valid_predictions(predictions):
            break

        print()
    report['time'] = str(timer)

    # >>> finish
    model.load_state_dict(lib.load_checkpoint(output)['model'])
    report['metrics'], predictions, _ = evaluate(
        ['train', 'val', 'test'], eval_batch_size
    )
    report['chunk_size'] = chunk_size
    report['eval_batch_size'] = eval_batch_size
    lib.dump_predictions(output, predictions)
    lib.dump_summary(output, lib.summarize(report))
    save_checkpoint()
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_MainFunction_cli(main)
