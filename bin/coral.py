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





class MLP(nn.Module):
    class Block(nn.Module):
        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: str,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    Head = nn.Linear

    def __init__(
        self,
        *,
        d_in: int,
        d_out: int,
        n_blocks: int,
        d_block: int,
        activation: str = "ReLU",
        dropout: float,
    ) -> None:
        assert n_blocks > 0
        super().__init__()
        d_layer = d_block
        self.blocks = nn.Sequential(
            *[
                MLP.Block(
                    d_in=d_layer if block_i else d_in,
                    d_out=d_layer,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for block_i in range(n_blocks)
            ]
        )
        self.head = None if d_out is None else MLP.Head(d_layer, d_out)

    @property
    def d_out(self) -> int:
        return (
            self.blocks[-1].linear.out_features  # type: ignore[code]
            if self.head is None
            else self.head.out_features
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        if self.head is not None:
            xnew = self.head(x)
        #print(x.shape)
        return xnew, x


class Model(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: None | dict = None,
        num_embeddings: None | dict = None,  # lib.deep.ModuleSpec
        backbone: dict,  # lib.deep.ModuleSpec
        X_train=None,
        device=None,
    ) -> None:
        assert n_num_features or n_bin_features or cat_cardinalities
        if num_embeddings is not None:
            assert n_num_features

        # To support new backbone:
        # - add it to the list below
        # - add it to lib.deep._CUSTOM_MODULES
        # - set the appropriate value for self.flat (see below)
        assert backbone['type'] in ['MLP', 'ResNet', 'SNN']
        super().__init__()

        if num_embeddings is None:
            self.m_num = nn.Identity() if n_num_features else None
            d_num = n_num_features
        else:
            if num_embeddings['type'] == 'PieceWiseEmbeddings':
                X_train_ss = X_train[torch.rand(len(X_train)) < 1.0]
                self.m_num = lib.make_module(num_embeddings, X_train=X_train_ss.to(device))
            else:
                self.m_num = lib.make_module(num_embeddings, n_features=n_num_features)
            d_num = n_num_features * num_embeddings['d_embedding']
        self.m_bin = nn.Identity() if n_bin_features else None
        self.m_cat = lib.deep.OneHotEncoding0d(cat_cardinalities) if cat_cardinalities else None
        backbone.pop("type")
        self.backbone = MLP(
            **backbone,
            d_in=d_num + n_bin_features + sum(cat_cardinalities),
            d_out=lib.deep.get_d_out(n_classes),
        )
        # For Transformer-like backbones, which operate over feature embeddings,
        # self.flat must be False.
        # For simple "flat" models (e.g. MLP, ResNet, etc.), self.flat must be True.
        self.flat = True

    def forward(
        self,
        *,
        x_num : None | Tensor = None, x_bin : None | Tensor = None, x_cat : None | Tensor = None) -> Tensor:
        x = []
        for module, x_ in [
            (self.m_num, x_num),
            (self.m_bin, x_bin),
            (self.m_cat, x_cat),
        ]:
            if x_ is None:
                assert module is None
            else:
                assert module is not None
                x.append(module(x_))
        del x_  # type: ignore[code]
        if self.flat:
            x = torch.cat([x_.flatten(1, -1) for x_ in x], dim=1)
        else:
            # for Transformer-like backbones
            assert all(x_.ndim == 3 for x_ in x)
            x = torch.cat(x, dim=1)

        x = self.backbone(x)
        return x


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

class MyLoss(nn.Module):
    def __init__(self, lf, times):
        super().__init__()
        self.lf = lf
        self.times = times
        n_parts = 9
        self.n_parts = n_parts
        times_sorted = self.times[torch.argsort(self.times)]
        self.fold_by_id = torch.zeros_like(self.times)
        for i in range(n_parts):
            thr_low = times_sorted[(len(times_sorted) + n_parts - 1) // n_parts * i]
            thr_high = times_sorted[min(len(times_sorted) - 1, (len(times_sorted) + n_parts - 1) // n_parts * (i + 1))]
            self.fold_by_id[(self.times <= thr_high) & (self.times > thr_low)] = i
    def get_covariance_matrices(self, vectors):
        vectors2 = (vectors - vectors.mean(0))
        return (vectors2.unsqueeze(-1) @ vectors2.unsqueeze(-2)).sum(0) / (vectors2.shape[0] - 1)
    def forward(self, xtuple, ys, ids):
        loss_basic = self.lf(xtuple[0].squeeze(-1).float(), ys)
        cov_matrices = []
        n_domains = (self.n_parts + 1) // 2
        for i in range((self.n_parts + 1) // 2):
            id_dom = (self.fold_by_id[ids] < i + n_domains) & (self.fold_by_id[ids] >= i)
            if sum(id_dom) > 1:
                #print(xtuple[1].shape)
                cov_matrices.append(self.get_covariance_matrices(xtuple[1][id_dom]))
        base_cov = self.get_covariance_matrices(xtuple[1])
        for i in range(len(cov_matrices)):
            loss_basic += 1 / 4 / xtuple[1].shape[-1]**2 * (base_cov - cov_matrices[i]).square().sum() / len(cov_matrices)
        return loss_basic


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
        n_classes=dataset.task.try_compute_n_classes(),
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
    loss_fn = MyLoss(lib.deep.get_loss_fn(dataset.task.type_), dataset['x_meta']['train'][:, 0])
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
                                apply_model(part, idx)[0]
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
                lambda idx: loss_fn(apply_model('train', idx), Y_train[idx], idx),
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
