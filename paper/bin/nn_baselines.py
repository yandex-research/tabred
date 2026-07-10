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


class SNN(nn.Module):
    """SNN from 'Self-Normalizing Neural Networks'."""

    def __init__(
        self, *, d_in: int, d_out: int, n_blocks: int, d_block: int, dropout: float
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_block if i else d_in, d_block),
                    nn.SELU(),
                    nn.AlphaDropout(dropout),
                )
                for i in range(n_blocks)
            ]
        )
        self.output = nn.Linear(d_block, d_out)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # The initialization follows the official implementation:
                # https://github.com/bioinf-jku/SNNs/blob/b578499301fcb801f8d4135dbd7cebb246722bfc/Pytorch/SelfNormalizingNetworks_MLP_MNIST.ipynb
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_in', nonlinearity='linear'
                )
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        if self.output is not None:
            x = self.output(x)
        return x


class DCNv2(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_deep: int,
        d_out: int,
        n_cross_layers: int,
        n_deep_layers: int,
        dropout_p: float,
        k_low_rank_cross: int | None = None,
        nonlin_cross: bool = False,
    ):
        super().__init__()

        def get_cross_layer():
            "Cross layer variations from the paper, no MoE variation"

            if k_low_rank_cross is None:
                m = nn.Linear(d_in, d_in, bias=True)
                torch.nn.init.zeros_(m.bias)
            elif nonlin_cross:
                d_low_rank_cross = d_in // k_low_rank_cross
                m = nn.Sequential(
                    nn.Linear(d_in, d_low_rank_cross, bias=False),
                    nn.ReLU(True),
                    nn.Linear(d_low_rank_cross, d_low_rank_cross, bias=False),
                    nn.ReLU(True),
                    nn.Linear(d_low_rank_cross, d_in),
                )
                torch.nn.init.zeros_(m[-1].bias)
            else:
                d_low_rank_cross = d_in // k_low_rank_cross
                m = nn.Sequential(
                    nn.Linear(d_in, d_low_rank_cross, bias=False),
                    nn.Linear(d_low_rank_cross, d_in),
                )
                torch.nn.init.zeros_(m[-1].bias)
            return m

        self.cross_layers = nn.ModuleList(
            [get_cross_layer() for _ in range(n_cross_layers)]
        )

        def get_dnn_layer(d_in=None):
            return nn.Sequential(
                nn.Linear(d_in if d_in is not None else d_deep, d_deep),
                nn.ReLU(True),
                nn.Dropout(dropout_p),
            )

        self.deep_layers = nn.Sequential(
            *[
                get_dnn_layer(d_in=d_in if i == 0 else None)
                for i in range(n_deep_layers)
            ]
        )
        self.head = nn.Linear(d_deep, d_out)

    def forward(self, x):
        x0 = x
        for c in self.cross_layers:
            x = x0 * c(x)

        x = self.deep_layers(x)
        x = self.head(x)

        return x

lib.deep.register_module(DCNv2.__name__, DCNv2)
lib.deep.register_module(SNN.__name__, SNN)


class Model(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: None | int,
        bins: None | list[Tensor],
        num_embeddings: None | dict = None,
        backbone: dict,
    ) -> None:
        assert n_num_features or n_bin_features or cat_cardinalities
        super().__init__()

        self.flat = backbone['type'] != 'FTTransformerBackbone'


        if not self.flat and n_bin_features > 0:
            self.m_bin = lib.deep.make_module("LinearEmbeddings", n_bin_features, backbone['d_block'])
        else:
            self.m_bin = None

        if num_embeddings is None:
            assert bins is None
            self.m_num = None
            d_num = n_num_features
        else:
            if not self.flat:
                num_embeddings['d_embedding'] = backbone['d_block']

            assert n_num_features > 0
            if num_embeddings['type'] in (
                rtdl_num_embeddings.PiecewiseLinearEmbeddings.__name__,
                rtdl_num_embeddings.PiecewiseLinearEncoding.__name__,
            ):
                assert bins is not None
                self.m_num = lib.deep.make_module(**num_embeddings, bins=bins)
                d_num = (
                    sum(len(x) - 1 for x in bins)
                    if num_embeddings['type'].startswith(
                        rtdl_num_embeddings.PiecewiseLinearEncoding.__name__
                    )
                    else n_num_features * num_embeddings['d_embedding']
                )
            else:
                assert bins is None
                self.m_num = lib.deep.make_module(
                    **num_embeddings, n_features=n_num_features
                )
                d_num = n_num_features * num_embeddings['d_embedding']

        if backbone['type'] in ['DCNv2', 'FTTransformerBackbone']:
            d_cat_embedding = backbone.pop('d_cat_embedding') if self.flat else backbone['d_block']
            self.m_cat = (
                lib.deep.CategoricalEmbeddings1d(cat_cardinalities, d_cat_embedding)
                if cat_cardinalities
                else None
            )
            d_cat = len(cat_cardinalities) * d_cat_embedding
        else:
            self.m_cat = (
                lib.deep.OneHotEncoding0d(cat_cardinalities)
                if cat_cardinalities
                else None
            )
            d_cat = sum(cat_cardinalities)

    
        if self.flat:
            backbone['d_in'] = d_num + n_bin_features + d_cat
        else:
            self.cls_embedding = lib.deep.CLSEmbedding(backbone['d_block'])
        

        self.backbone = lib.deep.make_module(
            **backbone,
            d_out=lib.deep.get_d_out(n_classes),
        )
        
        

    def forward(
        self,
        *,
        x_num: None | Tensor = None,
        x_bin: None | Tensor = None,
        x_cat: None | Tensor = None,
    ) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num if self.m_num is None else self.m_num(x_num))
        if x_bin is not None:
            x.append(x_bin if self.m_bin is None else self.m_bin(x_bin))
        if x_cat is None:
            assert self.m_cat is None
        else:
            assert self.m_cat is not None
            x.append(
                self.m_cat(x_cat).flatten(-2)
                if isinstance(self.backbone, DCNv2)
                else self.m_cat(x_cat)
            )
        
        if self.flat:
            x = torch.column_stack([x_.flatten(1, -1) for x_ in x])
        else:
            x = torch.cat([self.cls_embedding(x[0].shape[:1])] + x, dim=1)

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
        bins=bin_edges,
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
        ).squeeze(-1)

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
                                apply_model(part, idx)
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
                lambda idx: loss_fn(apply_model('train', idx), Y_train[idx]),
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
