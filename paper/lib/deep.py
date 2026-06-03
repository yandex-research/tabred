import itertools
import math
import warnings
from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import delu
import rtdl_num_embeddings
import rtdl_revisiting_models
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Parameter

from .util import TaskType, is_oom_exception


# ======================================================================================
# >>> modules <<<
# ======================================================================================
def _init_rsqrt_uniform_(weight: Tensor, dim: None | int, d: None | int = None) -> None:
    if d is None:
        assert dim is not None
        d = weight.shape[dim]
    else:
        assert dim is None
    d_rsqrt = 1 / math.sqrt(d)
    nn.init.uniform_(weight, -d_rsqrt, d_rsqrt)


class CLSEmbedding(nn.Module):
    def __init__(self, d_embedding: int) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.weight.shape[-1] ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)

    def forward(self, batch_dims: tuple[int]) -> Tensor:
        if not batch_dims:
            raise ValueError('The input must be non-empty')

        return self.weight.expand(*batch_dims, 1, -1)


class OneHotEncoding0d(nn.Module):
    # Input:  (*, n_cat_features=len(cardinalities))
    # Output: (*, sum(cardinalities))
    cardinalities: torch.IntTensor

    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self.register_buffer('cardinalities', torch.tensor(cardinalities))

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 1
        encoded_columns = [
            F.one_hot(x[..., column], int(cardinality) + 1)[:, :-1] # we want to have all zeros for unknown values
            for column, cardinality in zip(range(x.shape[-1]), self.cardinalities)
        ]

        return torch.cat(encoded_columns, -1)


class CategoricalEmbeddings1d(nn.Module):
    # Input:  (*, n_cat_features=len(cardinalities))
    # Output: (*, n_cat_features, d_embedding)
    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            # [nn.Embedding(c, d_embedding) for c in cardinalities]
            # NOTE: `+ 1` is here to support unknown values that are expected to have
            # the value `max-known-category + 1`.
            # This is not a good way to handle unknown values. This is just a quick
            # hack to stop failing on some datasets.
            [nn.Embedding(c + 1, d_embedding) for c in cardinalities]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.embeddings:
            _init_rsqrt_uniform_(m.weight, -1)  # type: ignore[code]

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 1
        return torch.stack(
            [m(x[..., i]) for i, m in enumerate(self.embeddings)], dim=-2
        )


# The implementation details related to the "weight" parameter are inspired by:
# https://github.com/openai/glow/blob/1f1352977cb1b21c7c0aa83b08efb24dfc216663/tfops.py#L141
# class ActNorm(nn.Module):
#     def __init__(self, d: int) -> None:
#         super().__init__()
#         self.ready = False
#         self.weight = Parameter(torch.empty(d))
#         self.bias = Parameter(torch.empty(d))
#         self.eps = 1e-5
#         self.logscale_factor = 3.0

#     def get_extra_state(self) -> dict:
#         return {'ready': self.ready}

#     def set_extra_state(self, state: dict):
#         self.ready = state['ready']

#     def forward(self, x: Tensor) -> Tensor:
#         if not self.ready:
#             with torch.inference_mode():
#                 self.weight.copy_(
#                     torch.log(1 / (x.std(0) + self.eps)) / self.logscale_factor
#                 )
#                 self.bias.copy_(-x.mean(0))
#         return (x + self.bias) * torch.exp(self.weight * self.logscale_factor)


class Mean(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=self.dim)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.norm(dim=-1, keepdim=True) * x.shape[-1] ** -0.5
        x = x / (rms + self.eps)
        x = self.weight * x
        return x



class ResNetNoNorm(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_out: None | int,
        n_blocks: int,
        d_block: int,
        d_hidden: None | int = None,
        d_hidden_multiplier: None | float,
        dropout1: float,
        dropout2: float,
    ) -> None:
        assert n_blocks > 0
        assert (d_hidden is None) != (d_hidden_multiplier is None)
        if d_hidden is None:
            d_hidden = int(d_block * cast(float, d_hidden_multiplier))

        super().__init__()
        self.input_projection = nn.Linear(d_in, d_block)
        self.blocks = nn.ModuleList(
            [
                named_sequential(
                    ('linear1', nn.Linear(d_block, d_hidden)),
                    ('activation', nn.ReLU()),
                    ('dropout1', nn.Dropout(dropout1)),
                    ('linear2', nn.Linear(d_hidden, d_block)),
                    ('dropout2', nn.Dropout(dropout2)),
                )
                for _ in range(n_blocks)
            ]
        )
        self.output = (
            None
            if d_out is None
            else named_sequential(
                ('activation', nn.ReLU()),
                ('linear', nn.Linear(d_block, d_out)),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_projection(x)
        for block in self.blocks:
            x = x + block(x)
        if self.output is not None:
            x = self.output(x)
        return x


_CUSTOM_MODULES = {
    # https://docs.python.org/3/library/stdtypes.html#definition.__name__
    CustomModule.__name__: CustomModule
    for CustomModule in [
        rtdl_revisiting_models.LinearEmbeddings,
        rtdl_revisiting_models.MLP,
        rtdl_revisiting_models.ResNet,
        rtdl_revisiting_models.FTTransformer,
        rtdl_revisiting_models.FTTransformerBackbone,
        rtdl_num_embeddings.LinearReLUEmbeddings,
        rtdl_num_embeddings.PeriodicEmbeddings,
        rtdl_num_embeddings.PiecewiseLinearEncoding,
        rtdl_num_embeddings.PiecewiseLinearEmbeddings,
        ResNetNoNorm,
    ]
}


def register_module(name: str, factory: Callable[..., nn.Module]) -> None:
    if name in _CUSTOM_MODULES:
        warnings.warn(f'The module "{name}" is already registered.')
    else:
        _CUSTOM_MODULES[name] = factory


def make_module(type: str, *args, **kwargs) -> nn.Module:
    Module = getattr(nn, type, None)
    if Module is None:
        Module = _CUSTOM_MODULES[type]
    return Module(*args, **kwargs)


def named_sequential(*modules) -> nn.Sequential:
    return nn.Sequential(OrderedDict(modules))


def get_n_parameters(m: nn.Module):
    return sum(x.numel() for x in m.parameters() if x.requires_grad)


def get_d_out(n_classes: None | int) -> int:
    return 1 if n_classes is None or n_classes == 2 else n_classes


_GRADIENT_STATISTICS: dict[str, Callable[[Tensor], Tensor]] = {
    'norm': torch.norm,
    'absmax': lambda x: x.abs().max(),
    'absmedian': lambda x: x.abs().median(),
}


@torch.inference_mode()
def compute_gradient_stats(module: nn.Module) -> dict[str, dict[str, float]]:
    named_grads = [
        (n, p.grad) for n, p in module.named_parameters() if p.grad is not None
    ]
    stats = {
        key: {n: fn(g).item() for n, g in named_grads}
        for key, fn in _GRADIENT_STATISTICS.items()
    }
    stats['norm']['model'] = (
        torch.cat([x.flatten() for _, x in named_grads]).norm().item()
    )
    return stats


@torch.inference_mode()
def compute_parameter_stats(module: nn.Module) -> dict[str, dict[str, float]]:
    stats = {'norm': {}, 'gradnorm': {}, 'gradratio': {}}
    for name, parameter in module.named_parameters():
        stats['norm'][name] = parameter.norm().item()
        if parameter.grad is not None:
            stats['gradnorm'][name] = parameter.grad.norm().item()
            # Avoid computing statistics for zero-initialized parameters.
            if (parameter.abs() > 1e-6).any():
                stats['gradratio'][name] = (
                    (parameter.grad.abs() / parameter.abs().clamp_min_(1e-6))
                    .mean()
                    .item()
                )
    stats['norm']['model'] = (
        torch.cat([x.flatten() for x in module.parameters()]).norm().item()
    )
    stats['gradnorm']['model'] = (
        torch.cat([x.grad.flatten() for x in module.parameters() if x.grad is not None])
        .norm()
        .item()
    )
    return stats


# ======================================================================================
# >>> optimization <<<
# ======================================================================================
def default_zero_weight_decay_condition(
    module_name: str, module: nn.Module, parameter_name: str, parameter: Parameter
):
    from rtdl_num_embeddings import _Periodic

    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        nn.BatchNorm1d
        | nn.LayerNorm
        | nn.InstanceNorm1d
        | rtdl_revisiting_models.LinearEmbeddings
        | rtdl_num_embeddings.LinearEmbeddings
        | rtdl_num_embeddings.LinearReLUEmbeddings
        | _Periodic
    )


def make_parameter_groups(
    module: nn.Module,
    zero_weight_decay_condition=default_zero_weight_decay_condition,
    custom_groups: None | list[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    if custom_groups is None:
        custom_groups = []
    custom_params = frozenset(
        itertools.chain.from_iterable(group['params'] for group in custom_groups)
    )
    assert len(custom_params) == sum(
        len(group['params']) for group in custom_groups
    ), 'Parameters in custom_groups must not intersect'
    zero_wd_params = frozenset(
        p
        for mn, m in module.named_modules()
        for pn, p in m.named_parameters()
        if p not in custom_params and zero_weight_decay_condition(mn, m, pn, p)
    )
    default_group = {
        'params': [
            p
            for p in module.parameters()
            if p not in custom_params and p not in zero_wd_params
        ]
    }
    return [
        default_group,
        {'params': list(zero_wd_params), 'weight_decay': 0.0},
        *custom_groups,
    ]


def make_optimizer(
    type: str, **kwargs
) -> torch.optim.Optimizer:
    return getattr(torch.optim, type)(**kwargs)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return next(iter(optimizer.param_groups))['lr']


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group['lr'] = lr


# ======================================================================================
# >>> training <<<
# ======================================================================================
def get_loss_fn(task_type: TaskType, **kwargs) -> Callable[..., Tensor]:
    loss_fn = (
        F.binary_cross_entropy_with_logits
        if task_type == TaskType.BINCLASS
        else F.cross_entropy
        if task_type == TaskType.MULTICLASS
        else F.mse_loss
    )
    return partial(loss_fn, **kwargs) if kwargs else loss_fn  # type: ignore[return-value,arg-type]


def zero_grad_forward_backward(
    optimizer: torch.optim.Optimizer,
    step_fn: Callable[[Tensor], Tensor],  # step_fn: chunk_idx -> loss
    batch_idx: Tensor,
    chunk_size: int,
) -> tuple[Tensor, int]:
    batch_size = len(batch_idx)
    loss = None
    while chunk_size != 0:
        optimizer.zero_grad()

        try:
            if batch_size <= chunk_size:
                # The simple forward-backward.
                loss = step_fn(batch_idx)
                loss.backward()
            else:
                # Forward-backward by chunks.
                # Mathematically, this is equivalent to the simple forward-backward.
                # Technically, this implementations uses less memory.
                loss = None
                for chunk_idx in batch_idx.split(chunk_size):
                    chunk_loss = step_fn(chunk_idx)
                    chunk_loss = chunk_loss * (len(chunk_idx) / batch_size)
                    chunk_loss.backward()
                    if loss is None:
                        loss = chunk_loss.detach()
                    else:
                        loss += chunk_loss.detach()
        except RuntimeError as err:
            if not is_oom_exception(err):
                raise
            delu.cuda.free_memory()
            chunk_size //= 2

        else:
            break

    if not chunk_size:
        raise RuntimeError('Not enough memory even for chunk_size=1')
    return cast(Tensor, loss), chunk_size
