import enum
import hashlib
import json
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import numpy as np
import sklearn.preprocessing
import torch
from loguru import logger
from torch import Tensor

import lib

from .metrics import calculate_metrics as calculate_metrics_
from .util import DataKey, PartKey, PredictionType, Score, TaskType, get_path



_SCORE_SHOULD_BE_MAXIMIZED = {
    Score.ACCURACY: True,
    Score.CROSS_ENTROPY: False,
    Score.MAE: False,
    Score.R2: True,
    Score.RMSE: False,
    Score.ROC_AUC: True,
}


@dataclass(frozen=True)
class Task:
    labels: dict[str, np.ndarray]
    type_: TaskType
    score: Score

    @classmethod
    def from_dir(cls, path: str | Path, split: str = 'default') -> 'Task':
        path = get_path(path)
        info = json.loads(path.joinpath('info.json').read_text())
        task_type = TaskType(info['task_type'])
        score = info.get('score')

        if score is None:
            score = {
                TaskType.BINCLASS: Score.ACCURACY,
                TaskType.MULTICLASS: Score.ACCURACY,
                TaskType.REGRESSION: Score.RMSE,
            }[task_type]
        else:
            score = Score(score)
            
        labels = np.load(path / 'Y.npy')
        return Task(
            {
                part: labels[np.load(path / f'split-{split}/{part}_idx.npy')]
                for part in ['train', 'val', 'test']
            },
            task_type,
            score,
        )

    def __post_init__(self):
        assert isinstance(self.type_, TaskType)
        assert isinstance(self.score, Score)
        if self.is_regression:
            assert all(
                value.dtype in (np.dtype('float32'), np.dtype('float64'))
                for value in self.labels.values()
            ), 'Regression labels must have dtype=float32'
            for key in self.labels:
                self.labels[key] = self.labels[key].astype('float32')

    @property
    def is_regression(self) -> bool:
        return self.type_ == TaskType.REGRESSION

    @property
    def is_binclass(self) -> bool:
        return self.type_ == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.type_ == TaskType.MULTICLASS

    @property
    def is_classification(self) -> bool:
        return self.is_binclass or self.is_multiclass

    def compute_n_classes(self) -> int:
        assert self.is_binclass or self.is_classification
        return len(np.unique(self.labels['train']))

    def try_compute_n_classes(self) -> None | int:
        return None if self.is_regression else self.compute_n_classes()

    def calculate_metrics(
        self,
        predictions: dict[PartKey, np.ndarray],
        prediction_type: str | PredictionType,
    ) -> dict[PartKey, Any]:
        metrics = {
            part: calculate_metrics_(
                self.labels[part], predictions[part], self.type_, prediction_type
            )
            for part in predictions
        }
        for part_metrics in metrics.values():
            part_metrics['score'] = (
                1.0 if _SCORE_SHOULD_BE_MAXIMIZED[self.score] else -1.0
            ) * part_metrics[self.score.value]
        return metrics  # type: ignore[code]


def load_data(path: str | Path, split: str) -> dict[DataKey, dict[PartKey, np.ndarray]]:
    path = get_path(path)
    data = {}
    for key in ['X_num', 'X_bin', 'X_cat', 'X_meta', 'Y']:
        if not path.joinpath(f'{key}.npy').exists():
            print(f'No {key}')
            continue

        arr = np.load(path / f'{key}.npy', allow_pickle=False)
        data[key.lower()] = {
            part: arr[np.load(path / f'split-{split}/{part}_idx.npy')]
            for part in ['train', 'val', 'test']
        }

    return data


T = TypeVar('T', np.ndarray, Tensor)


@dataclass
class Dataset(Generic[T]):
    """Dataset = Data + Task + simple methods for convenience."""

    data: dict[DataKey, dict[PartKey, T]]
    task: Task

    @classmethod
    def from_dir(cls, path: str | Path, split: str = "default") -> 'Dataset[np.ndarray]':
        return Dataset(load_data(path, split), Task.from_dir(path, split))

    def __post_init__(self):
        data = self.data

        # >>> Check data types.
        is_numpy = self._is_numpy()
        for key, allowed_dtypes in {
            'x_num': [np.dtype('float32')] if is_numpy else [torch.float32],
            'x_bin': [np.dtype('float32')] if is_numpy else [torch.float32],
            'x_cat': [] if is_numpy else [torch.int64],
            'y': (
                [np.dtype('float32'), np.dtype('float64'), np.dtype('int64')]
                if is_numpy
                else [torch.float32, torch.int64]
            ),
        }.items():
            if key in data:
                for part, value in data[key].items():
                    if key == 'x_cat' and is_numpy:
                        assert value.dtype in (
                            np.dtype('int32'),
                            np.dtype('int64'),
                        ) or isinstance(
                            value.dtype, np.dtypes.StrDType  # type: ignore[code]
                        )
                    else:
                        assert value.dtype in allowed_dtypes, (
                            f'The value data["{key}"]["{part}"] has dtype'
                            f' {value.dtype}, but it must be one of {allowed_dtypes}'
                        )

        # >>> Check nans.
        isnan = np.isnan if is_numpy else torch.isnan
        for key in ['x_num', 'x_bin']:
            if key in data:  # type: ignore[code]
                for part, value in data['y'].items():
                    assert not isnan(
                        value  # type: ignore[code]
                    ).any(), f'data["{key}"]["{part}"] contains nans'
        for part, value in data['y'].items():
            assert not isnan(value).any(), f'data["{key}"]["{part}"] contains nans'  # type: ignore[code]

    def _is_numpy(self) -> bool:
        return isinstance(self.data['y']['train'], np.ndarray)

    def __contains__(self, key: DataKey) -> bool:
        return key in self.data

    def __getitem__(self, key: DataKey) -> dict[PartKey, T]:
        return self.data[key]

    def __setitem__(self, key: DataKey, value: dict[PartKey, T]) -> None:
        self.data[key] = value

    @property
    def n_num_features(self) -> int:
        return self.data['x_num']['train'].shape[1] if 'x_num' in self.data else 0

    @property
    def n_bin_features(self) -> int:
        return self.data['x_bin']['train'].shape[1] if 'x_bin' in self.data else 0

    @property
    def n_cat_features(self) -> int:
        return self.data['x_cat']['train'].shape[1] if 'x_cat' in self.data else 0

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_bin_features + self.n_cat_features

    def size(self, part: None | PartKey) -> int:
        return (
            sum(map(len, self.data['y'].values()))
            if part is None
            else len(self.data['y'][part])
        )

    def parts(self) -> Iterable[PartKey]:
        return self.data['y'].keys()

    def compute_cat_cardinalities(self) -> list[int]:
        x_cat = self.data.get('x_cat')
        if x_cat is None:
            return []
        unique = np.unique if self._is_numpy() else torch.unique
        return (
            []
            if x_cat is None
            else [len(unique(column)) for column in x_cat['train'].T]
        )

    def to_torch(self, device: None | str | torch.device) -> 'Dataset[Tensor]':
        return Dataset(
            {
                key: {
                    part: torch.as_tensor(value).to(device)
                    for part, value in self.data[key].items()
                }
                for key in self.data
            },
            self.task,
        )


class NumPolicy(enum.Enum):
    STANDARD = 'standard'
    IDENTITY = 'identity'
    NOISY_QUANTILE = 'noisy-quantile'
    NOISY_QUANTILE_OLD = 'noisy-quantile-old'


# Inspired by: https://github.com/Yura52/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
def transform_num(
    X_num: dict[PartKey, np.ndarray], policy: str | NumPolicy, seed: None | int
) -> dict[PartKey, np.ndarray]:
    policy = NumPolicy(policy)
    X_num_train = X_num['train']
    if policy == NumPolicy.STANDARD:
        normalizer = sklearn.preprocessing.StandardScaler()
    elif policy == NumPolicy.NOISY_QUANTILE:
        normalizer = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=max(min(X_num['train'].shape[0] // 30, 1000), 10),
            output_distribution='normal',
            subsample=1_000_000_000,
            random_state=seed,
        )
        assert seed is not None
        X_num_train = X_num_train + np.random.RandomState(seed).normal(
            0.0, 1e-5, X_num_train.shape
        ).astype(X_num_train.dtype)
    elif policy == NumPolicy.NOISY_QUANTILE_OLD:
        normalizer = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=max(min(X_num['train'].shape[0] // 30, 1000), 10),
            output_distribution='normal',
            subsample=1_000_000_000,
            random_state=seed,
        )
        assert seed is not None
        stds = np.std(X_num_train, axis=0, keepdims=True)
        noise_std = 1e-3 / np.maximum(stds, 1e-3)  # type: ignore[code]
        X_num_train = X_num_train + noise_std * np.random.default_rng(
            seed
        ).standard_normal(X_num_train.shape)
    elif policy == NumPolicy.IDENTITY:
        normalizer = sklearn.preprocessing.FunctionTransformer()
    else:
        raise ValueError(f'Unknown policy={policy}')

    normalizer.fit(X_num_train)
    # return {k: normalizer.transform(v) for k, v in X_num.items()}  # type: ignore[code]
    return {
        # NOTE: this is not a good way to process NaNs.
        # This is just a quick hack to stop failing on some datasets.
        # NaNs are replaced with zeros (zero is the mean value for all features after
        # the conventional preprocessing techniques.
        k: np.nan_to_num(normalizer.transform(v)).astype(np.float32)  # type: ignore[code]
        for k, v in X_num.items()
    }


class CatPolicy(enum.Enum):
    ORDINAL = 'ordinal'
    ONE_HOT = 'one-hot'


def transform_cat(
    X_cat: dict[PartKey, np.ndarray], policy: str | CatPolicy
) -> dict[PartKey, np.ndarray]:
    policy = CatPolicy(policy)

    # The first step is always the ordinal encoding,
    # even for the one-hot encoding.
    unknown_value = np.iinfo('int64').max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value',  # type: ignore[code]
        unknown_value=unknown_value,  # type: ignore[code]
        dtype='int64',  # type: ignore[code]
    ).fit(X_cat['train'])
    X_cat = {k: encoder.transform(v) for k, v in X_cat.items()}
    max_values = X_cat['train'].max(axis=0)
    for part in ['val', 'test']:
        part = cast(PartKey, part)
        for column_idx in range(X_cat[part].shape[1]):
            X_cat[part][X_cat[part][:, column_idx] == unknown_value, column_idx] = (
                max_values[column_idx] + 1
            )

    # >>> encode
    if policy == CatPolicy.ORDINAL:
        return X_cat
    elif policy == CatPolicy.ONE_HOT:
        encoder = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore', sparse_output=False, dtype=np.float32  # type: ignore[code]
        )
        encoder.fit(X_cat['train'])
        return {k: cast(np.ndarray, encoder.transform(v)) for k, v in X_cat.items()}
    else:
        raise ValueError(f'Unknown policy={policy}')


@dataclass(frozen=True, kw_only=True)
class RegressionLabelStats:
    mean: float
    std: float


def standardize_labels(
    y: dict[PartKey, np.ndarray]
) -> tuple[dict[PartKey, np.ndarray], RegressionLabelStats]:
    assert y['train'].dtype == np.dtype('float32')
    mean = float(y['train'].mean())
    std = float(y['train'].std())
    return {k: (v - mean) / std for k, v in y.items()}, RegressionLabelStats(
        mean=mean, std=std
    )


def build_dataset(
    *,
    path: str | Path,
    num_policy: None | str | NumPolicy = None,
    cat_policy: None | str | CatPolicy = None,
    seed: int = 0,
    split: str = "default",
    cache: bool = False,
) -> Dataset[np.ndarray]:
    path = get_path(path)
    if cache:
        args = locals()
        args.pop('cache')
        args.pop('path')
        cache_path = lib.CACHE_DIR / (
            f'build_dataset__{path.name}__{hashlib.md5(str(args).encode("utf-8")).hexdigest()}.pickle'
        )
        if cache_path.exists():
            cached_args, cached_value = pickle.loads(cache_path.read_bytes())
            assert args == cached_args, f'Hash collision for {cache_path}'
            logger.info(f"Using cached dataset: {cache_path.name}")
            return cached_value
    else:
        args = None
        cache_path = None

    dataset = Dataset.from_dir(path, split)
    if num_policy is not None:
        dataset['x_num'] = transform_num(dataset['x_num'], num_policy, seed)
    if cat_policy is not None:
        dataset['x_cat'] = transform_cat(dataset['x_cat'], cat_policy)

    if cache_path is not None:
        cache_path.write_bytes(pickle.dumps((args, dataset)))
    return dataset
