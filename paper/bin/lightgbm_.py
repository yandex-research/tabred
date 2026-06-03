# LightGBM

import os
from pathlib import Path
from typing import Any
from typing_extensions import TypedDict

import delu
import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from loguru import logger

import lib
from lib import KWArgs


class Config(TypedDict):
    seed: int
    data: KWArgs
    model: KWArgs
    fit: KWArgs


def _check_config(config: Config):
    assert 'random_state' not in config['model']
    assert config['data'].get('cat_policy') in [None, 'ordinal']
    assert 'stopping_rounds' in config['model'], (
        'We require early-stopping'
    )
    if config['model']['device_type'] == 'gpu':
        assert os.environ.get('CUDA_VISIBLE_DEVICES')
    else:
        assert not os.environ.get('CUDA_VISIBLE_DEVICES')


def main(
    config: Config, output: str | Path, *, force: bool = False
) -> None | lib.JSONDict:
    # >>> start
    assert set(config) >= Config.__required_keys__
    assert set(config) <= Config.__required_keys__ | Config.__optional_keys__
    _check_config(config)
    if not lib.start(output, force=force):
        return None

    lib.show_config(config)  # type: ignore[code]
    output = Path(output)
    delu.random.seed(config['seed'])
    report = lib.create_report(config)  # type: ignore[code]

    # >>> data
    dataset = lib.data.build_dataset(**config['data'])
    if dataset.task.is_regression:
        dataset.data['y'], regression_label_stats = lib.data.standardize_labels(
            dataset.data['y']
        )
    else:
        regression_label_stats = None

    if 'x_num' in dataset:
        X = {
            part: pd.DataFrame(dataset['x_num'][part])
            for part in dataset.parts()
        }
    else:
        X = {
            part: pd.DataFrame() for part in dataset.parts()
        }


    if 'x_bin' in dataset:
        # Merge binary features to continuous features.
        X = {
            part: pd.concat([X[part], pd.DataFrame(dataset['x_bin'][part])], axis=1)
            for part in dataset.parts()
        }
        # Rename columns
        for part in X:
            X[part].columns = range(X[part].shape[1])

    if 'x_cat' in dataset:
        # Merge one-hot-encoded categorical features
        categorical_features = range(dataset.n_num_features + dataset.n_bin_features, dataset.n_features)

        if not 'x_num' in dataset:
            X = {
                k: pd.DataFrame(v) for k, v in dataset.data.pop('x_cat').items()
            }
        else:
            X = {
                part: pd.concat([
                    X[part],
                    pd.DataFrame(dataset['x_cat'][part], columns=categorical_features)
                ], axis=1)
                for part in dataset.parts()
            }
            dataset.data.pop('x_cat')
    else:
        categorical_features = None

    # >>> model
    fit_extra_kwargs: KWArgs = {}
    stopping_rounds = config['model'].pop('stopping_rounds')
    fit_extra_kwargs['callbacks'] = [lightgbm.early_stopping(stopping_rounds=stopping_rounds)]

    model_extra_kwargs: dict[str, Any] = {
        'random_state': config['seed'],
        'categorical_features': list(categorical_features) if categorical_features is not None else None,
    }

    if dataset.task.is_regression:
        model = LGBMRegressor(**config['model'], **model_extra_kwargs)
        fit_extra_kwargs = {'eval_metric': 'rmse'}
        predict = model.predict
    else:
        model = LGBMClassifier(**config['model'], **model_extra_kwargs)
        if dataset.task.is_multiclass:
            predict = model.predict_proba
            fit_extra_kwargs = {'eval_metric': 'multi_error'}
        else:
            predict = lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]  # noqa
            fit_extra_kwargs = {'eval_metric': 'auc' if dataset.task.score == lib.Score.ROC_AUC else 'binary_error'}

    report['prediction_type'] = 'labels' if dataset.task.is_regression else 'probs'


    # >>> training
    logger.info('training...')
    with delu.Timer() as timer:
        model.fit(
            X['train'],
            dataset['y']['train'],
            eval_set=[(X['val'], dataset['y']['val'])],
            **config['fit'],
            **fit_extra_kwargs,
        )
    report['time'] = str(timer)
    report['best_iteration'] = model.booster_.best_iteration

    # >>> finish
    # Faster to recreate this if needed, we need more disk-space :/
    # lib.dump_pickle(model, output / 'model.pickle')
    np.save(output / 'feature_importances.npy', model.feature_importances_)
    predictions: dict[str, np.ndarray] = {k: np.asarray(predict(v)) for k, v in X.items()}
    if regression_label_stats is not None:
        predictions = {
            k: v * regression_label_stats.std + regression_label_stats.mean
            for k, v in predictions.items()
        }
    report['metrics'] = dataset.task.calculate_metrics(
        predictions, report['prediction_type']  # type: ignore[code]
    )
    lib.dump_predictions(output, predictions)
    lib.dump_summary(output, lib.summarize(report))
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_MainFunction_cli(main)
