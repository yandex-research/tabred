# XGBoost.

import os
from pathlib import Path

import delu
import numpy as np
import pandas as pd
from loguru import logger
from typing_extensions import TypedDict
from xgboost import XGBClassifier, XGBRegressor

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
    assert 'early_stopping_rounds' in config['model'], (
        'XGBoost does not automatically use the best model,'
        ' so early stopping must be used'
    )
    if config['model'].get('tree_method') == 'gpu_hist' or config['model'].get('device') == 'cuda':
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
                    pd.DataFrame(
                        dataset['x_cat'][part].astype('int64'),
                        columns=categorical_features,
                        dtype='category',
                    )
                ], axis=1)
                for part in dataset.parts()
            }
            dataset.data.pop('x_cat')
    else:
        categorical_features = None

    # >>> model
    # NOTE: we use Scikit-Learn API AND we require "early_stopping_rounds" to be
    # explicitely provided. In this case, the "predict" and "predict_proba" methods
    # automatically use the best model.
    model_extra_kwargs = {
        'random_state': config['seed'],
        'enable_categorical': categorical_features is not None,
    }
    if dataset.task.is_regression:
        model = XGBRegressor(**config['model'], **model_extra_kwargs)
        predict = model.predict
    else:
        model = XGBClassifier(
            **config['model'],
            **model_extra_kwargs,
            disable_default_eval_metric=True,
            eval_metric=(
                'merror' if dataset.task.is_multiclass
                else 'auc' if dataset.task.score == lib.Score.ROC_AUC
                else "error"
            ),
        )
        if dataset.task.is_multiclass:
            predict = model.predict_proba
        else:
            predict = lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]  # noqa

    report['prediction_type'] = 'labels' if dataset.task.is_regression else 'probs'

    # >>> training
    logger.info('Training...')
    with delu.Timer() as timer:
        model.fit(
            X['train'],
            dataset['y']['train'],
            eval_set=[(X['val'], dataset['y']['val'])],
            **config['fit'],
        )
    report['time'] = str(timer)
    report['best_iteration'] = model.best_iteration

    # >>> finish
    model.save_model(str(output / 'model.ubj'))
    np.save(output / 'feature_importances.npy', model.feature_importances_)
    predictions = {k: predict(v) for k, v in X.items()}
    if regression_label_stats is not None:
        predictions = {
            k: v * regression_label_stats.std + regression_label_stats.mean
            for k, v in predictions.items()
        }
    report['metrics'] = dataset.task.calculate_metrics(
        predictions, report['prediction_type']
    )
    lib.dump_predictions(output, predictions)
    lib.dump_summary(output, lib.summarize(report))
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_MainFunction_cli(main)
