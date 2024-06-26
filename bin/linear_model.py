from sklearn.linear_model import SGDClassifier, SGDRegressor
import os
import json
import sys
import optuna
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import os.path

import lib

from lib.data import build_dataset
from lib.data import standardize_labels

def main(dataset_name):
    with open(f'data/{dataset_name}/info.json', 'rt') as fin:
        info_d = (json.load(fin))
    if info_d['task_type'] == 'binclass':
        target_policy = 'binclass'
        y_policy = None
        module = SGDClassifier
    else:
        target_policy = 'regression'
        y_policy = "standard"
        module = SGDRegressor

    if not(os.path.exists(f'data/{dataset_name}/X_cat.npy')):
        cat_policy = None
    else:
        cat_policy = 'one-hot'

    dataset = build_dataset(path=f':data/{dataset_name}', cat_policy=cat_policy, num_policy='noisy-quantile', cache=True, seed=0)

    if (dataset.n_bin_features != 0):
        train_x = np.concatenate((dataset['x_num']['train'], dataset['x_bin']['train']), axis=1)
        val_x = np.concatenate((dataset['x_num']['val'], dataset['x_bin']['val']), axis=1)
        test_x = np.concatenate((dataset['x_num']['test'], dataset['x_bin']['test']), axis=1)
    else:
        train_x = dataset['x_num']['train']
        val_x = dataset['x_num']['val']
        test_x = dataset['x_num']['test']
    if cat_policy == 'one-hot':
        train_x = np.concatenate((train_x, dataset['x_cat']['train']), axis=1)
        val_x = np.concatenate((val_x, dataset['x_cat']['val']), axis=1)
        test_x = np.concatenate((test_x, dataset['x_cat']['test']), axis=1)
    if dataset.task.is_regression:
        dataset.data['y'], regression_label_stats = standardize_labels(
            dataset.data['y']
        )

    train_y = dataset['y']['train']
    val_y = dataset['y']['val']
    test_y = dataset['y']['test']

    def objective(trial):

        penalty = 'elasticnet'

        max_iter = 10000

        alpha = trial.suggest_loguniform(name = "alpha", low=1e-5, high=1e-2)

        power_t = trial.suggest_loguniform(name="power_t", low=1e-2, high=4.0)

        l1_ratio = trial.suggest_uniform(name="l1_ratio", low=0.0, high=1.0)

        params = {
            "penalty": penalty,
            "max_iter": max_iter,
            "alpha": alpha,
            "power_t": power_t,
            "l1_ratio": l1_ratio,
        }

        if target_policy != 'regression':
            params['loss'] = 'log_loss'
        model = module(random_state=0, **params)

        model.fit(train_x, train_y)
        #cv_score = cross_val_score(model, X_train, y_train, n_jobs=4, cv=5)
        #mean_cv_accuracy = cv_score.mean()
        if target_policy == 'regression':
            y_pred = model.predict(val_x)
        else:
            y_pred = model.predict_proba(val_x)[:, 1]
        if target_policy == 'regression':
            score = ((y_pred - val_y)**2).mean()**0.5
        else:
            score = -roc_auc_score(val_y, y_pred)
        return score

    study = optuna.create_study()
    study.optimize(objective, n_trials=25)

    dd = study.best_params
    dd['penalty'] = 'elasticnet'
    dd['max_iter'] = 10000
    if target_policy != 'regression':
        dd['loss'] = 'log_loss'

    tt = time.time()

    os.mkdir(f'exp/linear_model/{dataset_name}')
    os.mkdir(f'exp/linear_model/{dataset_name}/0-evaluation')

    for seed in range(15):
        preds = dict()
        best_model = module(random_state=seed, **dd)
        best_model.fit(train_x, train_y)
        if target_policy == "regression":
            train_y_pred = best_model.predict(train_x)
            val_y_pred = best_model.predict(val_x)
            y_pred = best_model.predict(test_x)
        else:
            train_y_pred = best_model.predict_proba(train_x)[:, 1]
            val_y_pred = best_model.predict_proba(val_x)[:, 1]
            y_pred = best_model.predict_proba(test_x)[:, 1]

        if target_policy == 'regression':
            val_score = ((val_y_pred - val_y)**2).mean()**0.5 * regression_label_stats.std
        else:
            val_score = -roc_auc_score(val_y, val_y_pred)
        if target_policy == 'regression':
            test_score = ((y_pred - test_y)**2).mean()**0.5 * regression_label_stats.std
        else:
            test_score = -roc_auc_score(test_y, y_pred)
        j = dict()
        j['config'] = dict()
        j['config']['model'] = dd
        j['config']['data'] = dict()
        j['config']['data']['seed'] = seed
        j['config']['data']['path'] = f':data/{dataset_name}'
        j['metrics'] = dict()
        j['metrics']['val'] = dict()
        j['metrics']['test'] = dict()
        j['metrics']['val']['score'] = -val_score
        j['metrics']['test']['score'] = -test_score
        j['time'] = time.time() - tt
        tt = time.time()
        os.mkdir(f'exp/linear_model/{dataset_name}/0-evaluation/{seed}')
        if target_policy == 'regression':
            np.savez(f'exp/linear_model/{dataset_name}/0-evaluation/{seed}/predictions.npz', 
            test=y_pred * regression_label_stats.std + regression_label_stats.mean,
            val=val_y_pred * regression_label_stats.std + regression_label_stats.mean,
            train=train_y_pred * regression_label_stats.std + regression_label_stats.mean)
        else:
            np.savez(f'exp/linear_model/{dataset_name}/0-evaluation/{seed}/predictions.npz', 
            test=y_pred, val=val_y_pred, train=train_y_pred)

        with open(f'exp/linear_model/{dataset_name}/0-evaluation/{seed}/report.json', 'wt') as f:
            json.dump(j, f, indent=4)
        
        output = lib.get_path(f'exp/linear_model/{dataset_name}/0-evaluation')
        lib.finish(output, j)

main(sys.argv[1])
