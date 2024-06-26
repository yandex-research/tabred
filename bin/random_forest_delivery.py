from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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

n_jobs = 64

def main(dataset_name):
    with open(f'data/{dataset_name}/info.json', 'rt') as fin:
        info_d = (json.load(fin))
    if info_d['task_type'] == 'binclass':
        target_policy = 'binclass'
        y_policy = None
        module = RandomForestClassifier
    else:
        target_policy = 'regression'
        y_policy = "standard"
        module = RandomForestRegressor

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
        # Number of trees in random forest
        n_estimators = 1000

        # Number of features to consider at every split
        max_features = trial.suggest_categorical(name="max_features", choices=['log2', 'sqrt', 1.0, 0.5, 0.25]) 

        # Maximum number of levels in tree
        max_depth = trial.suggest_int(name="max_depth", low=10, high=110, step=20)

        # Minimum number of samples required to split a node
        min_samples_split = trial.suggest_int(name="min_samples_split", low=2, high=10, step=2)

        # Minimum number of samples required at each leaf node
        min_samples_leaf = trial.suggest_int(name="min_samples_leaf", low=1, high=4, step=1)
        
        params = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }
        model = module(random_state=0, **params, n_jobs=n_jobs)
        
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

    #study = optuna.create_study()
    #study.optimize(objective, n_trials=10)

    dd = {'max_features': 'sqrt', 'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 4}
    dd['n_estimators'] = 1000

    tt = time.time()

    os.mkdir(f'exp/random_forest/{dataset_name}')
    os.mkdir(f'exp/random_forest/{dataset_name}/0-evaluation')

    print("Starting", flush=True)
    for seed in range(5):
        preds = dict()
        best_model = module(random_state=seed, **dd, n_jobs=n_jobs)
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
        print(seed, val_score, test_score, flush=True)
        tt = time.time()
        os.mkdir(f'exp/random_forest/{dataset_name}/0-evaluation/{seed}')
        if target_policy == 'regression':
            np.savez(f'exp/random_forest/{dataset_name}/0-evaluation/{seed}/predictions.npz', 
            test=y_pred * regression_label_stats.std + regression_label_stats.mean,
            val=val_y_pred * regression_label_stats.std + regression_label_stats.mean,
            train=train_y_pred * regression_label_stats.std + regression_label_stats.mean)
        else:
            np.savez(f'exp/random_forest/{dataset_name}/0-evaluation/{seed}/predictions.npz', 
            test=y_pred, val=val_y_pred, train=train_y_pred)

        with open(f'exp/random_forest/{dataset_name}/0-evaluation/{seed}/report.json', 'wt') as f:
            json.dump(j, f, indent=4)
        
        output = lib.get_path(f'exp/random_forest/{dataset_name}/0-evaluation')
        lib.finish(output, j)

main(sys.argv[1])
