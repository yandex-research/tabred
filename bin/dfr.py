from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import os
import json
import sys
import optuna
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import os.path
from torch import Tensor

_project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.environ['PROJECT_DIR'] = _project_dir
sys.path.append(_project_dir)
del _project_dir

from bin.nn_baselines import Model

from lib.data import build_dataset
from lib.data import standardize_labels

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lib import KWArgs, PartKey

torch.set_num_threads(4)

def main(dataset_name):
    with open(f'data/{dataset_name}/info.json', 'rt') as fin:
        info_d = (json.load(fin))
    if info_d['task_type'] == 'binclass':
        target_policy = 'binclass'
        y_policy = None
    else:
        target_policy = 'regression'
        y_policy = "standard"

    if not(os.path.exists(f'data/{dataset_name}/X_cat.npy')):
        cat_policy = None
    else:
        cat_policy = 'ordinal'

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

    with open(f'exp/mlp/{dataset_name}/evaluation/0/report.json') as f:
        j = json.load(f)
    model = Model(**j['config']['model'], n_num_features=dataset.n_num_features,
      n_bin_features=dataset.n_bin_features, cat_cardinalities=dataset.compute_cat_cardinalities(),
      n_classes=dataset.task.try_compute_n_classes(), bins=None)

    meta_x = dataset['x_meta']['train'][:, 0][np.argsort(dataset['x_meta']['train'][:, 0])]
    thr_meta = meta_x[dataset['x_meta']['train'].shape[0] // 5 * 4]
    ids_train = np.arange(len(dataset['x_meta']['train']))[dataset['x_meta']['train'][:, 0] >= thr_meta]

    def apply_model(part: PartKey, idx: Tensor) -> Tensor:
        return model(
            **{
                key: dataset.data[key][part][idx]  # type: ignore[index]
                for key in ['x_num', 'x_bin', 'x_cat']
                if key in dataset  # type: ignore[index]
            }
        ).squeeze(-1)

    lr = 3e-5
    batch_size = 1024
    if dataset_name == 'sberbank-housing':
        batch_size = 256
    import time
    tt = time.time()
    train_batches = DataLoader(ids_train, batch_size=batch_size, shuffle=True)
    val_batches = DataLoader(np.arange(len(dataset['x_meta']['val'])), batch_size=2**16, shuffle=False)
    test_batches = DataLoader(np.arange(len(dataset['x_meta']['test'])), batch_size=2**16, shuffle=False)
    os.mkdir(f'exp/dfr/{dataset_name}')
    os.mkdir(f'exp/dfr/{dataset_name}/evaluation')
    device = 'cuda'
    dataset = dataset.to_torch(device)

    for seed in range(15):
        preds = dict()
        model.load_state_dict(torch.load(f'exp/mlp/{dataset_name}/evaluation/{seed}/checkpoint.pt')['model'])
        for i in range(len(model.backbone.blocks)):
            model.backbone.blocks[i].linear.requires_grad_(False)
        model.to(device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        val_scores = []
        for epoch in range(1000):
            model.train()
            for idx in train_batches:
                cur_pred = apply_model('train', idx)
                if target_policy == 'regression':
                    loss = ((cur_pred - dataset['y']['train'][idx].to(device))**2).mean()
                else:
                    loss = F.binary_cross_entropy_with_logits(cur_pred, dataset['y']['train'][idx].to(device).float())
                loss.backward()
                opt.step()
                opt.zero_grad()
            model.eval()
            val_loss_cur = 0
            val_loss_count = 0
            with torch.no_grad():
                if target_policy == 'regression':
                    for idx in val_batches:
                        cur_pred = apply_model('val', idx)
                        loss = ((cur_pred - dataset['y']['val'][idx].to(device))**2).sum()
                        val_loss_cur += loss
                        val_loss_count += len(idx)
                    val_scores.append((val_loss_cur / val_loss_count).item()**0.5 * regression_label_stats.std)
                else:
                    val_preds = []
                    for idx in val_batches:
                        cur_pred = apply_model('val', idx)
                        val_preds.append(cur_pred)
                    val_y_pred = torch.cat(val_preds, dim=0)
                    val_score = roc_auc_score(dataset['y']['val'].cpu().numpy().astype('int32'), val_y_pred.cpu().numpy())
                    val_scores.append(-val_score)

                if np.argmin(val_scores) < epoch - 16:
                    break
                if np.argmin(val_scores) == epoch:
                    val_preds = []
                    for idx in val_batches:
                        y_pred = apply_model('val', idx)
                        val_preds.append(y_pred)
                    val_y_pred = torch.cat(val_preds, dim=0)
                    test_preds = []
                    for idx in test_batches:
                        y_pred = apply_model('test', idx)
                        test_preds.append(y_pred)
                    y_pred = torch.cat(test_preds, dim=0)
                    if target_policy == 'regression':
                        val_score = (val_y_pred - dataset['y']['val']).square().mean().item()**0.5 * regression_label_stats.std
                        test_score = (y_pred - dataset['y']['test']).square().mean().item()**0.5 * regression_label_stats.std
                    else:
                        val_score = -roc_auc_score(dataset['y']['val'].cpu().numpy().astype('int32'), val_y_pred.cpu().numpy())
                        test_score = -roc_auc_score(dataset['y']['test'].cpu().numpy().astype('int32'), y_pred.cpu().numpy())
        j = dict()
        j['function'] = None
        j['config'] = dict()
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
        print(seed, val_score, test_score, np.argmin(val_scores))
        os.mkdir(f'exp/dfr/{dataset_name}/evaluation/{seed}')
        if target_policy == 'regression':
            np.savez(f'exp/dfr/{dataset_name}/evaluation/{seed}/predictions.npz', 
                test=y_pred.cpu() * regression_label_stats.std + regression_label_stats.mean,
                val=val_y_pred.cpu() * regression_label_stats.std + regression_label_stats.mean)
        else:
            np.savez(f'exp/dfr/{dataset_name}/evaluation/{seed}/predictions.npz', test=y_pred.cpu(), val=val_y_pred.cpu())

        with open(f'exp/dfr/{dataset_name}/evaluation/{seed}/report.json', 'wt') as f:
            json.dump(j, f, indent=4)
        fout = open(f'exp/dfr/{dataset_name}/evaluation/{seed}/DONE', 'wt')
        fout.close()

main(sys.argv[1])
