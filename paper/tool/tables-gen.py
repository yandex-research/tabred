
import lib

import itertools
import pprint

from pathlib import Path
from typing import cast

import pandas as pd
import numpy as np
import polars as pl
import plotnine as pn
from scipy.stats import tukey_hsd

# from sklearn.metrics import roc_auc_score, mean_absolute_error
# from scipy.special import expit

models = [
    'xgboost_',
    'lightgbm_',
    'catboost_',
    'random_forest',
    'linear_model',
    'mlp',
    'snn',
    'dcn2',
    'resnet',
    'ft_transformer',
    'mlp-plr',
    'tabr',
    'tabr-causal',
    'trompt',
    'coral',
    'dfr',
    #  TODO coral, linfin, linear, random forest, trompt
]

datasets = [
    'homesite-insurance',
    'ecom-offers',
    'homecredit-default',
    'sberbank-housing',
    'cooking-time',
    'delivery-eta',
    'maps-routing',
    'weather',
]

model_name = {
    'xgboost_': 'XGBoost',
    'catboost_': 'CatBoost',
    'lightgbm_': 'LightGBM',
    'snn': 'SNN',
    'mlp': 'MLP',
    'mlp-plr': 'MLP (PLR)',
    'coral': 'CORAL',
    'dfr': 'DFR',
    'resnet': 'ResNet',
    'dcn2': 'DCNv2',
    'ft_transformer': 'FT-Transformer',
    'tabr-causal': 'TabR (causal)',
    'tabr': 'TabR',
    'trompt': 'Trompt',
    'random_forest': 'RandomForest',
    'linear_model': 'Linear',
}

dataset_name = {
    'homesite-insurance': 'Homesite Insurance',
    'ecom-offers': 'E-Commerce Offers',
    'homecredit-default': 'Homecredit Default',
    'sberbank-housing': 'Sberbank Housing',
    'cooking-time': 'Cooking Time',
    'delivery-eta': 'Delivery ETA',
    'maps-routing': 'Maps Routing',
    'weather': 'Weather',
}




data = pl.DataFrame(cast(pd.DataFrame, pd.json_normalize(
    [
        lib.load_json(f) | {'model': m}
        for m in models
        for d in datasets
        for f in lib.EXP_DIR.glob(f'{m}/{d}/evaluation/**/report.json')
    ]
).fillna(0.0))[['model', 'config.data.path', 'metrics.test.score', 'config.seed']])

scores = data.select(
    pl.col('model'),
    pl.col('config.data.path').alias('data'),
    pl.col('metrics.test.score').alias('score'),
    pl.col('config.seed').alias('seed'),
)


# TODO, compute ranks with various methods
def get_ranks_ours(data, dataset):
    d_ranking = (
        data
        .filter(
            pl.col('model').is_in(models)
            & pl.col('data').eq(f':data/{dataset}')
        )
        .group_by('model')
        .agg(
            pl.col('score').mean().alias('mean'),
            pl.col('score').std().alias('std')
        ).sort('mean', descending=True)
    )

    model_rank = {}
    rank = 1
    current_mean, current_std = None, None

    for model, mean, std in d_ranking.iter_rows():
        if std is None:
            std = float('inf')
        if current_mean is None:
            model_rank[model] = rank
            current_mean = mean
            current_std = std
        elif current_mean - mean <= current_std:
            model_rank[model] = rank
        else:
            rank += 1
            model_rank[model] = rank
            current_mean = mean
            current_std = std

    ranks = []
    for m in models:
        ranks.append(model_rank.get(m, float('nan')))

    return ranks


def get_ranks_tukey(data, dataset, pvalue_threshold=0.05):
    dataset_results = (
        data
         .filter(
             pl.col('model').is_in(models)
             & pl.col('data').eq(f':data/{dataset}')
         )
         .group_by('model')
         .agg(pl.col('score'))
    )
    means = dataset_results.with_columns(pl.col('score').list.mean())
    pvalues = tukey_hsd(*list(dataset_results['score'].to_list())).pvalue
    key_to_idx = {}
    for i, key in enumerate(dataset_results['model'].to_list()):
        key_to_idx[key] = i
    sorted_res = means.sort(by='score', descending=True)['model'].to_list()
    prev = None
    first = None
    rank = 1
    ranks = []
    model_rank = {}
    for key in sorted_res:
        if first is not None:
            first_idx = key_to_idx[first]
            cur_idx = key_to_idx[key]
            if pvalues[cur_idx][first_idx] < pvalue_threshold:
                rank += 1
                first = key
        else:
            first = key
        model_rank[key] = rank
    for m in models:
        ranks.append(model_rank.get(m, float('nan')))
    return ranks


# ranks_ours = np.array([get_ranks_tukey(scores, d) for d in datasets])
ranks_ours = np.array([get_ranks_ours(scores, d) for d in datasets])
ranks_mean = np.nanmean(ranks_ours, axis=0)
ranks_std = np.nanstd(ranks_ours, axis=0)


# TabReD Score
# This is computed as a percentage difference to the tuned MLP scores

mean_scores = scores.group_by('model', 'data').agg(
    pl.col('score').mean()
)
mlp_scores = mean_scores.filter(pl.col('model').eq('mlp')).drop('model')
mean_scores = mean_scores.join(mlp_scores, on='data', suffix='_mlp')


tabred_score = mean_scores.with_columns(
    pl.when(pl.col('data').is_in([':data/homecredit-default', ':data/homesite-insurance', ':data/ecom-offers'])).then(
        (pl.col('score') - pl.col('score_mlp')) / pl.col('score_mlp') * 100
    ).otherwise(
        -1 * (pl.col('score') - pl.col('score_mlp')) / pl.col('score_mlp') * 100
    ).alias('uplift')
).group_by(
    'model'
).agg(
    pl.col('uplift').median().alias('median'),
    pl.col('uplift').min().alias('min'),
    pl.col('uplift').max().alias('max'),
).sort('median', descending=True).to_dicts()

trs = {}
for d_ in tabred_score:
    m = d_.pop('model')
    trs[m] = d_


mean_scores.with_columns(
    pl.when(pl.col('data').is_in([':data/homecredit-default', ':data/homesite-insurance', ':data/ecom-offers'])).then(
        (pl.col('score') - pl.col('score_mlp')) / pl.col('score_mlp') * 100
    ).otherwise(
        -1 * (pl.col('score') - pl.col('score_mlp')) / pl.col('score_mlp') * 100
    ).alias('uplift')
).filter(pl.col('model').eq('resnet')).sort('uplift')['uplift'][4]

# Main table

TREE_BASED_SEP = r"""
\midrule
\multicolumn{10}{l}{\hspace{0.1em} \textbf{Classical ML Baselines}} \vspace{4px} \\ 
"""

DL_SEP = r"""
\midrule
\multicolumn{10}{l}{\hspace{0.1em} \textbf{Deep Learning Methods}} \vspace{4px} \\ 
"""

OOD_SEP = r"""
\midrule
\multicolumn{10}{l}{\hspace{0.1em} \textbf{OOD Robustness Methods}} \vspace{4px} \\ 
"""


for i,m in enumerate(models):
    name = model_name[m]
    if m == 'xgboost_':
        print(TREE_BASED_SEP)
    elif m == 'mlp':
        print(DL_SEP)
    elif m == 'coral':
        print(OOD_SEP)

    print(name)

    for j,d in enumerate(datasets):
        info = lib.load_json(lib.DATA_DIR/d/'info.json')
        rank = ranks_ours[j][i]
        # color = rank_colors[rank - 1]

        e = scores.filter(
            pl.col('model').eq(m) & 
            pl.col('data').eq(f':data/{d}')
        )['score'] * (-1 if info['task_type'] == 'regression' else 1)

        e_mean = cast(float, e.mean())
        e_std = cast(float, e.std())

        if d in ['maps-routing', 'homesite-insurance']:
            precision = 4
        else:
            precision = 4

        if len(e) > 0:
            if rank == 1:
                pref = r'\textbf{'
                suff = r'}'
            elif rank == 2:
                pref = r'\underline{'
                suff = r'}'
            else:
                pref = suff = ''

            if e_std is None:
                e_std = 0
            print(
                ' & ' + pref + r'{\footnotesize ' + f'{e_mean:.{precision}f}' + '}' + suff
                # + r'{\tiny$\pm$' f'{e_std:.{precision+1}f}' '}'
            )
        else:
            print(' & --')

    # Uncomment if reporting tabred score
    # tr_min = round(trs[m]['min'], 1) + 0.0
    # tr_max = round(trs[m]['max'], 1) + 0.0
    # tr_q50 = round(trs[m]['median'], 2) + 0.0
    # print(r' & {\footnotesize' + fr' {tr_q50}'  + r'}\\')
    # Uncomment if reporting average rank
    print(r' & {\footnotesize' + fr' {ranks_mean[i]:.1f} $\pm$ {ranks_std[i]:.1f}' + r'}\\')
