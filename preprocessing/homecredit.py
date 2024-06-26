# Homecredit Default Stability Preprocessing
import itertools
from typing import Iterable

import numpy as np
import polars as pl
import kaggle
from sklearn.preprocessing import OrdinalEncoder
from loguru import logger

import lib
from preprocessing.util import save_dataset

TMP_DATA_PATH = lib.PROJECT_DIR/'preprocessing/tmp/homecredit-default'
DATA_FILES_PATH = TMP_DATA_PATH/'parquet_files/train'
TMP_DATA_PATH.mkdir(exist_ok=True, parents=True)


def set_table_dtypes(df):
    "set dtypes to optimize data size"

    for col in df.columns:
        if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
            df = df.with_columns(pl.col(col).cast(pl.Int32))
        elif col in ["date_decision"] or col[-1] == 'D':
            df = df.with_columns(pl.col(col).cast(pl.Date))

    # Type downcasting
    int_types = [pl.Int8,pl.Int16,pl.Int32,pl.Int64]
    float_types = [pl.Float32,pl.Float64]
    table_min = df.select(pl.col(df.columns).min()).collect(streaming=True)
    table_max = df.select(pl.col(df.columns).max()).collect(streaming=True)

    for col, col_type in df.schema.items():
        c_min = table_min[col].item()
        c_max = table_max[col].item()

        if col_type in int_types:
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df = df.with_columns(pl.col(col).cast(pl.Int8))
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df = df.with_columns(pl.col(col).cast(pl.Int16))
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df = df.with_columns(pl.col(col).cast(pl.Int32))
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
        elif col_type in float_types:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df = df.with_columns(pl.col(col).cast(pl.Float32))
    return df


def read_tables(table_paths: Iterable):
    table_paths = list(table_paths)
    res = pl.concat([
        pl.scan_parquet(p, low_memory=True, rechunk=True) for p in table_paths
    ], how="vertical_relaxed")

    logger.info(f'Read {[p.name for p in table_paths]}')
    return res


def aggregate_features(df: pl.LazyFrame):
    "Aggregation expressions"

    # Basic logic -- all numeric values we aggregate with max, min, std, mean

    cols = df.columns
    # Numeric values
    exprs = sum([
        [
            pl.col(col).max().alias(f"max_{col}"),
            pl.col(col).min().alias(f"min_{col}"),
            pl.col(col).mean().alias(f"mean_{col}"),
            pl.col(col).std().alias(f"std_{col}"),
        ] for col in cols
        if col[-1] in ("P", "A", "T", "L")
    ], [])

    # categorical expressions (strings)
    exprs += sum([
        [
            pl.col(col).last().alias(f"last_{col}"),
            pl.col(col).n_unique().alias(f"n_unique_{col}"),
            pl.col(col).first().alias(f"first_{col}"),
        ] for col in cols
        if col[-1] == 'M'
    ], [])
    # Dates
    exprs += sum([
        [
            pl.col(col).max().alias(f"max_{col}"),
            pl.col(col).min().alias(f"min_{col}"),
            pl.col(col).mean().alias(f"mean_{col}"),
            pl.col(col).std().alias(f"std_{col}"),
        ] for col in cols
        if col[-1] == "D"
    ], [])

    # Count aggregates
    exprs += [pl.col(col).max().alias(f'max_{col}') for col in cols if 'num_group' in cols]

    return [df.sort('num_group1').group_by("case_id").agg(exprs)]



def main():
    logger.info('Preprocessing homecredit-default-stability data')
    kaggle.api.competition_download_files('home-credit-credit-risk-model-stability', path=TMP_DATA_PATH)
    lib.unzip(TMP_DATA_PATH/'home-credit-credit-risk-model-stability.zip')

    # ================================================================
    # >> Load data aggregate and join
    # ================================================================

    train_basetable = read_tables(DATA_FILES_PATH.glob("train_base.parquet")).pipe(set_table_dtypes)
    train_static = read_tables(DATA_FILES_PATH.glob("train_static_0_*.parquet")).pipe(set_table_dtypes)
    train_static_cb = read_tables(DATA_FILES_PATH.glob("train_static_cb_0.parquet")).pipe(set_table_dtypes)


    train_aggregated = list(itertools.chain.from_iterable([
        aggregate_features(read_tables(DATA_FILES_PATH.glob(name)).pipe(set_table_dtypes))

        for name in [
            "train_applprev_1_*.parquet",
            "train_tax_registry_a_1.parquet",
            "train_tax_registry_b_1.parquet",
            "train_tax_registry_c_1.parquet",
            "train_credit_bureau_a_1_*.parquet",
            "train_credit_bureau_b_1.parquet",
            "train_other_1.parquet",
            "train_person_1.parquet",
            "train_deposit_1.parquet",
            "train_debitcard_1.parquet",
            "train_credit_bureau_a_2_*.parquet",
            "train_credit_bureau_b_2.parquet",
        ]
    ]))

    logger.info('Constructing train data table')

    data = train_basetable.clone()

    for i, df in enumerate([train_static, train_static_cb] + train_aggregated):
        data = data.join(df, how="left", on="case_id", suffix=f"_{i}")

    data = data.collect(streaming=True)
    data = data.with_columns([
        (pl.col(col) - pl.col('date_decision')).dt.total_days().cast(pl.Float32)
        for col in data.columns if col.endswith('D')
    ])

    # ================================================================
    # >> Drop redundant columns
    # ================================================================

    many_nulls = data.select(pl.col('*').is_null().mean().gt(0.95))
    n_unique = data.select(pl.col('*').drop_nulls().n_unique())

    drop_cols = [
        c for c, dtype in data.schema.items()
        if (many_nulls[c].item() or n_unique[c].item() == 1 or (dtype == pl.String and n_unique[c].item() > 50))
    ]

    data = data.drop(drop_cols)
    data = data.drop(
        'min_sex_738L', 'MONTH', 'WEEK_NUM', 
    )

    # ================================================================
    # >> Column types
    # ================================================================

    bin_cols = [
        pl.col('isbidproduct_1095L'),
        pl.col('max_sex_738L').eq('F'),
    ]

    cat_cols = [
        c for c, dtype in data.schema.items()
        if (
            c not in [n.meta.output_name() for n in bin_cols] + ['target', 'case_id', 'date_decision'] and
            dtype == pl.String
        )
    ]

    num_cols = [
        c for c, dtype in data.schema.items()
        if (
            c not in [n.meta.output_name() for n in bin_cols] + ['target', 'case_id', 'date_decision'] and
            dtype in [pl.Float32, pl.Float64, pl.UInt32, pl.Boolean]
        )
    ]

    # Cat features encoding and data subsampling
    data = data.with_row_index(name='index_in_full')
    data_cat = data.select(cat_cols)
    data = pl.concat([
        data.drop(cat_cols),
        pl.DataFrame(
            OrdinalEncoder(min_frequency=1/100).fit_transform(data_cat.to_numpy()),
            schema={c: pl.Int64 for c in data_cat.columns}
        ),
    ], how="horizontal")

    data_sample = data.sample(fraction=0.25, seed=0)
    data_sample = data_sample.with_columns(
        pl.col('date_decision').dt.weekday().alias('day_of_week'),
        pl.col('date_decision').dt.day().alias('day_of_month'),
        pl.col('date_decision').dt.ordinal_day().alias('day_of_year'),
    ).sort(by='date_decision')

    Y_data = data_sample.select('target').cast(pl.Int64)
    X_num_data = data_sample.select(num_cols + ['day_of_week', 'day_of_month', 'day_of_year']).cast(pl.Float32)
    X_bin_data = data_sample.select(bin_cols).cast(pl.Float32)
    X_cat_data = data_sample.select(cat_cols).cast(pl.Int32)
    X_meta_data = data_sample.select('date_decision', 'index_in_full').cast(pl.Int32)


    # ======================================================================================
    # >>> Default task split <<<
    # ======================================================================================

    # Validation and test are the last two weeks, train is the prior month

    default_train_idx = data_sample.with_row_index().filter(
        pl.col('date_decision').lt(pl.datetime(2020, 1, 1, 0, 0, 0))
    )['index'].to_numpy()

    default_val_idx = data_sample.with_row_index().filter(
      pl.col('date_decision').ge(pl.datetime(2020, 1, 1, 0, 0, 0)) &
      pl.col('date_decision').lt(pl.datetime(2020, 5, 1, 0, 0, 0))
    )['index'].to_numpy()

    default_test_idx = data_sample.with_row_index().filter(
      pl.col('date_decision').ge(pl.datetime(2020, 5, 1, 0, 0, 0))
    )['index'].to_numpy()

    default_split = {
        'train_idx': default_train_idx, 'val_idx': default_val_idx, 'test_idx': default_test_idx,
    }

    # ======================================================================================
    # >>> Sliding window splits <<<
    # ======================================================================================

    # Create 3 windowed time splits
    # We shift each window [val_size = 20_000] to the right
    test_val_size = 50_000
    num_splits = 3

    # Calculate the remaining train samples
    train_size = data_sample.shape[0] - (num_splits - 1) * test_val_size - 2 * test_val_size

    sliding_window_splits = [
        {
            'train_idx': np.arange(train_size),
            'val_idx': np.arange(train_size, train_size + test_val_size),
            'test_idx': np.arange(train_size + test_val_size, train_size + 2 * test_val_size),
        }
    ]

    for _ in range(1, num_splits):
        sliding_window_splits.append(
            {
                k: v + test_val_size for k,v in sliding_window_splits[-1].items()
            }
        )

    def time_range(df, split):
        print('Start:', df[split[0].item()]['date_decision'].item())
        print('End:', df[split[-1].item()]['date_decision'].item())

        return df[split[-1].item()]['date_decision'].item() - df[split[0].item()]['date_decision'].item()

    print('\n>>> Default Split')
    print('\nTrain time info:')
    print(time_range(data_sample, default_split['train_idx']))
    print('\nVal time info:')
    print(time_range(data_sample, default_split['val_idx']))
    print('\nTest time info:')
    print(time_range(data_sample, default_split['test_idx']))

    for s in sliding_window_splits:
        print('\n>>> Sliding Window Split')
        print('\nTrain time info:')
        print(time_range(data_sample, s['train_idx']))
        print('\nVal time info:')
        print(time_range(data_sample, s['val_idx']))
        print('\nTest time info:')
        print(time_range(data_sample, s['test_idx']))


    # ======================================================================================
    # >>> Random splits <<<
    # ======================================================================================

    # Random splits are created from the sliding_window splits by shuffling  all the data in the
    # respective window and respliting into same train validation and test dataset sizes

    np.random.seed(0)
    random_splits = []
    for split in sliding_window_splits:
        idxs = np.concatenate([v for v in split.values()])
        np.random.shuffle(idxs)
        random_splits.append({
            'train_idx': idxs[:train_size],
            'val_idx': idxs[train_size:train_size+test_val_size],
            'test_idx': idxs[train_size+test_val_size:],        
        })

    # ======================================================================================
    # >>> Save data and splits <<<
    # ======================================================================================

    data_parts = {n.rsplit('_', maxsplit=1)[0]: v for n, v in locals().items() if n.endswith('_data')}
    all_splits = (
        {'default': default_split} |
        {f'sliding-window-{i}': split for i, split in enumerate(sliding_window_splits)} |
        {f'random-{i}': split for i, split in enumerate(random_splits)}
    )

    logger.info('Writing data to disk')
    save_dataset(
        name='homecredit-default',
        task_type='binclass',
        data=data_parts,
        splits=all_splits,
    )


if __name__ == "__main__":
    main()
