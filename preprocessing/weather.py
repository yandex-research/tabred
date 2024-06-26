# Weather Forecasting

import kaggle
import polars as pl
import numpy as np
from loguru import logger

import lib
from preprocessing.util import save_dataset


TMP_DATA_PATH = lib.PROJECT_DIR/'preprocessing/tmp/weather'
TMP_DATA_PATH.mkdir(exist_ok=True, parents=True)


def main():
    # ================================================================
    # >> Load and split into num, bin, cat
    # ================================================================

    logger.info('Preprocessing Weather data')
    kaggle.api.dataset_download_files('pcovkrd84mejm/tabred-weather', path=TMP_DATA_PATH)
    lib.unzip(TMP_DATA_PATH/'tabred-weather.zip')


    data = pl.read_parquet(TMP_DATA_PATH/'weather.parquet').with_row_index(name="index_in_full")
    data = data.sample(fraction=0.025, seed=0)
    data = data.sort('fact_time')

    # Generate time features
    data = data.with_columns(
        (pl.col('fact_time') * 10**6).cast(pl.Datetime)
    )
    data = data.with_columns(
        pl.col('fact_time').dt.weekday().alias('day_of_week'),
        pl.col('fact_time').dt.day().alias('day_of_month'),
        pl.col('fact_time').dt.time().cast(pl.Duration).dt.total_minutes().alias('minute_of_day'),
        pl.col('fact_time').dt.time().cast(pl.Duration).dt.total_hours().alias('hour_of_day'),
        pl.col('fact_time').dt.month().alias('month'),
    )


    target_cols = ['fact_temperature']
    meta_cols = ['fact_time', 'apply_time_rl', 'fact_latitude', 'fact_longitude', 'fact_station_id', 'index_in_full']

    bin_cols = [c for c in data.columns if 'available' in c and c not in target_cols + meta_cols]
    num_cols = [c for c in data.columns if not 'available' in c and c not in target_cols + meta_cols]

    Y_data = data.select(target_cols).cast(pl.Float32)
    X_meta_data = data.select(meta_cols)
    X_bin_data = data.select(bin_cols).cast(pl.Float32)
    X_num_data = data.select(num_cols).cast(pl.Float32)


    # ======================================================================================
    # >>> Default task split <<<
    # ======================================================================================

    # Validation and test are one month long, train is whats left of the year before (10 month including
    # the test month one year prior)

    default_train_idx = data.with_row_index().filter(
        pl.col('fact_time').lt(pl.datetime(2023, 6, 1, 0, 0, 0))
    )['index'].to_numpy()

    default_val_idx = data.with_row_index().filter(
      pl.col('fact_time').ge(pl.datetime(2023, 6, 1, 0, 0, 0)) &
      pl.col('fact_time').lt(pl.datetime(2023, 7, 1, 0, 0, 0))
    )['index'].to_numpy()

    default_test_idx = data.with_row_index().filter(
      pl.col('fact_time').ge(pl.datetime(2023, 7, 1, 0, 0, 0))
    )['index'].to_numpy()

    default_split = {
        'train_idx': default_train_idx, 'val_idx': default_val_idx, 'test_idx': default_test_idx,
    }


    # ======================================================================================
    # >>> Sliding window splits <<<
    # ======================================================================================

    # test/val size is roughly 1 month long, and the train is roughly 8-9 prior month 

    test_val_size = 40_000
    num_splits = 3

    # Calculate the remaining train samples
    train_size = data.shape[0] - (num_splits - 1) * test_val_size - 2 * test_val_size

    sliding_window_splits = [
        {
            'train_idx': np.arange(train_size),
            'val_idx': np.arange(train_size, train_size + test_val_size),
            'test_idx': np.arange(train_size + test_val_size, train_size + 2 * test_val_size),
        }
    ]

    for i in range(1, num_splits):
        sliding_window_splits.append(
            {
                k: v + test_val_size for k,v in sliding_window_splits[-1].items()
            }
        )

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
    # >>> Save <<<
    # ======================================================================================


    data_parts = {n.rsplit('_', maxsplit=1)[0]: v for n, v in locals().items() if n.endswith('_data')}
    all_splits = (
        {'default': default_split} |
        {f'sliding-window-{i}': split for i, split in enumerate(sliding_window_splits)} |
        {f'random-{i}': split for i, split in enumerate(random_splits)}
    )

    logger.info('Writing data to disk')
    save_dataset(
        name='weather', 
        task_type='regression',
        data=data_parts,
        splits=all_splits,
    )


if __name__ == "__main__":
    main()
