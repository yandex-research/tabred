# > Maps Routing task and dataset

import kaggle
import polars as pl
import numpy as np
from loguru import logger
from sklearn.preprocessing import OrdinalEncoder

import lib
from preprocessing.util import save_dataset



TMP_DATA_PATH = lib.PROJECT_DIR/'preprocessing/tmp/maps-routing'
TMP_DATA_PATH.mkdir(exist_ok=True, parents=True)


def main():
    # ================================================================
    # >> Load and split into num, bin, cat
    # ================================================================

    logger.info('Preprocessing maps routing dataset')
    kaggle.api.dataset_download_files('pcovkrd84mejm/maps-routing', path=TMP_DATA_PATH)
    lib.unzip(TMP_DATA_PATH/'maps-routing.zip')

    # Store full_index to help with potential future experiments on large data
    data = pl.read_parquet(TMP_DATA_PATH/'maps_routing.parquet').with_row_index(name="index_in_full")

    cat_cols = [c for c in data.columns if c.startswith('cat')]
    num_cols = [c for c in data.columns if c.startswith('num') or c in ['day_of_week', 'minute_of_day', 'hour_of_day']]

    
    data_cat = data.select(cat_cols)
    data = data.drop(cat_cols)
    data = pl.concat([
        data,
        pl.DataFrame(
            OrdinalEncoder(min_frequency=1/100).fit_transform(data_cat.to_numpy()),
            schema={c: pl.Int64 for c in data_cat.columns}
        )
    ], how="horizontal")

    # Subsample to allow for faster experiment iterations
    data = data.sample(fraction=0.025, seed=0)
    data = data.with_columns(pl.col('timestamp').cast(pl.Datetime))
    data = data.sort('timestamp')
    data = data.with_columns(
        pl.col('timestamp').dt.weekday().alias('day_of_week'),
        pl.col('timestamp').dt.time().cast(pl.Duration).dt.total_minutes().alias('minute_of_day'),
        pl.col('timestamp').dt.time().cast(pl.Duration).dt.total_hours().alias('hour_of_day'),
    )

    Y_data = data.select(pl.col('target_log_spkm'))
    X_meta_data = data.select('timestamp', 'index_in_full', 'track_length')
    X_cat_data = data.select(cat_cols)
    X_num_data = data.select(num_cols)

    # ======================================================================================
    # >>> Default task split <<<
    # ======================================================================================

    # Validation and test are the last two weeks, train is the prior month

    default_train_idx = data.with_row_index().filter(
        pl.col('timestamp').lt(pl.datetime(2023, 11, 20, 0, 0, 0))
    )['index'].to_numpy()

    default_val_idx = data.with_row_index().filter(
      pl.col('timestamp').ge(pl.datetime(2023, 11, 20, 0, 0, 0)) &
      pl.col('timestamp').lt(pl.datetime(2023, 11, 27, 0, 0, 0))
    )['index'].to_numpy()

    default_test_idx = data.with_row_index().filter(
      pl.col('timestamp').ge(pl.datetime(2023, 11, 27, 0, 0, 0)) &
      pl.col('timestamp').lt(pl.datetime(2023, 12, 4, 0, 0, 0))
    )['index'].to_numpy()

    default_split = {
        'train_idx': default_train_idx, 'val_idx': default_val_idx, 'test_idx': default_test_idx,
    }


    # ======================================================================================
    # >>> Sliding window splits <<<
    # ======================================================================================

    # test/val size is roughly a week, and the train is whats left of the month (~1.5 weeks)

    test_val_size = 60_000
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

    for _ in range(1, num_splits):
        sliding_window_splits.append(
            {
                k: v + test_val_size for k,v in sliding_window_splits[-1].items()
            }
        )

    def time_range(df, split):
        print('Start:', df[split[0].item()]['timestamp'].item())
        print('End:', df[split[-1].item()]['timestamp'].item())

        return df[split[-1].item()]['timestamp'].item() - df[split[0].item()]['timestamp'].item()

    for s in sliding_window_splits:
        print('\nTrain time info:')
        print(time_range(data, s['train_idx']))
        print('\nVal time info:')
        print(time_range(data, s['val_idx']))
        print('\nTest time info:')
        print(time_range(data, s['test_idx']))

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


    # Save dataset in the following format
    # x_[bin|num|cat], y, split/default/ids_[train|val|test], split/sliding-window-N/ids-[train|val|test], split/random-N/ids-[train|val|test]
    # sliding window splits are formed by a custom increment, with the same train/test/val sizes
    # random splits match sliding window time-based splits in train/test/val sizes, but othervise are just random

    data_parts = {n.rsplit('_', maxsplit=1)[0]: v for n, v in locals().items() if n.endswith('_data')}
    all_splits = (
        {'default': default_split} |
        {f'sliding-window-{i}': split for i, split in enumerate(sliding_window_splits)} |
        {f'random-{i}': split for i, split in enumerate(random_splits)}
    )

    logger.info('Writing data to disk')
    save_dataset(
        name='maps-routing',
        task_type='regression',
        data=data_parts,
        splits=all_splits,
    )


if __name__ == "__main__":
    main()
