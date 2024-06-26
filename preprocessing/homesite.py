# Homesite Insurance preprocessing and splitting script

import zipfile

import polars as pl
import numpy as np
import kaggle
from loguru import logger
from sklearn.preprocessing import OrdinalEncoder

import lib
from preprocessing.util import save_dataset


TMP_DATA_PATH = lib.PROJECT_DIR/'preprocessing/tmp/homesite'
TMP_DATA_PATH.mkdir(exist_ok=True, parents=True)


def main():
    logger.info('Preprocessing homesite-insurance dataset')

    # Download and load dataset to memory

    kaggle.api.competition_download_files('homesite-quote-conversion', path=TMP_DATA_PATH)
    lib.unzip(TMP_DATA_PATH/'homesite-quote-conversion.zip')

    # ======================================================================================
    # >>> Split into num, cat, bin and add time features <<<
    # ======================================================================================

    data = pl.read_csv(zipfile.ZipFile(TMP_DATA_PATH/'train.csv.zip').read('train.csv'))
    data = data.with_columns(pl.col('Original_Quote_Date').str.to_datetime().alias('timestamp'))
    data = data.drop('QuoteNumber', 'Original_Quote_Date')

    # Replace -1 in ordinal values with Null as it indicates missing values
    # Columns ending with A,B are ordinal values and -1 means missing data

    data = data.with_columns(pl.col([c for c in data.columns if c.endswith(('A', 'B'))]).replace({-1: None}))

    # We always add some basic features based on timestamp that are generalizable for the dataset at
    # hand
    data = data.with_columns(
        pl.col('timestamp').dt.weekday().alias('day_of_week'),
        pl.col('timestamp').dt.day().alias('day_of_month'),
        pl.col('timestamp').dt.ordinal_day().alias('day_of_year'),
        pl.col('timestamp').dt.month().alias('month'),
    ).sort('timestamp')

    # Filter columns with all null values
    all_nulls = [k for k, v in data.select(pl.col('*').is_null().mean().eq(1.0)).to_dicts()[0].items() if v]
    data = data.drop(all_nulls)

    target_columns = ['QuoteConversion_Flag']
    timestamp_meta_columns = ['timestamp']

    value_counts = data.select(pl.col('*').n_unique()).to_dicts()[0]

    bin_columns = [
        c for c, count in value_counts.items()
        if count == 2 if c not in target_columns + timestamp_meta_columns
    ]
    X_bin_data = data[bin_columns].with_columns(
        # ordinals with only 1 non null value
        pl.col(col.name for col in data[bin_columns].null_count() if col.item() > 0).is_null().not_(),
        # N/Y binary columns
        pl.col(col for col in bin_columns if data.schema[col] == pl.String).replace({"N": 0, "Y": 1}),
        # The rest are 0/1 numerics
    ).cast(pl.Float32)

    cat_columns = [
        c for c, dtype in data.schema.items()
        if dtype == pl.String and c not in bin_columns + target_columns + timestamp_meta_columns
    ]
    X_cat_data = data.select(pl.col(cat_columns).cast(pl.Categorical).rank('dense'))

    # Remove non-frequent categorical variables
    X_cat_data_np = X_cat_data.to_numpy()
    X_cat_data_np = OrdinalEncoder(min_frequency=1/100).fit_transform(X_cat_data_np)
    X_cat_data = pl.DataFrame(X_cat_data_np).cast(pl.Int64)

    num_columns = list(set(data.columns) - set(bin_columns + cat_columns + target_columns + timestamp_meta_columns))
    X_num_data = data[num_columns].cast(pl.Float32)

    X_meta_data = data[timestamp_meta_columns].cast(pl.Int64)
    Y_data = data[target_columns].cast(pl.Int64)

    # ======================================================================================
    # >>> Default task split <<<
    # ======================================================================================

    # Last 2 month
    default_test_idx = data.with_row_index().filter(
        pl.col('timestamp').dt.year().eq(2015) &
        pl.col('timestamp').dt.month().is_in([4,5])
    )['index'].to_numpy()

    # 2 month prior to last 2 month
    default_val_idx = data.with_row_index().filter(
        pl.col('timestamp').dt.year().eq(2015) &
        pl.col('timestamp').dt.month().is_in([2,3])
    )['index'].to_numpy()

    # The rest
    default_train_idx = np.arange(default_val_idx[0])

    default_split = {
        'train_idx': default_train_idx, 'val_idx': default_val_idx, 'test_idx': default_test_idx,
    }

    # ======================================================================================
    # >>> Sliding window splits <<<
    # ======================================================================================

    # Create 3 windowed time splits
    # We shift each window [val_size = 20_000] to the right
    test_val_size = 20_000
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

    print('\n>>> Default Split')
    print('\nTrain time info:')
    print(time_range(data, default_split['train_idx']))
    print('\nVal time info:')
    print(time_range(data, default_split['val_idx']))
    print('\nTest time info:')
    print(time_range(data, default_split['test_idx']))

    for s in sliding_window_splits:
        print('\n>>> Sliding Window Split')
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
        name='homesite-insurance', 
        task_type='binclass',
        data=data_parts,
        splits=all_splits,
    )


if __name__ == "__main__":
    main()
