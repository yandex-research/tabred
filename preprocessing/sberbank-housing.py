# Sberbank Housing preprocessing and splitting script

import zipfile

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import kaggle
from sklearn.preprocessing import OrdinalEncoder
from loguru import logger

import lib
from preprocessing.util import save_dataset


TMP_DATA_PATH = lib.PROJECT_DIR/'preprocessing/tmp/sberbank-housing'
TMP_DATA_PATH.mkdir(exist_ok=True, parents=True)

def main():

    kaggle.api.competition_download_files('sberbank-russian-housing-market', path=TMP_DATA_PATH)
    lib.unzip(TMP_DATA_PATH/'sberbank-russian-housing-market.zip')
    lib.download("https://storage.googleapis.com/kaggle-forum-message-attachments/190521/6630/BAD_ADDRESS_FIX.xlsx", TMP_DATA_PATH/'BAD_ADDRESS_FIX.xlsx')


    # ======================================================================================
    # >>> Preprocessing <<<
    # ======================================================================================
    logger.info('Preprocessing sberbank housing dataset')

    data = pl.read_csv(
        zipfile.ZipFile(TMP_DATA_PATH/'train.csv.zip').read('train.csv'),
        null_values=["NA"],
        infer_schema_length=30_000
    ).with_columns(
        pl.col('timestamp').str.strptime(pl.Date)
    )

    data_macro = pl.read_csv(
        zipfile.ZipFile(TMP_DATA_PATH/'macro.csv.zip').read('macro.csv'),
        null_values=["NA"],
        infer_schema_length=30_000
    ).with_columns(
        pl.col('timestamp').str.strptime(pl.Date),
        pl.col('child_on_acc_pre_school').str.replace(',', '.').cast(pl.Float32, strict=False),
        pl.col('modern_education_share').str.replace(',', '.').cast(pl.Float32),
        pl.col('old_education_build_share').str.replace(',', '.').cast(pl.Float32),
    ).drop('provision_retail_space_modern_sqm') # this feature has one value except for nulls

    # Tverskoe issue fix (something like 600 properties have wrong location features),
    # details in preprocessing/readme.md

    data_fixup = pl.read_excel(
        TMP_DATA_PATH/'BAD_ADDRESS_FIX.xlsx',
        read_options=dict(null_values=["NA"])
    )

    data = data.filter(
        pl.col('kremlin_km').ne(pl.col('kremlin_km').min()) |
        pl.col('id').is_in(data_fixup['id'])
    ).update(data_fixup, on='id')

    # Plot outliers
    fig = plt.figure()
    plt.hist(data.filter(pl.col('price_doc').lt(10_000_000))['price_doc'], bins=100, label='Prices <= 10kk')
    plt.vlines([2_000_000], ymin=[0], ymax=[800], color='red', linewidth=2, label='Outliers from discussions')
    plt.vlines([3_000_000], ymin=[0], ymax=[450], color='red', linewidth=2)
    plt.vlines([1_000_000], ymin=[0], ymax=[800], color='red', linewidth=2)
    plt.legend()
    plt.xticks(range(1_000_000, 10_000_000, 1_000_000), [f'{i}kk' for i in range(1,10)])
    plt.xlabel('price_doc')
    plt.ylabel('count')
    plt.show()
    fig.savefig('sberbank-outliers.png', dpi=300, bbox_inches='tight')

    # Filtering errors in *_sq features
    data = data.filter(
        pl.col('full_sq').gt(5.0) &
        pl.col('full_sq').ne(5326.0) &

        # These ~=1000 instances are most likely outliers, that appear only in the middle of the dataset
        # (in terms of timestamps)
        pl.col('price_doc').gt(1_000_000) &
        pl.col('price_doc').ne(2_000_000) &
        pl.col('price_doc').ne(3_000_000)
    )

    data = data.join(data_macro, on="timestamp")
    data = data.with_columns(
        pl.col('timestamp').dt.weekday().alias('day_of_week'),
        pl.col('timestamp').dt.day().alias('day_of_month'),
        pl.col('timestamp').dt.ordinal_day().alias('day_of_year'),
        pl.col('timestamp').dt.month().alias('month'),
        pl.col('timestamp').dt.year().alias('year'),
    )

    # ======================================================================================
    # >>> Split into num, cat, bin <<<
    # ======================================================================================

    cat_cols = [
        'ID_railroad_station_walk',
        'ID_railroad_station_avto',
        'ID_big_road1',
        'ID_big_road2',
        'ID_railroad_terminal',
        'ID_bus_terminal',
        'ecology',
        'material',
        'state',
        'sub_area',
    ]

    bin_cols = [data.columns[i] for i in np.flatnonzero(data.select(pl.col('*').n_unique() == 2).to_numpy())]

    assert len(set(cat_cols) & set(bin_cols)) == 0

    num_cols = [
        c for c in data.columns
        if c not in ['price_doc', 'id', 'timestamp'] + cat_cols + bin_cols
    ]

    target_cols = [(pl.col('price_doc') / pl.col('full_sq')).log()]
    meta_cols = ['timestamp']

    X_bin_data = data.select(pl.col(bin_cols).rank('dense').cast(pl.Float32) - 1)

    X_cat_data_np = data.select(cat_cols).to_numpy()
    X_cat_data_np = OrdinalEncoder(min_frequency=1/100, encoded_missing_value=-1).fit_transform(X_cat_data_np)
    max_values = X_cat_data_np.max(axis=0)
    for c in range(X_cat_data_np.shape[1]):
        X_cat_data_np[X_cat_data_np[:, c] == -1, c] = max_values[c] + 1
    X_cat_data = pl.DataFrame(X_cat_data_np, schema={c: pl.Int64 for c in cat_cols})

    X_num_data = data.select(pl.col(num_cols)).cast(pl.Float32)
    X_meta_data = data.select(meta_cols).cast(pl.Int64)
    Y_data = data.select(target_cols).cast(pl.Float32)

    # ======================================================================================
    # >>> Default task split <<<
    # ======================================================================================

    # Last 6+ month
    default_test_idx = data.with_row_index().filter(
        pl.col('timestamp').ge(pl.datetime(2014, 12, 1))
    )['index'].to_numpy()

    # Prior 6 month
    default_val_idx = data.with_row_index().filter(
        pl.col('timestamp').ge(pl.datetime(2014, 6, 30)) &
        pl.col('timestamp').lt(pl.datetime(2014, 12, 1))
    )['index'].to_numpy()

    # The rest is train
    default_train_idx = data.with_row_index().filter(
        pl.col('timestamp').lt(pl.datetime(2014, 6, 30))
    )['index'].to_numpy()

    default_split = {
        'train_idx': default_train_idx, 'val_idx': default_val_idx, 'test_idx': default_test_idx,
    }

    # ======================================================================================
    # >>> Sliding window splits <<<
    # ======================================================================================

    # Create 3 windowed time splits
    # We shift each window [val_size = 20_000] to the right
    test_val_size = 4_500
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
        name='sberbank-housing',
        task_type='regression',
        data=data_parts,
        splits=all_splits,
    )



if __name__ == "__main__":
    main()
