# Ecom Offers (Acquire Valued Shoppers by DMDave) preprocessign and splitting script
import gzip

import numpy as np
import polars as pl
import kaggle
from loguru import logger

import lib
from preprocessing.util import save_dataset

TMP_DATA_PATH = lib.PROJECT_DIR/'preprocessing/tmp/ecom-offers'
TMP_DATA_PATH.mkdir(exist_ok=True, parents=True)

def main():
    kaggle.api.competition_download_files('acquire-valued-shoppers-challenge', path=TMP_DATA_PATH)
    lib.unzip(TMP_DATA_PATH/'acquire-valued-shoppers-challenge.zip')

    # ======================================================================================
    # >>> Preprocessing <<<
    # ======================================================================================

    logger.info('Preprocessing ecom offers dataset')

    data_offers = pl.read_csv(gzip.open(TMP_DATA_PATH/'offers.csv.gz').read())
    data_train_history = pl.read_csv(gzip.open(TMP_DATA_PATH/'trainHistory.csv.gz').read()).with_columns(
        pl.col('offerdate').str.strptime(pl.Date)
    )
    data_transactions = pl.read_csv(gzip.open(TMP_DATA_PATH/'transactions.csv.gz').read()).with_columns(
        pl.col('date').str.strptime(pl.Date)
    )

    data_train_offer = (
        data_train_history
        .join(data_offers, on='offer')
        .with_columns(pl.col('repeater').eq('t').cast(pl.Int32).alias('target'))
        .drop('repeater')
    )

    data_transactions = (
        data_transactions
        .join(data_train_offer, on='id')
        .with_columns((pl.col('offerdate') - pl.col('date')).dt.total_days().alias('date_diff'))
    )

    filters = {
        'bought_company': pl.col('company').eq(pl.col('company_right')),
        'bought_category': pl.col('category').eq(pl.col('category_right')),
        'bought_brand': pl.col('brand').eq(pl.col('brand_right')),
    }

    date_diffs = [
        pl.col('date_diff').lt(d).alias(f'{d}') for d in [1, 3, 7, 14, 21, 28, 60, 90, 120, 150, 180]
    ]

    # Expressions used in aggregation of transaction histories

    # first works, because data_train has one unique id for each offer, thus after join all data (like
    # target, offervalue come from training data and have one uniuqe value)

    exprs = [
        pl.col('purchaseamount').cast(pl.Float64).sum().alias('total_spend'),
        pl.col('target').first(),
        pl.col('offervalue').first(),
        pl.col('offerdate').first(),
        pl.col('offerdate').first().dt.weekday().alias('day_of_week'),
        pl.col('offerdate').first().dt.day().alias('day_of_month'),
        pl.col('offerdate').first().dt.ordinal_day().alias('day_of_year'),

    ]

    exprs += sum([
        [
            fv.sum().alias(f'has_{fn}'),
            pl.col('purchasequantity').cast(pl.Float64).filter(fv).alias(f'has_{fn}_q').sum(),
            pl.col('purchaseamount').cast(pl.Float64).filter(fv).sum().alias(f'has_{fn}_a')
        ]
        for fn,fv in filters.items()
    ], [])

    exprs += sum([
        [
            fv.and_(d).sum().alias(f'has_{fn}_{d.meta.output_name()}'),
            pl.col('purchasequantity').cast(pl.Float64).filter(fv.and_(d)).alias(f'has_{fn}_q_{d.meta.output_name()}').sum(),
            pl.col('purchaseamount').cast(pl.Float64).filter(fv.and_(d)).alias(f'has_{fn}_a_{d.meta.output_name()}').sum() 
        ]
        for d in date_diffs for fn,fv in filters.items()
    ], [])


    data = (
        data_transactions
        .group_by('id')
        .agg(*exprs)
        .sort(by='offerdate')
    )

    X_num_data = data.select(
        *[pl.col(c) for c in data.columns if c not in ['id', 'target', 'offerdate']]
    ).cast(pl.Float32)

    X_bin_data = data.select(

        *[pl.col(f'has_bought_{c}').eq(0).alias(f'never_bought_{c}') for c in ['company', 'category', 'brand']],
        pl.col('has_bought_brand').ne(0).and_(pl.col('has_bought_category').ne(0)).and_(pl.col('has_bought_company').ne(0)).alias('has_bought_brand_company_category'),
        pl.col('has_bought_brand').ne(0).and_(pl.col('has_bought_category').ne(0)).alias('has_bought_brand_category'),
        pl.col('has_bought_brand').ne(0).and_(pl.col('has_bought_company').ne(0)).alias('has_bought_brand_company'),
    ).cast(pl.Float32)

    X_meta_data = data.select(pl.col('offerdate').alias('timestamp')).cast(pl.Int64)
    Y_data = data.select(pl.col('target')).cast(pl.Int64)

    # ======================================================================================
    # >>> Default task split <<<
    # ======================================================================================

    # Last 5 days of offers
    default_test_idx = data.with_row_index().filter(
        pl.col('offerdate').ge(pl.datetime(2013, 4, 25))
    )['index'].to_numpy()

    # Second to last 5 days of offers
    default_val_idx = data.with_row_index().filter(
        pl.col('offerdate').ge(pl.datetime(2013, 4, 20)) &
        pl.col('offerdate').lt(pl.datetime(2013, 4, 25))
    )['index'].to_numpy()

    # The rest is train
    default_train_idx = data.with_row_index().filter(
        pl.col('offerdate').lt(pl.datetime(2013, 4, 20))
    )['index'].to_numpy()

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
        print('Start:', df[split[0].item()]['offerdate'].item())
        print('End:', df[split[-1].item()]['offerdate'].item())

        return df[split[-1].item()]['offerdate'].item() - df[split[0].item()]['offerdate'].item()

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
        name='ecom-offers',
        task_type='binclass',
        data=data_parts,
        splits=all_splits,
    )


if __name__ == "__main__":
    main()
