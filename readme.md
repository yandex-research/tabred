# TabReD: A Benchmark of Tabular Machine Learning in-the-Wild

Code for the tabred paper. Providing dataset preprocessing and code for reproducing experiments from the paper.

> Benchmarks that closely reflect downstream application scenarios are essential for the streamlined adoption of new research in tabular machine learning (ML). In this work, we examine existing tabular benchmarks and find two common characteristics of industry-grade tabular data that are underrepresented in the datasets available to the academic community. First, tabular data often changes over time in real-world deployment scenarios. This impacts model performance and requires time-based train and test splits for correct model evaluation. Yet, existing academic tabular datasets often lack timestamp metadata to enable such evaluation. Second, a considerable portion of datasets in production settings stem from extensive data acquisition and feature engineering pipelines. For each specific dataset, this can have a different impact on the absolute and relative number of predictive, uninformative, and correlated features, which in turn can affect model selection. To fill the aforementioned gaps in academic benchmarks, we introduce TabReD -- a collection of eight industry-grade tabular datasets covering a wide range of domains from finance to food delivery services. We assess a large number of tabular ML models in the feature-rich, temporally-evolving data setting facilitated by TabReD. We demonstrate that evaluation on time-based data splits leads to different methods ranking, compared to evaluation on random splits more common in academic benchmarks. Furthermore, on the TabReD datasets, MLP-like architectures and GBDT show the best results, while more sophisticated DL models are yet to prove their effectiveness.


## TabReD datasets 

You can download and preprocess TabReD datasets by running scripts from the
[`./preprocessing`](./preprocessing) directory. For Kaggle datasets you shoul enroll the respective
competitions and have a Kaggle account.

Here is the initial rendition of TabReD with links to datasets and basic metadata:

| Dataset            | Features | Task           | Instances Used | Instances Available | Link                                                                                       |
|--------------------|----------|----------------|----------------|---------------------|--------------------------------------------------------------------------------------------|
| Homesite Insurance | 299      | Classification | 260,753        | -                   | [Competition](https://www.kaggle.com/competitions/homesite-quote-conversion)               |
| Ecom Offers        | 119      | Classification | 160,057        | -                   | [Competition](https://www.kaggle.com/c/acquire-valued-shoppers-challenge)                  |
| Homecredit Default | 696      | Classification | 381,664        | 1,526,659           | [Competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) |
| Sberbank Housing   | 392      | Regression     | 28,321         | -                   | [Competition](https://www.kaggle.com/competitions/sberbank-russian-housing-market)         |
| Cooking Time       | 192      | Regression     | 319,986        | 12,799,642          | [Dataset](https://www.kaggle.com/datasets/pcovkrd84mejm/cooking-time)                      |
| Delivery ETA       | 223      | Regression     | 416,451        | 17,044,043          | [Dataset](https://www.kaggle.com/datasets/pcovkrd84mejm/delivery-eta)                      |
| Maps Routing       | 986      | Regression     | 340,981        | 13,639,272          | [Dataset](https://www.kaggle.com/datasets/pcovkrd84mejm/tabred-weather)                    |
| Weather            | 103      | Regression     | 423,795        | 16,951,828          | [Dataset](https://www.kaggle.com/datasets/pcovkrd84mejm/maps-routing)                      |

## Repository Structure

- [`./preprocessing`](./preprocessing) directory contains preprocessing scripts for all the datasets
- [`./exp`](./exp) all exeperiment logs are in this folder
- [`./bin`](./bin) scripts for launching the experiments
- [`./lib`](./lib) library, dataloading, utilities 

## Environment

There are two environments: one for local development on machines without gpus -
`tabred-env-local.yaml`, another for the machines with GPUs `tabred-env.yaml`.

To create the environment with all the dependencies run `micromamba create -f` with the env file of
choice (for example `micromamba create -f tabred-env.yaml` on a server with GPUs).

## Example

To reproduce results for the MLP on the maps-routing dataset.

1. Create an environment
2. Create dataset (run preprocessing script)
3. Run `export CUDA_VISIBLE_DEVICES=0` (or whatever device you like)
4. Run `python bin/go.py exp/mlp/maps-routing/tuning.toml --force` (force, deletes the existing outputs)

## Dataset Details

There is also a [datasheet](./datasheet.md) for the benchmark.
