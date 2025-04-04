# TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks

:scroll: [arXiv](https://arxiv.org/abs/2406.19380)
&nbsp; :books: [Other tabular DL projects](https://github.com/yandex-research/rtdl)

> [!IMPORTANT]
> Check out the new tabular DL model: [TabM](https://github.com/yandex-research/tabm) (SoTA on TabReD)

*TL;DR: TabReD is a new benchmark of industry-grade tabular datasets with temporally evolving and feature-rich real-world datasets*

> Advances in machine learning research drive progress in real-world applications. To ensure this progress, it is important to understand the potential pitfalls on the way from a novel method's success on academic benchmarks to its practical deployment. In this work, we analyze existing tabular benchmarks and find two common characteristics of tabular data in typical industrial applications that are underrepresented in the datasets usually used for evaluation in the literature. First, in real-world deployment scenarios, distribution of data often changes over time. To account for this distribution drift, time-based train/test splits should be used in evaluation. However, popular tabular datasets often lack timestamp metadata to enable such evaluation. Second, a considerable portion of datasets in production settings stem from extensive data acquisition and feature engineering pipelines. This can have an impact on the absolute and relative number of predictive, uninformative, and correlated features compared to academic datasets. In this work, we aim to understand how recent research advances in tabular deep learning transfer to these underrepresented conditions. To this end, we introduce TabReD -- a collection of eight industry-grade tabular datasets. We reassess a large number of tabular ML models and techniques on TabReD. We demonstrate that evaluation on time-based data splits leads to different methods ranking, compared to evaluation on random splits, which are common in current benchmarks. Furthermore, simple MLP-like architectures and GBDT show the best results on the TabReD datasets, while other methods are less effective in the new setting. 

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
| Maps Routing       | 986      | Regression     | 340,981        | 13,639,272          | [Dataset](https://www.kaggle.com/datasets/pcovkrd84mejm/maps-routing)                      |
| Weather            | 103      | Regression     | 423,795        | 16,951,828          | [Dataset](https://www.kaggle.com/datasets/pcovkrd84mejm/tabred-weather)                    |

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
