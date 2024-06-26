Datasheet
=========

This datasheet describes a set of tabular datasets currated and introduced under the TabReD benchark. We provide details for all datasets in one datasheet. For datasets from kaggle competitions we refer to details from the competition descrpition, and forums. For newly introduced datasets we describe data sources, composition and preprocessing.

## Motivation

**`Q`: For what purpose was the dataset created?** Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.

**`A`**: The benchmark was introduced to bridge the gap between academic tabular ML datasets and datasets used in real-world (industrial) ML applications. In particular, all datasets in the TabReD benchmark come with timestamps, and temporal-splits. As we argue in the accompanying [paper](https://arxiv.org/TODO), evaluation under gradual temporal shift closer resambles realistic tabular ML application scenarious, but was not used prior to TabReD (either due to the lack of timestamps, or adopting standard protocols from prior work). Datasets curated under the TabReD benchark also have more features and samples available than many academic tabular datasets.

---

**`Q`: Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?**

**`A`**: Datasets Homecredit Default, Homesite Insurance, Sberbank Housing and DMDave Offers were created and shared on the kaggle platform by competition organizers and people from respective organizations ([[1](), [2](), [3](), [4]()]]). The preprocessing was done by the team from Yandex Research (Ivan Rubachev, Nikolay Kartashev and Artem Babenko at the time of writing). The remaining four datasets (Cooking Time, Delivery ETA, Maps Routing and Weather) were created and prepared for publication by ML practitioners at Yandex and the Yandex Research team.

---

**`Q`: Who funded the creation of the dataset?** If there is an associated grant, please provide the name of the grantor and the grant name and number.

**`A`**: All dataset authors were employed by Yandex and HSE University for the duration of TabReD
benchmark creation. No grants were involved in preparation of the project.

---

**`Q`: Any other comments?**

**`A`**: No.

Composition
-----------

**`Q`: What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?** Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.

**`A`**: Each dataset in the TabReD benchmark is comprised of table rows with various types of features -- numeric, categorical or binary. Features represent real-world objects, like bank or insurance company clients, online restaurant orders, navigation route features, etc. Features are measurements or aggregations of such measurements like physical weather model forecasts, past delivery time aggregations, user purchase historical aggregations, etc.
    
Here are the specifics for individual datasets:
    
- *Homesite Insurance*. Each instance in Homesite dataset represents a potential insurance customer. Features include include specific coverage information, sales information, personal information, property information, and geographic information. The task for each instance is to predict whether the customer will purchase a given quote (variable is known, as it is from historical data for quote conversion from homesite). More dataset information is on the [competition page](https://www.kaggle.com/competitions/homesite-quote-conversion/overview))
    
- *Ecom Offers*. Each instance in the dataset represents a promotion offerings to online store customers. The task is to predict which customers would redeem the offer based on their past transaction history. The features for each instance are time and category based aggregations (e.g. how much of the offer brand items the customer bought in the past 30 days). Other dataset details are available on the [competition page](https://www.kaggle.com/competitions/acquire-valued-shoppers-challenge/overview)

- *Homecredit Default*. Each instance in the dataset represent a bank client. The task is to predict wether the client would default on a loan based on their historical data (both internal bank data and credit bureau history). Feature descriptions are available on the [competiton page](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/overview). We use the preprocessing from one of the top publicly available solutions with slight modifications.

- *Sberbank Housing*. Each instance in the dataset represents a property on sale in Moscow housing market, the task is predicting the price for which property would be sold. The features include property information (e.g. living square meters), property neighborhoud information (e.g. number of cafe's and building density in the surrounding area) and macro-economic features. All feature names are available on the [competition page](https://www.kaggle.com/c/sberbank-russian-housing-market/overview). 

- *Delivery ETA*. Each instance in the dataset represents a potential order in an online groccery delivery service, the task is estimating courier time of arrival. The features are current courier availability for the location, geo and navigation based time-estimate features and various aggregation features based on prior delivery times and times of day.

- *Cooking Time*. Each instance in the dataset represent an order to one of many restaurants. The task is to predict the time it would take to cook an order. The features are the order contents (number of items, weight, etc.) and various aggregations features based on historical cooking data for particular brands, restaurants and times of week and day.

- *Maps Routing*. Each instance in the dataset represent a query to the navigation system, the task is to predict time of arrival based on features computed from the road graph and the built route. Features are static road information and dynamic road load.
    
- *Weather Forecasting*. Each instance in the Weather Forecasting dataset comes from logs of a production weather forecasting system. The features are from the climate models. The feature include climate preasure, cloud coverage, temperature estimates and more heterogeneous features. The list of features is a subset of the features used in the dataset from the shift 1.0 challenge (see the [paper](https://arxiv.org/abs/2107.07455) for details)

---

**`Q`: How many instances are there in total (of each type, if appropriate)?**

**`A`**: Here are the number of instances and feature types for each dataset in TabReD

We randomly subsample large datasets, the ibstances used column shows the number of instances used in the benchmark datasets. Inspite of using smsller datasets we make all data available for future research.

| Task               | Features | Instances Used | Instances Available |
|--------------------|----------|----------------|---------------------|
| Homesite Insurance | 299      | 260,753         | -                   |
| Ecom Offers        | 119      | 160,057         | -                   |
| Homecredit Default | 696      | 381,664         | 1,526,659           |
| Sberbank Housing   | 392      | 28,321          | -                   |
| Cooking Time       | 192      | 319,986         | 12,799,642          |
| Delivery ETA       | 223      | 416,451         | 17,044,043          |
| Maps Routing       | 986      | 340,981         | 13,639,272          |
| Weather            | 103      | 423,795         | 16,951,828          |

---

**`Q`: Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?** If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable)

**`A`**: All datasets contain random (time-based) samples of the underlying populations: insurance company website visitors, DMDave website users, homecredit users, house properties in the Moscow market, restaurant orders, grocerry delivery orders, navigation system queries and weather station measurements (weather is measured only for the "A (Tropical)" climate zone by the [Köppen climate classification](https://en.wikipedia.org/wiki/K%C3%B6ppen_climate_classification))

---

**`Q`: What data does each instance consist of?** "Raw" data (e.g., unprocessed text or images) or features? In either case, please provide a description.

**`A`**: Each instance consists of features. Feature descriptions are available for the Sberbank Housing, Ecom Offers, HomeCredit Default and Weather datasets. Homesite Insurance, Cooking Time, Maps Routing and Delivery ETA features are anonymized (no feature names are available), the general feature-generation process is described in the previous answer on "what instances in the dataset comprise".

---

**`Q`: Is there a label or target associated with each instance?** If so, please provide a description.

In all datasets targets come from observing the real world:
- *Ecom Offers* whether customer bought the product from the coupon offer
- *Homesite Insurance* whether a customer bought the insurance offer
- *HomeCredit Default* whether a customer defaulted on a loan
- *Sberbank Housing* the price of a deal in the housing market 
- *Cooking Time* the time it took to cook an order in reality 
- *Drlivery ETA* real delivery time
- *Maps Routing* the time it took for a quried ride. The target is transformed and is measured in seconds per kilometer. The route lenght is provided with the data
- *Weather* the real weather reading for the time of prediction

**`Q`: Is any information missing from individual instances?** If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.

**`A`**: All datasets include some missing values in features. The exact reasons are unknown.

---

**`Q`: Are relationships between individual instances made explicit (e.g., users' movie ratings, social network links)?** If so, please describe how these relationships are made explicit.

**`A`**: There are no clear relationships between individual instances in the introduced datasets.

---

**`Q`: Are there recommended data splits (e.g., training, development/validation, testing)?** If so, please provide a description of these splits, explaining the rationale behind them.

**`A`**: We do provide recommended time-based validation and test splits. Split details for each dataset are in the [`./preprocessing`](./preprocessing) directory.

---

**`Q`: Are there any errors, sources of noise, or redundancies in the dataset?** If so, please provide a description.

**`A`**: Sberbank Housing contained noisy labels, which we cleaned, by consulting the Kaggle public forums. In the delivery logs datasets we filter a few instances with less than a minute delivery time, same for the cooking data. Other datasets do not contain any errors we are aware of, except for user input errors or missing data.

---

**`Q`: Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?** If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.

**`A`**: Kaggle datasets are downloaded and preprocessed after accepting competitino terms. Yandex production-based datasets (Maps, Delivery, Cooknig and Weather) are available to download from the Kaggle datasets platform.

---

**`Q`: Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor patient confidentiality, data that includes the content of individuals' non-public communications)?** If so, please provide a description.

**`A`**: No. Kaggle datasets are open, without any confidential information (with measures taken to protect and remove confidentital information if necessary). Four newly introduced datasets are free from any personal or confidential data.

---

**`Q`: Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?** If so, please describe why.

**`A`**: No.

---

**`Q`: Does the dataset relate to people?** If not, you may skip the remaining questions in this section.

**`A`**: Some datasets relate to people (HomeCredit, Homesite, DMDave).

---

**`Q`: Does the dataset identify any subpopulations (e.g., by age, gender)?** If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.

**`A`**: HomeCredit Default dataset contains some features that could identify subpopulations. E.g. The primary language of the client, gender of the client. Other datasets do not containt such features.

---

**`Q`: Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?** If so, please describe how.

**`A`**: All features in the datasets are transformed and anonymized, to the best of our knowledge it is not possible to identify or match individuals neither directly nor inderctly.

---

**`Q`: Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?** If so, please provide a description.

**`A`**: Datasets do not contain such sensitive data.

---

**`Q`: Any other comments?**

**`A`**: No.

Collection process
------------------

**`Q`: How was the data associated with each instance acquired?** Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.

**`A`**: In all datasets dat comes from observing the real-world (e.g. collecting service usage logs, gathering existing economics data, etc.).

---

**`Q`: What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?** How were these mechanisms or procedures validated?

**`A`**: The data was collected in-house by the respective service ML engineers.

---

**`Q`: If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?**

**`A`**: All datasets are samples over finite timeframes. For craeting smaller subsets of the datasets we use random subsampling.

---

**`Q`: Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?**

**`A`**: Paid ML professionals were involved in collecting all datasets and preparing either for Kaggle competitions, or for the TabReD publication.

---

**`Q`: Over what timeframe was the data collected?** Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created.

**`A`**: Here are the timeframes for each of the datasets:

| Task               | Start Time | End Time   |
|--------------------|------------|------------|
| Homesite Insurance | 2013-01-01 | 2015-05-18 |
| Ecom Offers        | 2013-03-01 | 2013-04-30 |
| Homecredit Default | 2019-01-01 | 2020-10-05 |
| Sberbank Housing   | 2011-08-20 | 2015-06-30 |
| Cooking Time       | 2023-11-15 | 2024-01-03 |
| Delivery ETA       | 2023-10-20 | 2024-01-25 |
| Weather            | 2022-07-01 | 2023-07-30 |
| Maps Routing       | 2023-11-01 | 2023-12-04 |

---

**`Q`: Were any ethical review processes conducted (e.g., by an institutional review board)?** If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.

**`A`**: For the four newly introduced datasets, we did the internal review to ensure no private or identifyiable information would be published (this resulted in protocols for excluding features and transforming feature values: e.g. we excluded some location and id features and encoded categorical variable and transformed numerical features). For the Kaggle datasets we do not know of such procedures.

---

**`Q`: Does the dataset relate to people? If not, you may skip the remainder of the questions in this section.**

**`A`**: Some datasets relate to people. (Homesite, HomeCredit, DMDave Ecom Offers)

---

**`Q`: Did you collect the data from the individuals in question
directly, or obtain it via third parties or other sources (e.g.,
websites)?**

**`A`**: We obtained said datasets from Kaggle, the original data is based on internal company data and logs of user interractions.

---

**`Q`: Were the individuals in question notified about the data collection?** If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.

**`A`**: There is no detailed information on this on the competition websites.

---

**`Q`: Did the individuals in question consent to the collection and use of their data?** If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.

**`A`**: There is no detailed information on this on the competition websites.

---

**`Q`: If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or forcertain uses?** If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).

**`A`**: There is no detailed information on this on the competition websites.

---

**`Q`: Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?** If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.

**`A`**: The datasets used in the benchmark do not provide personal data. Proposed prediction tasks do not increase privacy risks.

---

**`Q`: Any other comments?**

**`A`**: No.

Preprocessing / cleaning / labeling
-----------------------------------

**`Q`: Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?** If so, please provide a description. If not, you may skip the remainder of the questions in this section.

**`A`**: There was some preprocessing, including feature engineering for the Kaggle datasets and some data filtering. The preprocessing is described in the the [`./preprocessing`](./preprocessing) directory.

---

**`Q`: Was the "raw" data saved in addition to the
preprocessed/cleaned/labeled data (e.g., to support unanticipated
future uses)?** If so, please provide a link or other access point
to the "raw" data.

**`A`**: The original raw data (e.g. raw logs) are not available. But the raw features data before subsampling, preprocessing and feature engineering is made available.

---

**`Q`: Is the software used to preprocess/clean/label the instances
available?** If so, please provide a link or other access point.

**`A`**: The software used to preprocess clean and split datasets is available. All preprocessing scripts are in the [`./preprocessing`](./preprocessing) directory.

---

**`Q`: Any other comments?**

**`A`**: No.

Uses
----

**`Q`: Has the dataset been used for any tasks already?** If so, please
provide a description.

**`A`**: In the acompannying dataset paper. We propose a benchmark with realistic time-based train/val/test splits. We evaluate recent tabular ML methods on the introduced TabReD benchmark. We also provide an analysis on importance of time splits by comparing methods across different time-based splits and random dataset splits.

---

**`Q`: Is there a repository that links to any or all papers or systems
that use the dataset?** If so, please provide a link or other access
point.

**`A`**: Yes the repository with dataset uses is at https://github.com/yandex-research/tabred

---

**`Q`: What (other) tasks could the dataset be used for?**

**`A`**: Datasets could be used to study model scaling, robustness to temporal shifts, continual learning methods, missing data imputation methods, generative modeling, etc.

---

**`Q`: Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?** For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other undesirable harms (e.g., financial harms, legal risks) If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?

**`A`**: 

The fact that for the Maps Routing, Cooking Time, Delivery ETA and Homesite Insurance the features descriptions are unavailable might limit future uses in human-interpretability of state-of-the-art models on the benchmark. Anothre limitiation is that raw data is not available for all the datasets. Thus future research on feature engineering is limited. Last, the TabReD benchmark is biased towards industry use-cases which do not cover all the important real-world use-cases for tabular ML.

**`Q`: Are there tasks for which the dataset should not be used?** If so,
please provide a description.

We don't see any specific harmful or particularly wrong tasks the datasets could be used for. We encourage further users to use the correct validation and evaluation splits.

**`Q`: Any other comments?**

**`A`**: No.

Distribution
------------

**`Q1`: Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?** If so, please provide a description.

**`Q2`: How will the dataset will be distributed (e.g., tarball on
website, API, GitHub)?** Does the dataset have a digital object
identifier (DOI)?

**`A`**: All datasets will be available through the Kaggle datasets and competitions platform. For the competitions datasets access you need to use a Kaggle account for downloading datasets. For the newly introduced datasets all data is available without restrictions. Newly created datasets could be found under the TabReD kaggle [account](https://www.kaggle.com/pcovkrd84mejm/datasets).

---

**`Q`: When will the dataset be distributed?**

**`A`**: Datasets are already available.

---

**`Q1` Will the dataset be distributed under a copyright or otherintellectual property (IP) license, and/or under applicable terms of use (ToU)?** If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.

**`Q2`: Have any third parties imposed IP-based or other restrictions on
the data associated with the instances?** If so, please describe
these restrictions, and provide a link or other access point to, or
otherwise reproduce, any relevant licensing terms, as well as any
fees associated with these restrictions.

**`A`**: Kaggle competition datasets are subject to competition rules. The newly introduced datasets are provided with the CC BY-NC-SA 4.0 liciense

---

**`Q`: Do any export controls or other regulatory restrictions apply to
the dataset or to individual instances?** If so, please describe
these restrictions, and provide a link or other access point to, or
otherwise reproduce, any supporting documentation.

**`A`**: To our knowledge, no export controls or regulatory restrictions
apply to the dataset.

---

**`Q`: Any other comments?**

**`A`**: No.

Maintenance
-----------

**`Q`: Who is supporting/hosting/maintaining the dataset?**

**`A`**: The dataset will be hosted on Kaggle and supported by the Yandex Research tabular DL team
(Ivan Rubachev, Nikolay Kartashev, Artem Babenko)

---

**`Q`: How can the owner/curator/manager of the dataset be contacted (e.g., email address)?**

**`A`**: Through the GitHub repository discussions <https://github.com/yandex-research/tabred/discussions>

---

**`Q`: Is there an erratum?** If so, please provide a link or other
access point.

**`A`**: An erratum will be hosted in the dataset repo <https://github.com/yandex-research/tabred>.

---

**`Q`: Will the dataset be updated (e.g., to correct labeling errors, add
new instances, delete instances)?** If so, please describe how
often, by whom, and how updates will be communicated to users (e.g.,
mailing list, GitHub)?

**`A`**: The data will be updated, corrected if any errors are to be found. The main communication channel would be the GitHub releases.

---

**`Q`: If the dataset relates to people, are there applicable limits on
the retention of the data associated with the instances (e.g., were
individuals in question told that their data would be retained for a
fixed period of time and then deleted)?** If so, please describe
these limits and explain how they will be enforced.

**`A`**: For all the Kaggle datasets, the competiton rules do apply. For newly introduced datasets there are no restrictions as they do not realte to people.

---

**`Q`: Will older versions of the dataset continue to be supported/hosted/maintained?** If so, please describe how. If not, please describe how its obsolescence will be communicated to users.

**`A`**: Older versions would be clearly indicated through the GitHub releases.

---

**`Q`: If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?** If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.

**`A`**: Users wishing to contribute datasets, other preprocessing pipelines or other enhancements are encouraged to do so via the discussions in the benchmark GitHub repository. <https://github.com/yandex-research/tabred/discussions>

---

**`Q`: Any other comments?**

**`A`**: No.
