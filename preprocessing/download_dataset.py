# /// script
# requires-python = ">=3.11.9"
# dependencies = [
#     "category_encoders==2.6.3",
#     "joblib==1.4.0",
#     "kaggle==1.7.4.5",
#     "loguru==0.7.2",
#     "matplotlib==3.8.0",
#     "numpy==1.26.4",
#     "pandas==2.2.1",
#     "polars==0.20.19",
#     "pyarrow==15.0.2",
#     "scikit-learn==1.4.1.post1",
#     "scipy==1.13.0",
#     "tqdm==4.66.2",
#     "tomli==2.0.1",
#     "tomli-w==1.0.0",
#     "pytest==8.2.1",
#     "xlsx2csv==0.8.1",
# ]
# ///

import argparse

from cooking_time import main as download_cooking
from delivery_eta import main as download_delivery
from ecom_offers import main as download_ecom
from homecredit import main as download_homecredit
from homesite import main as download_homesite
from maps_routing import main as download_maps_routing
from sberbank_housing import main as download_sberbank_housing
from weather import main as download_weather

DATASET_TO_COMMAND = {
    "cooking-time": download_cooking,
    "delivery-eta": download_delivery,
    "ecom-offers": download_ecom,
    "homecredit": download_homecredit,
    "homesite": download_homesite,
    "maps_routing": download_maps_routing,
    "sberbank-housing": download_sberbank_housing,
    "weather": download_weather,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--original-path", type=str, required=False)

    args = parser.parse_args()
    assert args.dataset in DATASET_TO_COMMAND.keys()
    DATASET_TO_COMMAND[args.dataset](args.original_path)


if __name__ == "__main__":
    main()
