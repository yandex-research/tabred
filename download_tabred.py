# /// script
# requires-python = ">=3.11.9"
# dependencies = [
#     "kaggle==1.7.4.5",
# ]
# ///

"""
Downloads only the TabReD splits used in the paper. Not the full datasets.

```
uv run download_tabred.py
```
"""

import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi


class RulesNotAcceptedError(Exception):
    """Raised when a user has not accepted the rules."""

    def __init__(self, competition: str):
        self.competition = competition
        self.link = "https://www.kaggle.com/code/rototo/get-tabred/notebook"

    def __str__(self) -> str:
        return f"{self.competition}. Follow {self.link}, click Copy&Edit and accept the rules."  # noqa: E501


def main():
    api = KaggleApi()
    api.authenticate()

    print(">>> Checking if the rules are accepted")
    for competition in [
        "acquire-valued-shoppers-challenge",
        "home-credit-credit-risk-model-stability",
        "homesite-quote-conversion",
        "sberbank-russian-housing-market",
    ]:
        try:
            api.competition_list_files(competition)
        except Exception as e:
            if "Forbidden" in str(e):
                raise RulesNotAcceptedError(competition)
            raise e

    print(">>> Downloading started")
    api.kernels_output("rototo/get-tabred", path="./")


if __name__ == "__main__":
    main()
