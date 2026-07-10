# Preprocessing - common functions
import enum
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import requests

# >>>>> Copy-paste from lib, so we do not import torch and other libs


class TaskType(enum.Enum):
    REGRESSION = "regression"
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"


PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"


def dump_json(x: Any, path: str | Path, **kwargs) -> None:
    kwargs.setdefault("indent", 4)
    Path(path).write_text(json.dumps(x, **kwargs) + "\n")


def download(url: str, path: Path):
    response = requests.get(url)
    with open(path, "wb") as f:
        for chunk in response.iter_content(16384):
            f.write(chunk)


def unzip(path: Path):
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(path.parent)


# <<<<<


def save_dataset(
    name: str,
    task_type: str | TaskType,
    data: dict[str, pl.DataFrame],
    splits: dict[str, dict[str, np.ndarray]],
):
    task_type = TaskType(task_type)

    out_path = DATA_DIR / name
    preview_path = out_path / "csv"
    out_path.mkdir(exist_ok=True)
    preview_path.mkdir(exist_ok=True)

    dtypes = {
        "X_num": np.float32,
        "X_bin": np.float32,
        "X_cat": np.int64,
        "X_meta": np.int64,
        "Y": np.float32 if task_type == TaskType.REGRESSION else np.uint64,
    }

    # create an info.json file with high level meta-data
    info = {
        "name": name,
        "task_type": task_type.value,
    }

    if task_type == TaskType.BINCLASS:
        info["score"] = "roc-auc"

    dump_json(info, out_path / "info.json")

    # All data is stored in a binary format on disk
    # Small csv preview is also stored

    for n, v in data.items():
        v_np = v.to_numpy().astype(dtypes[n])
        if n == "Y":
            v_np = np.squeeze(v)

        np.save(out_path / f"{n}.npy", v_np, allow_pickle=False)
        v.head(20).write_csv(preview_path / f"{n}.csv", include_header=True)

    for split, idxs in splits.items():
        split_path = out_path / f"split-{split}"
        split_path.mkdir(exist_ok=True)

        assert (
            len(set(idxs["train_idx"]) & set(idxs["val_idx"]) & set(idxs["test_idx"]))
            == 0
        ), "Splits intersect, FIX this"

        for n, v in idxs.items():
            np.save(split_path / f"{n}.npy", v, allow_pickle=False)
