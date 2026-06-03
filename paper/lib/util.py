import pickle
import argparse
import zipfile
import datetime
import enum
import importlib
import inspect
import json
import os
import shutil
import sys
import time
import types
import typing
import warnings
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Any, Literal, TypeVar, Union

import plotnine as p9
import pandas as pd
import numpy as np
import tomli
import tomli_w
import torch
import requests
from loguru import logger


KWArgs = dict[str, Any]
JSONDict = dict[str, Any]  # Must be JSON-serializable.

PartKey = str
DataKey = Literal['x_num', 'x_bin', 'x_cat', 'x_meta', 'y']

PROJECT_DIR = Path(__file__).parent.parent
CACHE_DIR = PROJECT_DIR / 'cache'
DATA_DIR = PROJECT_DIR / 'data'
EXP_DIR = PROJECT_DIR / 'exp'

assert PROJECT_DIR.exists()
CACHE_DIR.mkdir(exist_ok=True)

class TaskType(enum.Enum):
    REGRESSION = 'regression'
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'


class PredictionType(enum.Enum):
    LABELS = 'labels'
    PROBS = 'probs'
    LOGITS = 'logits'


class Score(enum.Enum):
    ACCURACY = 'accuracy'
    CROSS_ENTROPY = 'cross-entropy'
    MAE = 'mae'
    R2 = 'r2'
    RMSE = 'rmse'
    ROC_AUC = 'roc-auc'


def get_path(path: str | Path) -> Path:
    path = str(path)
    if path.startswith(":"):
        path = PROJECT_DIR / path[1:]
    return Path(path).absolute().resolve()


def unzip(path: Path):
    with zipfile.ZipFile(path, 'r') as z:
        z.extractall(path.parent)


def download(url: str, path: Path):
    response = requests.get(url)
    with open(path, 'wb') as f:
        for chunk in response.iter_content(16384):
            f.write(chunk)

# ======================================================================================
# >>> MainFunction <<<
# ======================================================================================
# By convention, MainFunction is any function with the following signature:
# MainFunction: (
#     config: JSONDict,
#     output: str | Path,
#     *,
#     force = False,
#     [continue_ = False]
# ) -> None | JSONDict
MainFunction = Callable[..., None | JSONDict]


def start(output: str | Path, *, force: bool = False, continue_: bool = False) -> bool:
    """Start MainFunction."""
    print_sep('=')
    output = get_path(output)
    print(f'[>>>] {output} | {datetime.datetime.now()}')

    if output.exists():
        if force:
            logger.warning('Removing the existing output')
            time.sleep(1.0)
            shutil.rmtree(output)
            output.mkdir()
            return True
        elif not continue_:
            backup_output(output)
            logger.warning('The output already exists!')
            return False
        elif output.joinpath('DONE').exists():
            backup_output(output)
            logger.info('Already done!\n')
            return False
        else:
            logger.info('Continuing with the existing output')
            return True
    else:
        logger.info('Creating the output')
        output.mkdir()
        return True


def create_report(config: dict) -> JSONDict:
    # report is a JSON-serializable Python dictionary
    # for storing arbitrary information about a run.
    report: JSONDict = {}

    # 1. The function's full name (e.g. "bin.xgboost_.main").
    try:
        caller_frame = inspect.stack()[1]
        caller_relative_path = get_path(caller_frame.filename).relative_to(
            PROJECT_DIR
        )
        caller_module = str(caller_relative_path.with_suffix('')).replace('/', '.')
        caller_function_qualname = f'{caller_module}.{caller_frame.function}'
        import_(caller_function_qualname)
        report['function'] = caller_function_qualname
    except Exception as err:
        warnings.warn(
            f'The key "function" will be missing in the report. Reason: {err}'
        )

    # 2. Names of the available CUDA devices.
    report['gpus'] = [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ]

    # 3. The config in a JSON-friendly representation.
    def jsonify(value):
        if value is None or isinstance(value, bool | int | float | str | bytes):
            return value
        elif isinstance(value, list):
            return [jsonify(x) for x in value]
        elif isinstance(value, dict):
            return {k: jsonify(v) for k, v in value.items()}
        else:
            return '<nonserializable>'

    report['config'] = jsonify(config)
    return report


def show_config(config: dict) -> None:
    print_sep()
    pprint(config, sort_dicts=False, width=100)
    print_sep()


def summarize(report: JSONDict) -> JSONDict:
    summary = {'function': report.get('function')}

    if 'best' in report:
        # The gpus info is collected from the best report.
        summary['best'] = summarize(report['best'])
    elif 'gpus' in report:
        summary['gpus'] = report['gpus']

    for key in ['n_parameters', 'best_stage', 'best_epoch', 'tuning_time', 'trial_id']:
        if key in report:
            summary[key] = deepcopy(report[key])

    metrics = report.get('metrics')
    if metrics is not None and 'score' in next(iter(metrics.values())):
        summary['scores'] = {part: metrics[part]['score'] for part in metrics}

    for key in ['n_completed_trials', 'time']:
        if key in report:
            summary[key] = deepcopy(report[key])

    return summary


def run_MainFunction_cli(function: MainFunction, argv: None | list[str] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE')
    parser.add_argument('--force', action='store_true')
    if 'continue_' in inspect.signature(function).parameters:
        can_continue_ = True
        parser.add_argument('--continue', action='store_true', dest='continue_')
    else:
        can_continue_ = False
    args = parser.parse_args(argv)

    # >>> snippet for the internal infrastructure, ignore it
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if snapshot_dir and Path(snapshot_dir).joinpath('CHECKPOINTS_RESTORED').exists():
        assert can_continue_ and args.continue_
    # <<<

    config_path = get_path(args.config)
    assert config_path.exists(), f'The config {config_path} does not exist'
    output = config_path.with_suffix('')
    function(
        load_config(config_path),
        output,
        force=args.force,
        **({'continue_': args.continue_} if can_continue_ else {}),
    )


_LAST_SNAPSHOT_TIME = None


def backup_output(output: Path) -> None:
    """
    A function for the internal infrastructure, ignore it.
    """
    backup_dir = os.environ.get('TMP_OUTPUT_PATH')
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if backup_dir is None:
        assert snapshot_dir is None
        return
    assert snapshot_dir is not None

    try:
        relative_output_dir = output.relative_to(PROJECT_DIR)
    except ValueError:
        return

    for dir_ in [backup_dir, snapshot_dir]:
        new_output = dir_ / relative_output_dir
        prev_backup_output = new_output.with_name(new_output.name + '_prev')
        new_output.parent.mkdir(exist_ok=True, parents=True)
        if new_output.exists():
            new_output.rename(prev_backup_output)
        shutil.copytree(output, new_output)
        # the case for evaluate.py which automatically creates configs
        if output.with_suffix('.toml').exists():
            shutil.copyfile(
                output.with_suffix('.toml'), new_output.with_suffix('.toml')
            )
        if prev_backup_output.exists():
            shutil.rmtree(prev_backup_output)

    global _LAST_SNAPSHOT_TIME
    if _LAST_SNAPSHOT_TIME is None or time.time() - _LAST_SNAPSHOT_TIME > 10 * 60:
        import nirvana_dl.snapshot  # type: ignore[code]

        nirvana_dl.snapshot.dump_snapshot()
        _LAST_SNAPSHOT_TIME = time.time()
        print('The snapshot was saved!')


def finish(output: Path, report: JSONDict) -> None:
    dump_report(output, report)

    # >>> A code block for the internal infrastructure, ignore it.
    JSON_OUTPUT_FILE = os.environ.get('JSON_OUTPUT_FILE')
    if JSON_OUTPUT_FILE:
        try:
            key = str(output.relative_to(PROJECT_DIR))
        except ValueError:
            pass
        else:
            json_output_path = Path(JSON_OUTPUT_FILE)
            try:
                json_data = json.loads(json_output_path.read_text())
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                json_data = {}
            json_data[key] = load_report(output)
            json_output_path.write_text(json.dumps(json_data, indent=4))
            shutil.copyfile(
                json_output_path,
                os.path.join(os.environ['SNAPSHOT_PATH'], 'json_output.json'),
            )
    # <<<

    output.joinpath('DONE').touch()
    backup_output(output)
    print()
    print_sep()
    try:
        print_summary(output)
    except FileNotFoundError:
        pass
    print_sep()
    print(f'[<<<] {output} | {datetime.datetime.now()}')


# ======================================================================================
# >>> output <<<
# ======================================================================================

def load_pickle(path: str | Path) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(x: Any, path: str | Path) -> None:
    with open(path, 'wb') as f:
        pickle.dump(x, f)


def load_config(output_or_config: str | Path) -> JSONDict:
    return tomli.loads(get_path(output_or_config).with_suffix('.toml').read_text())


def load_json(path: str | Path, **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)


def dump_json(x: Any, path: str | Path, **kwargs) -> None:
    kwargs.setdefault('indent', 4)
    Path(path).write_text(json.dumps(x, **kwargs) + '\n')


def dump_config(
    output_or_config: str | Path, config: JSONDict, *, force: bool = False
) -> None:
    config_path = get_path(output_or_config).with_suffix('.toml')
    if config_path.exists() and not force:
        raise RuntimeError(
            'The following config already exists (pass force=True to overwrite it)'
            f' {config_path}'
        )
    config_path.write_text(tomli_w.dumps(config))


def load_report(output: str | Path) -> JSONDict:
    return json.loads(get_path(output).joinpath('report.json').read_text())


def dump_report(output: str | Path, report: JSONDict) -> None:
    dump_json(report, get_path(output) / 'report.json')


def load_summary(output: str | Path) -> JSONDict:
    return json.loads(get_path(output).joinpath('summary.json').read_text())


def print_summary(output: str | Path):
    pprint(load_summary(output), sort_dicts=False, width=60)


def dump_summary(output: str | Path, summary: JSONDict) -> None:
    dump_json(summary, get_path(output) / 'summary.json')


def load_predictions(output: str | Path) -> dict[PartKey, np.ndarray]:
    x = np.load(get_path(output) / 'predictions.npz')
    return {key: x[key] for key in x}


def dump_predictions(
    output: str | Path, predictions: dict[PartKey, np.ndarray]
) -> None:
    np.savez(get_path(output) / 'predictions.npz', **predictions)


def get_checkpoint_path(output: str | Path) -> Path:
    return get_path(output) / 'checkpoint.pt'


def load_checkpoint(output: str | Path, **kwargs) -> Any:
    return torch.load(get_checkpoint_path(output), **kwargs)


def dump_checkpoint(output: str | Path, checkpoint: JSONDict, **kwargs) -> None:
    torch.save(checkpoint, get_checkpoint_path(output), **kwargs)



# ======================================================================================
# >>> other <<<
# ======================================================================================
def configure_libraries():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[code]
    torch.backends.cudnn.allow_tf32 = False  # type: ignore[code]
    torch.backends.cudnn.benchmark = False  # type: ignore[code]
    torch.backends.cudnn.deterministic = True  # type: ignore[code]

    logger.remove()
    logger.add(sys.stderr, format='<level>{message}</level>')


def are_valid_predictions(predictions: dict[PartKey, np.ndarray]) -> bool:
    return all(np.isfinite(x).all() for x in predictions.values())


def print_sep(ch='-'):
    print(ch * 80)


def print_metrics(loss: float, metrics: dict) -> None:
    print(
        f'(val) {metrics["val"]["score"]:.3f}'
        f' (test) {metrics["test"]["score"]:.3f}'
        f' (loss) {loss:.5f}'
    )


def log_scores(metrics: dict) -> None:
    logger.debug(
        f'[val] {metrics["val"]["score"]:.4f} [test] {metrics["test"]["score"]:.4f}'
    )


def import_(qualname: str) -> Any:
    # Example: import_('bin.xgboost_.main')
    try:
        module, name = qualname.rsplit('.', 1)
        return getattr(importlib.import_module(module), name)
    except Exception as err:
        raise ValueError(f'Cannot import "{qualname}"') from err


def get_device() -> torch.device:
    if torch.cuda.is_available():
        assert os.environ.get('CUDA_VISIBLE_DEVICES') is not None, (
            'When CUDA is available, CUDA_VISIBLE_DEVICES must be set explicitly,'
            ' for example: export CUDA_VISIBLE_DEVICES=0'
        )
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def is_oom_exception(err: RuntimeError) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or any(
        x in str(err)
        for x in [
            'CUDA out of memory',
            'CUBLAS_STATUS_ALLOC_FAILED',
            'CUDA error: out of memory',
        ]
    )


T = TypeVar('T')


def run_cli(fn: Callable[..., T]) -> T:
    parser = argparse.ArgumentParser()
    for name, arg in inspect.signature(fn).parameters.items():
        origin = typing.get_origin(arg.annotation)
        if origin in (types.UnionType, Union):
            # Only `None | Type` & `Optional[Type]` are supported.
            none_index, type_index = (0, 1) if origin is types.UnionType else (1, 0)
            assert len(typing.get_args(arg.annotation)) == 2 and (
                typing.get_args(arg.annotation)[none_index] is types.NoneType
            )
            assert arg.default is None
            type_ = typing.get_args(arg.annotation)[type_index]
        else:
            assert origin is None
            type_ = arg.annotation
        assert type_ in (bool, int, float, str, Path) or issubclass(type_, enum.Enum)

        has_default = arg.default is not inspect.Parameter.empty
        if type_ is bool:
            if has_default and arg.default:
                parser.add_argument('--no-' + name, action='store_false', dest=name)
            else:
                parser.add_argument('--' + name, action='store_true')
        else:
            parser.add_argument(
                ('--' if has_default else '') + name,
                type=type_,
                **({'default': arg.default} if has_default else {}),
            )
    return fn(**vars(parser.parse_args()))



# ======================================================================================
# >>> plotting and visualization <<<
# ======================================================================================

# Think about the fast dataframe creation from a set of variables, ready to plot
def gg(**kwargs):
    "Fast ggplot from numpy arrays anything"
    return p9.ggplot(
        pd.DataFrame({k: v} for k, v in kwargs.items())
    )
    
