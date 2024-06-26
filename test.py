import pytest

import time
import os
import sys
import shutil
import subprocess

import numpy as np
import lib

from loguru import logger
from pathlib import Path


def test_datasets_dtype():
    for d in lib.DATA_DIR.iterdir():
        if (d/'X_num.npy').exists():
            assert np.load(d/'X_num.npy').dtype == np.float32
        if (d/'X_bin.npy').exists():
            assert np.load(d/'X_bin.npy').dtype == np.float32
        if (d/'X_cat.npy').exists():
            assert np.load(d/'X_cat.npy').dtype == np.int64

        info = lib.load_json(d/'info.json')
        task_type = lib.TaskType(info['task_type'])

        if task_type in (lib.TaskType.BINCLASS, lib.TaskType.MULTICLASS):
            assert np.load(d/'Y.npy').dtype == np.int64
        else:
            assert np.load(d/'Y.npy').dtype == np.float32


def test_all_runs_start_successfull(tmp_path: Path):
    print()
    for tuning_config in lib.EXP_DIR.glob('**/tuning.toml'):
        if tuning_config.parent.name in [
            'cooking-time',
            'delivery-eta',
            'homesite-insurance',
            'maps-routing',
            'weather',
        ]:
            continue

        # if tuning_config.parent.parent.name in [
        #     "xgboost_",
        #     "catboost_",
        #     "lightgbm_",
        #     "mlp",
        #     "mlp-plr",
        #     "resnet",
        #     "snn",
        #     "dcn2",
        #     "ft_transformer",
        # ]:
        #     continue

        # All algorithms are using cuda devices
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        shutil.copy(tuning_config, tmp_path/tuning_config.name)
        
        process = subprocess.Popen(f"python bin/go.py {str(tmp_path)}/tuning.toml --force".split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if tuning_config.parent.parent.name in ['xgboost_', 'catboost_', 'lightgbm_']:
            # All boostings start training with this
            wait_on = 'training...'
        else:
            # This appears in nn logs
            wait_on = 'new best epoch!'

        stop = False
        while not stop:
            
            if process.stdout is not None:
                s = process.stdout.readline().decode().lower()
                stop = wait_on.lower() in s
            
            if stop:
                logger.info(f'{tuning_config.relative_to(lib.EXP_DIR)} OK, killing')
                process.kill()

            process.poll()
            assert process.returncode is None or process.returncode == 0, f"{tuning_config.name} fails {process.stderr.read()}"
            
