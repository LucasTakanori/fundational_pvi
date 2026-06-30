"""
Just a convenient place to declare all the imports
"""

# Built-in packages
import sys
import os
import platform
import gc
import time
import copy
import random
import datetime
import itertools
from pathlib import Path
from math import prod

# Data handling
import h5py
import csv
import yaml
import json

# OOP
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# ML utilities
import numpy as np
import sklearn
import pandas as pd
import ot
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

# Optional (only imported when needed, pasted here but commented for convenience)

# import torch.nn.functional as F
# import numbers
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# from scipy.stats import pearsonr
# import pandas as pd

# Explicit exports - controls what gets imported with "from master_import import *"
__all__ = [
    # built-in
    'sys', 'os', 'platform', 'gc', 'time', 'copy', 'random', 'datetime', 'itertools', 'Path', 'prod',

    # data handling
    'h5py', 'csv', 'yaml', 'json',

    # OOP
    'ABC', 'abstractmethod', 'dataclass', 'Enum',

    # ML
    'np', 'sklearn', 'pd', 'torch', 'nn', 'optim',
    'Dataset', 'DataLoader', 'Subset',

    # Optional
    # 'numbers', 'F', 'train_test_split', 'tqdm', 'pearsonr', 'pd',
]

def get_platform_info() -> dict:
    info = {'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': os.cpu_count()}

    return info

def get_python_info() -> dict:
    implementation = platform.python_implementation()

    if 'Anaconda' in sys.version or 'Continuum' in sys.version:
        distribution = 'Anaconda'
    elif 'conda' in sys.executable.lower():
        distribution = 'Miniconda/Conda/Conda-forge'
    elif 'pypy' in implementation.lower():
        distribution = 'PyPy'
    elif 'intel' in sys.version.lower():
        distribution = 'Intel Python'
    elif implementation == 'CPython':
        distribution = 'CPython (Standard)'
    else:
        distribution = 'Unknown'

    info = {'version': platform.python_version(),
            'implementation': platform.python_implementation(),
            'distribution': distribution}

    return info

def get_torch_info() -> dict:
    info = {'torch_version': str(torch.__version__),
            'cuda_available': torch.cuda.is_available(),
            }
    if torch.cuda.is_available():
        info.update({
            'cuda_available': True,
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'device_count': torch.cuda.device_count(),
            'device_name': torch.cuda.get_device_name(0),
        })

    return info

def get_versions(modules: list[str]=None) -> dict:
    if modules is None:
        modules = ['h5py',
                   'numpy',
                   'scipy',
                   'ot',
                   'sklearn',
                   'pandas',
                   'torch',
                   'json']

    versions = {}
    for kw in modules:
        # only print what is loaded
        if kw in sys.modules.keys() and hasattr(sys.modules[kw], '__version__'):
            versions[sys.modules[kw].__name__] = sys.modules[kw].__version__

    return versions

def get_environment() -> dict[str,dict]:

    dict_out = {'platform': get_platform_info(),
                'python': get_python_info(),
                'torch': get_torch_info(),
                'packages': get_versions(),
                }

    return dict_out
if __name__ == '__main__':
    get_environment()