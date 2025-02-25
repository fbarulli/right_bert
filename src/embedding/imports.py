"""
Common imports for the embedding module.
This file centralizes all external dependencies and internal utilities.
"""
from __future__ import annotations

# Standard library
import logging
import os
import random
import gc
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    TypeVar,
    cast,
)
from dataclasses import dataclass

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Transformers
from transformers import (
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup,
)

# Progress bars
from tqdm.notebook import tqdm

# Optuna (optional)
try:
    import optuna
except ImportError:
    optuna = None

# Internal imports - Logging utilities
from src.embedding.logging_utils import logger  # Moved logger import here

# Internal imports - Common utilities
from src.common.logging_utils import (
    log_function,
    LogConfig,
    create_progress_bar,
)

# Internal imports - Memory management
from src.embedding.memory_utils import (
    GPUMemoryManager,
    TensorPool,
    MemoryTracker,
    CachingDict,
)

# Internal imports - Managers
from src.common.managers.cuda_manager import CUDAManager
from src.common.managers.batch_manager import BatchManager
from src.common.managers.amp_manager import AMPManager
from src.common.managers.tokenizer_manager import TokenizerManager
from src.common.managers.metrics_manager import MetricsManager
from src.common.managers.storage_manager import StorageManager
from src.common.managers.wandb_manager import WandbManager

# Internal imports - Base classes
from src.data.csv_dataset import CSVDataset
from src.training.base_trainer import BaseTrainer

__all__ = [
    # Standard library
    'logging',
    'os',
    'random',
    'gc',
    'Path',
    'Dict', 'List', 'Optional', 'Set', 'Tuple', 'Union', 'TypeVar', 'Any', 'cast',
    'dataclass',
    
    # PyTorch
    'torch',
    'nn',
    'F',
    'Tensor',
    'Optimizer',
    'Dataset',
    'DataLoader',
    'autocast',
    'GradScaler',
    
    # Transformers
    'BertConfig',
    'BertModel',
    'BertPreTrainedModel',
    'PreTrainedModel',
    'PreTrainedTokenizerFast',
    'get_linear_schedule_with_warmup',
    
    # Progress bars
    'tqdm',
    
    # Optuna
    'optuna',
    
    # Common utilities
    'log_function',
    'LogConfig',
    'create_progress_bar',
    
    # Memory management
    'GPUMemoryManager',
    'TensorPool',
    'MemoryTracker',
    'CachingDict',
    
    # Managers
    'CUDAManager',
    'BatchManager',
    'AMPManager',
    'TokenizerManager',
    'MetricsManager',
    'StorageManager',
    'WandbManager',
    
    # Base classes
    'CSVDataset',
    'BaseTrainer',
    
    # Logger
    'logger',
]