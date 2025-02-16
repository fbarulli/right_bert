# src/common/__init__.py
"""Manager access layer to prevent circular imports."""

from __future__ import annotations
from pathlib import Path
import yaml

# --- Load Configuration (Once, at import time) ---
try:
    with open("config/embedding_config.yaml", "r") as f:  # Correct path
        _config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError("config/embedding_config.yaml not found.  Please ensure the config file exists.")
except yaml.YAMLError as e:
    raise ValueError(f"Error parsing YAML configuration: {e}")

# --- Shared Tokenizer (Placeholder) ---
_shared_tokenizer = None
_tokenizer_lock = __import__('multiprocessing').Lock()


def get_amp_manager():
    from src.common.managers.amp_manager import AMPManager
    return AMPManager()

def get_batch_manager():
    from src.common.managers.batch_manager import BatchManager
    return BatchManager(get_cuda_manager(), get_tensor_manager())

def get_cuda_manager():
    from src.common.managers.cuda_manager import CUDAManager
    return CUDAManager()

def get_data_manager():
    from src.common.managers.data_manager import DataManager
    return DataManager()

def get_dataloader_manager():
    from src.common.managers.dataloader_manager import DataLoaderManager
    return DataLoaderManager()

def get_directory_manager():
    from src.common.managers.directory_manager import DirectoryManager
    return DirectoryManager(Path(_config['output']['dir']))

def get_metrics_manager():
    from src.common.managers.metrics_manager import MetricsManager
    return MetricsManager()

def get_model_manager():
    from src.common.managers.model_manager import ModelManager
    return ModelManager()

def get_parameter_manager():
    from src.common.managers.parameter_manager import ParameterManager
    return ParameterManager(_config)

def get_resource_manager():
    from src.common.managers.resource_manager import ProcessResourceManager
    return ProcessResourceManager(_config)

def get_storage_manager():
    from src.common.managers.storage_manager import StorageManager
    return StorageManager(Path(_config['output']['storage_dir']))

def get_tensor_manager():
    from src.common.managers.tensor_manager import TensorManager
    return TensorManager(get_cuda_manager())

def get_tokenizer_manager():
    from src.common.managers.tokenizer_manager import TokenizerManager
    return TokenizerManager()

#Corrected config passing
def get_worker_manager():
    from src.common.managers.worker_manager import WorkerManager
    return WorkerManager(n_jobs= _config['training']['n_jobs'], config=_config, study_name="embedding_study", storage_url = f"sqlite:///{Path(_config['output']['dir']) / 'storage' / 'optuna.db'}?timeout=60")

#Corrected config passing
def get_wandb_manager():
    from src.common.managers.wandb_manager import WandbManager
    return WandbManager(_config, "embedding_study")

#Corrected config passing
def get_optuna_manager():
    from src.common.managers.optuna_manager import OptunaManager
    return OptunaManager(study_name="embedding_study", config=_config, storage_dir=Path(_config['output']['dir']) / 'storage')

def set_shared_tokenizer(tokenizer):
    """Set the shared tokenizer instance."""
    global _shared_tokenizer
    with _tokenizer_lock:
        _shared_tokenizer = tokenizer

def get_shared_tokenizer():
    """Get the shared tokenizer instance."""
    with _tokenizer_lock:
        return _shared_tokenizer

__all__ = [
    'get_amp_manager',
    'get_batch_manager',
    'get_cuda_manager',
    'get_data_manager',
    'get_dataloader_manager',
    'get_directory_manager',
    'get_metrics_manager',
    'get_model_manager',
    'get_parameter_manager',
    'get_resource_manager',
    'get_storage_manager',
    'get_tensor_manager',
    'get_tokenizer_manager',
    'get_worker_manager',
    'get_wandb_manager',
    'get_optuna_manager',
    'set_shared_tokenizer',
    'get_shared_tokenizer'
]