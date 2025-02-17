# src/common/__init__.py (FINAL CORRECTED)
from __future__ import annotations
"""Manager access layer to prevent circular imports."""
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
    manager = AMPManager(_config)  # Pass the config
    return manager

def get_batch_manager():
    from src.common.managers.batch_manager import BatchManager
    manager = BatchManager(_config) # Pass config
    return manager

def get_cuda_manager():
    from src.common.managers.cuda_manager import CUDAManager
    manager = CUDAManager(_config) #Pass config
    return manager

def get_data_manager():
    from src.common.managers.data_manager import DataManager
    manager = DataManager(_config)
    return manager

def get_dataloader_manager():
    from src.common.managers.dataloader_manager import DataLoaderManager
    manager =  DataLoaderManager(_config)
    return manager

def get_directory_manager():
    from src.common.managers.directory_manager import DirectoryManager
    manager =  DirectoryManager(Path(_config['output']['dir']), _config) # Pass config
    return manager


def get_metrics_manager():
    from src.common.managers.metrics_manager import MetricsManager
    manager = MetricsManager(_config)
    return manager

def get_model_manager():
    from src.common.managers.model_manager import ModelManager
    manager = ModelManager(_config)
    return manager

def get_parameter_manager():
    from src.common.managers.parameter_manager import ParameterManager
    manager = ParameterManager(_config) # config in init
    return manager


def get_resource_manager():
    from src.common.managers.resource_manager import ProcessResourceManager
    manager = ProcessResourceManager(_config) # config in init
    return manager


def get_storage_manager():
    from src.common.managers.storage_manager import StorageManager
    manager = StorageManager(Path(_config['output']['storage_dir']))  # Pass config
    return manager

def get_tensor_manager():
    from src.common.managers.tensor_manager import TensorManager
    manager = TensorManager(_config)
    return manager

def get_tokenizer_manager():
    from src.common.managers.tokenizer_manager import TokenizerManager
    manager = TokenizerManager(_config) #Pass config
    return manager

def get_worker_manager():
    from src.common.managers.worker_manager import WorkerManager
    manager = WorkerManager(config=_config, study_name="embedding_study", storage_url = f"sqlite:///{Path(_config['output']['dir']) / 'storage' / 'optuna.db'}?timeout=60")
    return manager

def get_wandb_manager():
    from src.common.managers.wandb_manager import WandbManager
    manager =  WandbManager(_config, "embedding_study") # config in init
    return manager

def get_optuna_manager():
    from src.common.managers.optuna_manager import OptunaManager
    manager = OptunaManager(study_name="embedding_study", config=_config, storage_dir=Path(_config['output']['dir']) / 'storage')
    return manager

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