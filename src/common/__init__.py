# src/common/__init__.py (COMPLETE and CORRECT)
from __future__ import annotations
"""Manager access layer to prevent circular imports."""
from pathlib import Path

_config = {} # Initialize _config


def get_amp_manager():
    from src.common.managers.amp_manager import AMPManager
    return AMPManager(_config)  # Pass config

def get_batch_manager():
    from src.common.managers.batch_manager import BatchManager
    return BatchManager()

def get_cuda_manager():
    from src.common.managers.cuda_manager import CUDAManager
    return CUDAManager(_config) # Pass config

def get_data_manager():
    from src.common.managers.data_manager import DataManager
    return DataManager(_config) # Pass config

def get_dataloader_manager():
    from src.common.managers.dataloader_manager import DataLoaderManager
    return DataLoaderManager(_config) # Pass config

def get_directory_manager():
    from src.common.managers.directory_manager import DirectoryManager
    return DirectoryManager(Path(_config['output']['dir']), _config)  # Pass base_dir and config

def get_metrics_manager():
    from src.common.managers.metrics_manager import MetricsManager
    return MetricsManager(_config) # Pass config

def get_model_manager():
    from src.common.managers.model_manager import ModelManager
    return ModelManager(_config) # Pass config

def get_parameter_manager():
    from src.common.managers.parameter_manager import ParameterManager
    return ParameterManager(_config)

def get_resource_manager():
    from src.common.managers.resource_manager import ProcessResourceManager
    return ProcessResourceManager(_config)

def get_storage_manager():
    from src.common.managers.storage_manager import StorageManager
    return StorageManager(Path(_config['output']['dir']) / 'storage', _config) # Pass storage_dir and config

def get_tensor_manager():
    from src.common.managers.tensor_manager import TensorManager
    return TensorManager(_config) # Pass config

def get_tokenizer_manager():
    from src.common.managers.tokenizer_manager import TokenizerManager
    return TokenizerManager(_config) # Pass config

def get_worker_manager():
    from src.common.managers.worker_manager import WorkerManager
    return WorkerManager(_config, "embedding_study", f"sqlite:///{Path(_config['output']['dir']) / 'storage' / 'optuna.db'}?timeout=60") # Pass config, study name, and storage_url

def get_wandb_manager():
    from src.common.managers.wandb_manager import WandbManager
    return WandbManager(_config, "embedding_study")

def get_optuna_manager():
    from src.common.managers.optuna_manager import OptunaManager
    return OptunaManager("embedding_study", _config, Path(_config['output']['dir']) / 'storage') # Pass study_name, config and storage_dir

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
    'get_optuna_manager'
]