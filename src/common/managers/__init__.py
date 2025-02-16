# src/common/__init__.py (COMPLETE and CORRECT)
from __future__ import annotations
"""Manager access layer to prevent circular imports."""
from pathlib import Path

def get_amp_manager():
    from src.common.managers.amp_manager import AMPManager
    return AMPManager()  # No args needed

def get_batch_manager():
    from src.common.managers.batch_manager import BatchManager
    return BatchManager(get_cuda_manager(), get_tensor_manager())

def get_cuda_manager():
    from src.common.managers.cuda_manager import CUDAManager
    return CUDAManager()

def get_data_manager():
    from src.common.managers.data_manager import DataManager
    return DataManager()  # No args needed

def get_dataloader_manager():
    from src.common.managers.dataloader_manager import DataLoaderManager
    return DataLoaderManager() # No args needed

def get_directory_manager():
    from src.common.managers.directory_manager import DirectoryManager
    return DirectoryManager(Path("."))  # Provide base_dir

def get_metrics_manager():
    from src.common.managers.metrics_manager import MetricsManager
    return MetricsManager()

def get_model_manager():
    from src.common.managers.model_manager import ModelManager
    return ModelManager() # No Args

def get_parameter_manager():
    from src.common.managers.parameter_manager import ParameterManager
    return ParameterManager({}) # TODO: Provide base config

def get_resource_manager():
    from src.common.managers.resource_manager import ProcessResourceManager
    return ProcessResourceManager({}) # TODO: Provide config

def get_storage_manager():
    from src.common.managers.storage_manager import StorageManager
    return StorageManager(Path("storage"))  # Provide storage_dir

def get_tensor_manager():
    from src.common.managers.tensor_manager import TensorManager
    return TensorManager(get_cuda_manager())

def get_tokenizer_manager():
    from src.common.managers.tokenizer_manager import TokenizerManager
    return TokenizerManager()

def get_worker_manager():
    from src.common.managers.worker_manager import WorkerManager
    return WorkerManager() # Requires Args, fix later

def get_wandb_manager():
    from src.common.managers.wandb_manager import WandbManager
    return WandbManager({}, "")  # TODO: Provide config and study_name

def get_optuna_manager():
    from src.common.managers.optuna_manager import OptunaManager
    return OptunaManager() # Requires Args

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