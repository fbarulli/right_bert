
#src/common/__init__.py
from __future__ import annotations
from pathlib import Path

_config = {}


def get_amp_manager():
    from src.common.managers.amp_manager import AMPManager
    return AMPManager(_config)

def get_batch_manager():
    from src.common.managers.batch_manager import BatchManager
    return BatchManager()

def get_data_manager():
    from src.common.managers.data_manager import DataManager
    return DataManager(_config)

def get_dataloader_manager():
    from src.common.managers.dataloader_manager import DataLoaderManager
    return DataLoaderManager(_config)

def get_directory_manager():
    from src.common.managers.directory_manager import DirectoryManager
    return DirectoryManager(Path(_config['output']['dir']), _config)

def get_metrics_manager():
    from src.common.managers.metrics_manager import MetricsManager
    return MetricsManager(_config)

def get_model_manager():
    from src.common.managers.model_manager import ModelManager
    return ModelManager(_config)

def get_parameter_manager():
    from src.common.managers.parameter_manager import ParameterManager
    return ParameterManager(_config)

def get_resource_manager():
    from src.common.managers.resource_manager import ProcessResourceManager
    return ProcessResourceManager(_config)

def get_storage_manager():
    from src.common.managers.storage_manager import StorageManager
    return StorageManager(Path(_config['output']['dir']) / 'storage', _config)

def get_tensor_manager():
    from src.common.utils import get_tensor_manager as tensor_util # Import from utils
    return tensor_util()

def get_tokenizer_manager():
    from src.common.managers.tokenizer_manager import TokenizerManager
    return TokenizerManager(_config)

def get_cuda_manager():
    from src.common.utils import get_cuda_manager as cuda_util # Import from utils
    return cuda_util()

def get_wandb_manager():
    from src.common.managers.wandb_manager import WandbManager
    return WandbManager(_config, "embedding_study")

def get_optuna_manager():
    from src.common.managers.optuna_manager import OptunaManager
    return OptunaManager("embedding_study", _config, Path(_config['output']['dir']) / 'storage')

def get_worker_manager():
    from src.common.managers import get_worker_manager as managers_get_worker_manager
    return managers_get_worker_manager()

def get_shared_tokenizer():
    from src.common.managers.tokenizer_manager import TokenizerManager
    tokenizer_manager = TokenizerManager()
    return tokenizer_manager.get_shared_tokenizer()

def set_shared_tokenizer(tokenizer):
    from src.common.managers.tokenizer_manager import TokenizerManager
    tokenizer_manager = TokenizerManager()
    tokenizer_manager.set_shared_tokenizer(tokenizer)


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
    'get_wandb_manager',
    'get_optuna_manager',
    'get_shared_tokenizer',
    'set_shared_tokenizer',
    'get_worker_manager' # Corrected: Added back to __all__ (if it was removed accidentally)
]
