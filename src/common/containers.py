"""Container for dependency injection."""
from dependency_injector import containers, providers
import logging
import os

from src.common.managers.cuda_manager import CUDAManager
from src.common.managers.data_manager import DataManager
from src.common.managers.model_manager import ModelManager
from src.common.managers.tokenizer_manager import TokenizerManager
from src.common.managers.directory_manager import DirectoryManager
from src.common.managers.parameter_manager import ParameterManager
from src.common.managers.wandb_manager import WandbManager
from src.common.managers.optuna_manager import OptunaManager
from src.common.managers.amp_manager import AMPManager
from src.common.managers.tensor_manager import TensorManager
from src.common.managers.batch_manager import BatchManager
from src.common.managers.metrics_manager import MetricsManager
from src.common.managers.dataloader_manager import DataLoaderManager
from src.common.managers.storage_manager import StorageManager
from src.common.managers.resource_manager import ProcessResourceManager
from src.common.managers.worker_manager import WorkerManager

logger = logging.getLogger(__name__)

class Container(containers.DeclarativeContainer):
    """Container for managers."""
    
    config = providers.Configuration()
    
    # Create providers for core managers with context-aware config
    cuda_manager = providers.Singleton(
        CUDAManager,
        config=providers.Callable(
            lambda c: c if isinstance(c, dict) and 'paths' in c else {'paths': {'base_dir': os.getcwd()}} | c,
            config
        )
    )
    
    directory_manager = providers.Singleton(
        DirectoryManager,
        config=providers.Callable(
            lambda c: c if isinstance(c, dict) and 'paths' in c else {'paths': {'base_dir': os.getcwd()}} | c,
            config
        )
    )
    
    tokenizer_manager = providers.Singleton(
        TokenizerManager,
        config=config
    )
    
    parameter_manager = providers.Singleton(
        ParameterManager,
        base_config=config,  # Fix: specify base_config parameter
        config=config        # Keep config parameter as well
    )
    
    # Managers with dependencies
    storage_manager = providers.Singleton(
        StorageManager,
        directory_manager=directory_manager,
        config=config
    )
    
    dataloader_manager = providers.Singleton(
        DataLoaderManager,
        cuda_manager=cuda_manager,
        config=config
    )
    
    model_manager = providers.Singleton(
        ModelManager,
        cuda_manager=cuda_manager,
        tokenizer_manager=tokenizer_manager,
        config=config
    )
    
    data_manager = providers.Singleton(
        DataManager,
        tokenizer_manager=tokenizer_manager,
        dataloader_manager=dataloader_manager,
        config=config
    )
    
    amp_manager = providers.Singleton(
        AMPManager,
        cuda_manager=cuda_manager,
        config=config
    )
    
    # Optional managers that may or may not be defined
    tensor_manager = providers.Singleton(
        TensorManager,
        cuda_manager=cuda_manager,
        config=config
    )
    
    batch_manager = providers.Singleton(
        BatchManager,
        cuda_manager=cuda_manager,
        tensor_manager=tensor_manager,  # Add the missing tensor_manager dependency
        config=config
    )
    
    metrics_manager = providers.Singleton(
        MetricsManager,
        cuda_manager=cuda_manager,  # Add the required cuda_manager dependency
        config=config
    )
    
    resource_manager = providers.Singleton(
        ProcessResourceManager,
        config=config
    )
    
    worker_manager = providers.Singleton(
        WorkerManager,
        config=config
    )
    
    # Special managers with unique dependencies
    wandb_manager = providers.Singleton(
        WandbManager,
        config=config,
        # Add the study_name parameter, extracting it from config
        study_name=providers.Callable(
            lambda c: c.get('training', {}).get('study_name', 'default_study'),
            config
        )
    )
    
    optuna_manager = providers.Singleton(
        OptunaManager,
        config=config,
        storage_manager=storage_manager,
        parameter_manager=parameter_manager
    )