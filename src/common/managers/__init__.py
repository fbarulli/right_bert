from pathlib import Path
from typing import Dict, Any, Optional
from dependency_injector import containers, providers
import logging  # Ensure this is present

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

logger = logging.getLogger(__name__)  # Define logger

class ManagerContainer(containers.DeclarativeContainer):
    """Dependency injection container for managers."""
    config = providers.Configuration()

    cuda_manager = providers.Singleton(CUDAManager, config=config)
    directory_manager = providers.Singleton(
        DirectoryManager,
        base_dir=config.output.dir,  # Use dot notation for Configuration
        config=config
    )
    data_manager = providers.Singleton(DataManager, config=config)
    model_manager = providers.Singleton(ModelManager, config=config)
    tokenizer_manager = providers.Singleton(TokenizerManager, config=config)
    parameter_manager = providers.Singleton(ParameterManager, config=config)
    wandb_manager = providers.Singleton(WandbManager, config=config)
    tensor_manager = providers.Singleton(TensorManager, config=config)
    batch_manager = providers.Singleton(BatchManager, config=config)
    metrics_manager = providers.Singleton(MetricsManager, config=config)
    dataloader_manager = providers.Singleton(DataLoaderManager, config=config)
    resource_manager = providers.Singleton(ProcessResourceManager, config=config)
    amp_manager = providers.Singleton(
        AMPManager,
        cuda_manager=cuda_manager,
        config=config
    )
    storage_manager = providers.Singleton(
        StorageManager,
        directory_manager=directory_manager,
        config=config
    )
    optuna_manager = providers.Singleton(
        OptunaManager,
        storage_manager=storage_manager,
        study_name="embedding_study",
        config=config
    )
    worker_manager = providers.Singleton(
        WorkerManager,
        cuda_manager=cuda_manager,
        model_manager=model_manager,
        tokenizer_manager=tokenizer_manager,
        config=config,
        study_name="embedding_study",
        storage_url=lambda: f"sqlite:///{Path(config.output.dir) / 'storage' / 'optuna.db'}?timeout=60"  # Update this too
    )

_container = None




def initialize_factory(config: Dict[str, Any]) -> None:
    """
    Initialize the manager factory with the given configuration.

    Args:
        config: Configuration dictionary to initialize managers.
    """
    global _container
    _container = ManagerContainer(config=config)
    logger.info("Manager factory initialized with provided configuration")

def get_factory() -> ManagerContainer:
    """Return the initialized manager container."""
    if _container is None:
        raise ValueError("Manager factory not initialized. Call initialize_factory first.")
    return _container

# Helper functions to access managers
def get_cuda_manager() -> CUDAManager:
    return get_factory().cuda_manager()

def get_data_manager() -> DataManager:
    return get_factory().data_manager()

def get_model_manager() -> ModelManager:
    return get_factory().model_manager()

def get_tokenizer_manager() -> TokenizerManager:
    return get_factory().tokenizer_manager()

def get_directory_manager() -> DirectoryManager:
    return get_factory().directory_manager()

def get_parameter_manager() -> ParameterManager:
    return get_factory().parameter_manager()

def get_wandb_manager() -> WandbManager:
    return get_factory().wandb_manager()

def get_optuna_manager() -> OptunaManager:
    return get_factory().optuna_manager()

def get_amp_manager() -> AMPManager:
    return get_factory().amp_manager()

def get_tensor_manager() -> TensorManager:
    return get_factory().tensor_manager()

def get_batch_manager() -> BatchManager:
    return get_factory().batch_manager()

def get_metrics_manager() -> MetricsManager:
    return get_factory().metrics_manager()

def get_dataloader_manager() -> DataLoaderManager:
    return get_factory().dataloader_manager()

def get_storage_manager() -> StorageManager:
    return get_factory().storage_manager()

def get_resource_manager() -> ProcessResourceManager:
    return get_factory().resource_manager()

def get_worker_manager() -> WorkerManager:
    return get_factory().worker_manager()

def cleanup_managers(container: Optional[containers.DeclarativeContainer] = None):
    """
    Clean up all manager instances.

    Args:
        container: Optional dependency injection container. If provided,
                  will use it to get and cleanup managers in the correct order.
    """
    container = container or _container
    if container is not None:
        try:
            # Clean up in reverse dependency order
            if hasattr(container, 'resource_manager'):
                container.resource_manager().cleanup()
            if hasattr(container, 'worker_manager'):
                container.worker_manager().cleanup()
            if hasattr(container, 'data_manager'):
                container.data_manager().cleanup()
            if hasattr(container, 'model_manager'):
                container.model_manager().cleanup()
            if hasattr(container, 'metrics_manager'):
                container.metrics_manager().cleanup()  # dependent on cuda_manager
            if hasattr(container, 'batch_manager'):
                container.batch_manager().cleanup()

            # Secondary managers
            if hasattr(container, 'amp_manager'):
                container.amp_manager().cleanup()
            if hasattr(container, 'tensor_manager'):  # dependent on cuda_manager
                container.tensor_manager().cleanup()
            if hasattr(container, 'dataloader_manager'):
                container.dataloader_manager().cleanup()
            if hasattr(container, 'storage_manager'):
                container.storage_manager().cleanup()

            # Primary managers last
            if hasattr(container, 'tokenizer_manager'):
                container.tokenizer_manager().cleanup()
            if hasattr(container, 'directory_manager'):
                container.directory_manager().cleanup()
            if hasattr(container, 'cuda_manager'):
                container.cuda_manager().cleanup()

            logger.info("Successfully cleaned up all managers using DI container")
        except Exception as e:
            logger.error(f"Error during manager cleanup: {str(e)}")
            raise
    else:
        logger.warning("No DI container provided for cleanup - some resources may not be properly released")

__all__ = [
    'initialize_factory',
    'get_factory',
    'get_cuda_manager',
    'get_data_manager',
    'get_model_manager',
    'get_tokenizer_manager',
    'get_directory_manager',
    'get_parameter_manager',
    'get_wandb_manager',
    'get_optuna_manager',
    'get_amp_manager',
    'get_tensor_manager',
    'get_batch_manager',
    'get_metrics_manager',
    'get_dataloader_manager',
    'get_storage_manager',
    'get_resource_manager',
    'get_worker_manager',
    'cleanup_managers',
]
