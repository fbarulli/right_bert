from pathlib import Path
from typing import Dict, Any, Optional
import logging
import os
import traceback

# Import container
from src.common.containers import Container

logger = logging.getLogger(__name__)

# Container instance
_container = None

def initialize_factory(config: Dict[str, Any]) -> None:
    """Initialize the manager factory with the given configuration."""
    global _container
    
    try:
        # Register this process in the registry
        from src.common.process_registry import register_process
        register_process(process_type='manager_factory')
        
        # Create container instance
        _container = Container()
        
        # Set configuration
        _container.config.from_dict(config)
        
        # Initialize core managers explicitly with proper dependency management
        logger.info("Initializing managers in dependency order...")
        
        # First tier: Core managers with no dependencies
        cuda_manager = _container.cuda_manager()
        if hasattr(cuda_manager, 'setup'):
            cuda_manager.setup(config)
        else:
            cuda_manager._initialize_process_local(config)
        logger.debug("CUDAManager initialized")
        
        # IMPORTANT: Check that cuda_manager is properly initialized before proceeding
        if not cuda_manager.is_initialized():
            raise RuntimeError("CUDAManager failed to initialize properly")
            
        directory_manager = _container.directory_manager()
        if hasattr(directory_manager, 'setup'):
            directory_manager.setup(config)
        else:
            directory_manager._initialize_process_local(config)
        logger.debug("DirectoryManager initialized")
        
        # Second tier: Core managers that depend on first tier
        tokenizer_manager = _container.tokenizer_manager()
        if hasattr(tokenizer_manager, 'setup'):
            tokenizer_manager.setup(config)
        else:
            tokenizer_manager._initialize_process_local(config)
        logger.debug("TokenizerManager initialized")
        
        parameter_manager = _container.parameter_manager()
        if hasattr(parameter_manager, 'setup'):
            parameter_manager.setup(config)
        else:
            parameter_manager._initialize_process_local(config)
        logger.debug("ParameterManager initialized")
        
        # Third tier: Mid-level managers
        storage_manager = _container.storage_manager()
        if hasattr(storage_manager, 'setup'):
            storage_manager.setup(config)
        logger.debug("StorageManager initialized")
        
        # IMPORTANT: DataLoader depends on CUDA, so make sure CUDA is initialized first
        dataloader_manager = _container.dataloader_manager()
        # We already verified CUDA is initialized, so this should be safe
        if hasattr(dataloader_manager, 'setup'):
            dataloader_manager.setup(config)
        else:
            dataloader_manager._initialize_process_local(config)
        logger.debug("DataLoaderManager initialized")
        
        # Fourth tier: Managers with multiple dependencies
        # Model depends on tokenizer and CUDA
        model_manager = _container.model_manager()
        if hasattr(model_manager, 'setup'):
            model_manager.setup(config)
        else:
            model_manager._initialize_process_local(config)
        logger.debug("ModelManager initialized")
        
        # Data manager depends on tokenizer and DataLoader
        data_manager = _container.data_manager()
        if hasattr(data_manager, 'setup'):
            data_manager.setup(config)
        else:
            data_manager._initialize_process_local(config)
        logger.debug("DataManager initialized")
        
        # AMP depends on CUDA
        amp_manager = _container.amp_manager()
        if hasattr(amp_manager, 'setup'):
            amp_manager.setup(config)
        else:
            amp_manager._initialize_process_local(config)
        logger.debug("AMPManager initialized")
        
        # Fifth tier: Remaining managers
        tensor_manager = _container.tensor_manager()
        if hasattr(tensor_manager, 'setup'):
            tensor_manager.setup(config)
        else:
            tensor_manager._initialize_process_local(config)
        logger.debug("TensorManager initialized")
        
        logger.info("All managers initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize managers: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

def get_factory() -> Container:
    """Return the initialized manager container."""
    if _container is None:
        raise ValueError("Manager factory not initialized. Call initialize_factory first.")
    return _container

# Helper functions to access managers
def get_cuda_manager():
    return get_factory().cuda_manager()

def get_data_manager():
    return get_factory().data_manager()

def get_model_manager():
    return get_factory().model_manager()

def get_tokenizer_manager():
    return get_factory().tokenizer_manager()

def get_directory_manager():
    return get_factory().directory_manager()

def get_parameter_manager():
    return get_factory().parameter_manager()

def get_wandb_manager():
    return get_factory().wandb_manager()

def get_optuna_manager():
    try:
        return get_factory().optuna_manager()
    except Exception as e:
        logger.error(f"Error getting OptunaManager: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

def get_amp_manager():
    return get_factory().amp_manager()

def get_tensor_manager():
    return get_factory().tensor_manager()

def get_batch_manager():
    return get_factory().batch_manager()

def get_metrics_manager():
    return get_factory().metrics_manager()

def get_dataloader_manager():
    return get_factory().dataloader_manager()

def get_storage_manager():
    return get_factory().storage_manager()

def get_resource_manager():
    return get_factory().resource_manager()

def get_worker_manager():
    return get_factory().worker_manager()

def cleanup_managers() -> None:
    """Clean up all manager instances."""
    if _container is not None:
        try:
            # Nothing special needed - the container will handle cleanup of all singletons
            logger.info("Cleaning up all managers...")
        except Exception as e:
            logger.error(f"Error during manager cleanup: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise

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
    'cleanup_managers'
]
