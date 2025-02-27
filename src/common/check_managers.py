"""Utility module for checking manager health."""
import os
import logging
from typing import Dict, Any, List, Set

logger = logging.getLogger(__name__)

def check_manager_health(manager: Any, name: str) -> bool:
    """
    Check if a manager is healthy and properly initialized.
    
    Args:
        manager: The manager to check
        name: Manager name for logging
        
    Returns:
        bool: Whether the manager is healthy
    """
    try:
        # Basic checks
        if not hasattr(manager, '_local'):
            logger.error(f"{name} has no _local attribute in process {os.getpid()}")
            return False
            
        if not hasattr(manager._local, 'initialized'):
            logger.error(f"{name} has no initialized flag in process {os.getpid()}")
            return False
            
        if not manager._local.initialized:
            logger.error(f"{name} is not initialized in process {os.getpid()}")
            return False
        
        # Check pid is correct    
        if not hasattr(manager._local, 'pid') or manager._local.pid != os.getpid():
            logger.error(f"{name} has incorrect pid ({getattr(manager._local, 'pid', 'None')}) in process {os.getpid()}")
            return False
            
        # If manager has a is_initialized method, use it too
        if hasattr(manager, 'is_initialized') and callable(manager.is_initialized):
            is_init = manager.is_initialized()
            if not is_init:
                logger.error(f"{name}.is_initialized() returned False in process {os.getpid()}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking {name} health: {str(e)}")
        return False

def check_all_managers() -> Dict[str, bool]:
    """
    Check health of all managers.
    
    Returns:
        Dict[str, bool]: Manager health status
    """
    from src.common.managers import (
        get_cuda_manager, get_data_manager, get_model_manager, get_tokenizer_manager,
        get_directory_manager, get_parameter_manager, get_wandb_manager, get_amp_manager,
        get_tensor_manager, get_batch_manager
    )
    
    managers = {
        "CUDAManager": get_cuda_manager,
        "DirectoryManager": get_directory_manager,
        "TokenizerManager": get_tokenizer_manager,
        "ParameterManager": get_parameter_manager,
        "WandbManager": get_wandb_manager,
        "AMPManager": get_amp_manager,
        "TensorManager": get_tensor_manager,
        "BatchManager": get_batch_manager,
        "DataManager": get_data_manager,
        "ModelManager": get_model_manager
    }
    
    results = {}
    for name, getter in managers.items():
        try:
            manager = getter()
            results[name] = check_manager_health(manager, name)
        except Exception as e:
            logger.error(f"Error getting {name}: {str(e)}")
            results[name] = False
            
    return results
