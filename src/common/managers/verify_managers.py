"""Utility to verify manager initialization status and fix any issues."""
import os
import logging
import traceback
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def verify_managers() -> Dict[str, bool]:
    """
    Verify that all critical managers are properly initialized.
    
    Returns:
        Dict[str, bool]: Status of each manager
    """
    from src.common.managers import (
        get_cuda_manager, get_directory_manager, get_wandb_manager,
        get_batch_manager, get_parameter_manager, get_metrics_manager,
        get_tokenizer_manager, get_data_manager, get_dataloader_manager
    )
    
    manager_getters = {
        "CUDAManager": get_cuda_manager,
        "DirectoryManager": get_directory_manager, 
        "TokenizerManager": get_tokenizer_manager,
        "DataLoaderManager": get_dataloader_manager,
        "DataManager": get_data_manager,
        "MetricsManager": get_metrics_manager,
        "BatchManager": get_batch_manager,
        "ParameterManager": get_parameter_manager,
        "WandbManager": get_wandb_manager
    }
    
    results = {}
    for name, getter in manager_getters.items():
        try:
            manager = getter()
            # First check if manager exists
            if manager is None:
                logger.error(f"{name} is None")
                results[name] = False
                continue
                
            # Check if manager has _local attribute
            if not hasattr(manager, '_local'):
                logger.error(f"{name} has no _local attribute")
                results[name] = False
                continue
                
            # Check if manager has pid and initialized attributes
            if not hasattr(manager._local, 'pid'):
                logger.error(f"{name} has no pid attribute")
                results[name] = False
                continue
                
            if not hasattr(manager._local, 'initialized'):
                logger.error(f"{name} has no initialized attribute")
                results[name] = False
                continue
                
            # Check if manager has correct pid
            if manager._local.pid != os.getpid():
                logger.error(f"{name} has wrong pid: {manager._local.pid} vs {os.getpid()}")
                results[name] = False
                continue
                
            # Check if manager is initialized
            if not manager._local.initialized:
                logger.error(f"{name} is not initialized")
                results[name] = False
                continue
                
            # Check the is_initialized method if available
            if hasattr(manager, 'is_initialized') and callable(manager.is_initialized):
                if not manager.is_initialized():
                    logger.error(f"{name}.is_initialized() returned False")
                    results[name] = False
                    continue
            
            # All checks passed
            results[name] = True
            
        except Exception as e:
            logger.error(f"Error verifying {name}: {e}")
            logger.error(traceback.format_exc())
            results[name] = False
    
    return results

def check_manager_health_and_repair(config: Dict[str, Any] = None) -> Dict[str, bool]:
    """
    Check the health of all managers and repair if needed.
    
    Args:
        config: Optional configuration to use for repairs
        
    Returns:
        Dict[str, bool]: Status of each manager after repairs
    """
    # First check manager health
    status = verify_managers()
    
    # Identify managers that need repair
    need_repair = [name for name, ok in status.items() if not ok]
    
    if need_repair:
        logger.warning(f"The following managers need repair: {need_repair}")
        
        # Import repair functionality
        from src.common.managers.manager_repair import repair_all_managers
        repair_results = repair_all_managers()
        
        # Log results
        for name, repaired in repair_results.items():
            if name in need_repair:
                if repaired:
                    logger.info(f"Successfully repaired {name}")
                else:
                    logger.error(f"Failed to repair {name}")
        
        # Verify again after repairs
        status = verify_managers()
    else:
        logger.info("All managers are healthy")
    
    return status
