"""Utilities for automatically initializing managers when needed."""
import os
import logging
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def auto_initialize_manager(manager: Any, name: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Automatically initialize a manager if it's not already initialized.
    
    Args:
        manager: The manager instance to initialize
        name: Name of the manager for logging
        config: Optional configuration
        
    Returns:
        bool: True if initialization was successful
    """
    try:
        # Check if manager needs initialization
        needs_init = False
        
        if not hasattr(manager, '_local'):
            logger.warning(f"{name} has no _local attribute, creating it")
            manager._local = threading.local()
            needs_init = True
            
        if not hasattr(manager._local, 'initialized') or not manager._local.initialized:
            logger.warning(f"{name} not initialized, initializing it now")
            needs_init = True
            
        if not hasattr(manager._local, 'pid') or manager._local.pid != os.getpid():
            logger.warning(f"{name} has wrong process ID, reinitializing")
            manager._local.pid = os.getpid()
            needs_init = True
            
        # Initialize if needed
        if needs_init:
            if hasattr(manager, 'setup') and callable(manager.setup):
                manager.setup(config)
            elif hasattr(manager, '_initialize_process_local') and callable(manager._initialize_process_local):
                manager._initialize_process_local(config)
            else:
                # Last resort: just set basic flags
                manager._local.initialized = True
                manager._local.pid = os.getpid()
                
            logger.info(f"Auto-initialized {name}")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to auto-initialize {name}: {e}")
        return False
        
def ensure_managers_initialized(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Ensure all critical managers are initialized.
    
    Args:
        config: Optional configuration dictionary
    """
    from src.common.managers import (
        get_cuda_manager, get_directory_manager, get_batch_manager,
        get_amp_manager, get_model_manager, get_tokenizer_manager
    )
    
    # Initialize critical managers in dependency order
    managers = {
        "CUDAManager": get_cuda_manager(),
        "DirectoryManager": get_directory_manager(),
        "TokenizerManager": get_tokenizer_manager(),
        "ModelManager": get_model_manager(),
        "AMPManager": get_amp_manager(),
        "BatchManager": get_batch_manager()
    }
    
    for name, manager in managers.items():
        auto_initialize_manager(manager, name, config)
