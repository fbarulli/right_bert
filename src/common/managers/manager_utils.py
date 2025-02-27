"""Utility functions for manager initialization and maintenance."""
import os
import logging
import threading
from typing import Any, Dict, Type, Optional

logger = logging.getLogger(__name__)

def ensure_manager_initialized(
    manager: Any,
    required_attrs: Dict[str, Any],
    name: str
) -> bool:
    """
    Ensure a manager has all required thread-local attributes.
    
    Args:
        manager: Manager instance to check
        required_attrs: Dictionary of attribute names and default values
        name: Manager name for logging
        
    Returns:
        bool: True if initialization was successful
    """
    if not hasattr(manager, '_local'):
        manager._local = threading.local()
        manager._local.pid = os.getpid()
        logger.warning(f"Created new thread-local storage for {name} in process {os.getpid()}")
    
    local_attrs = vars(manager._local) if hasattr(manager._local, '__dict__') else {}
    
    for attr, default in required_attrs.items():
        if not hasattr(manager._local, attr):
            setattr(manager._local, attr, default)
            logger.debug(f"Set missing attribute {attr}={default} for {name}")
    
    # Set initialized flag if it doesn't exist
    if not hasattr(manager._local, 'initialized'):
        manager._local.initialized = True
        logger.debug(f"Set initialized=True for {name}")
        
    # Ensure pid is set correctly
    if not hasattr(manager._local, 'pid') or manager._local.pid != os.getpid():
        manager._local.pid = os.getpid()
        logger.debug(f"Updated pid to {os.getpid()} for {name}")
    
    return True

def check_manager_health(manager: Any, name: str) -> bool:
    """
    Check if a manager is properly initialized and healthy.
    
    Args:
        manager: Manager instance to check
        name: Manager name for logging
        
    Returns:
        bool: True if the manager is healthy
    """
    try:
        # Check if _local exists
        if not hasattr(manager, '_local'):
            logger.error(f"{name} has no _local attribute")
            return False
            
        # Check if initialized flag exists and is True
        if not hasattr(manager._local, 'initialized') or not manager._local.initialized:
            logger.error(f"{name} is not initialized")
            return False
            
        # Check if pid matches current process
        if not hasattr(manager._local, 'pid') or manager._local.pid != os.getpid():
            logger.error(f"{name} has wrong pid: {getattr(manager._local, 'pid', None)} vs {os.getpid()}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error checking {name} health: {e}")
        return False

__all__ = ['ensure_manager_initialized', 'check_manager_health']
