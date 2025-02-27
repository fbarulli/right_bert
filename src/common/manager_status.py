"""Utility for checking status of managers in the framework."""
import os
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

def check_manager_attributes(manager: Any, name: str, required_attrs: List[str]) -> Tuple[bool, List[str]]:
    """
    Check that a manager has the required attributes.
    
    Args:
        manager: The manager to check
        name: Name of the manager for logging
        required_attrs: List of attribute names that should exist
        
    Returns:
        Tuple containing:
        - bool: True if all required attributes exist
        - List[str]: List of missing attributes
    """
    missing_attrs = []
    
    # Check thread-local storage exists
    if not hasattr(manager, '_local'):
        logger.error(f"{name} has no _local attribute!")
        return False, ["_local"]
    
    # Check required attributes
    for attr in required_attrs:
        if not hasattr(manager._local, attr):
            missing_attrs.append(attr)
    
    if missing_attrs:
        logger.error(f"{name} missing required thread-local attributes: {missing_attrs}")
        return False, missing_attrs
    
    return True, []

def fix_manager(manager: Any, name: str, defaults: Dict[str, Any]) -> None:
    """
    Fix a manager by adding missing attributes with defaults.
    
    Args:
        manager: The manager to fix
        name: Name of the manager for logging
        defaults: Dictionary of attribute names and default values
    """
    # Create _local if missing
    if not hasattr(manager, '_local'):
        import threading
        manager._local = threading.local()
        manager._local.pid = os.getpid()
        manager._local.initialized = False
        logger.warning(f"Created missing _local storage for {name}")
    
    # Set missing attributes
    for attr, default in defaults.items():
        if not hasattr(manager._local, attr):
            setattr(manager._local, attr, default)
            logger.warning(f"Added missing attribute {attr}={default} to {name}")

def dump_manager_state(manager: Any, name: str) -> Dict[str, Any]:
    """
    Dump the state of a manager for debugging.
    
    Args:
        manager: The manager to dump
        name: Name of the manager
        
    Returns:
        Dict: The manager's state
    """
    state = {
        "name": name,
        "has_local": hasattr(manager, '_local')
    }
    
    if state["has_local"]:
        state["local_attrs"] = {}
        
        # Copy all attributes we can access safely
        for attr_name in dir(manager._local):
            # Skip private attributes
            if attr_name.startswith('_'):
                continue
                
            try:
                value = getattr(manager._local, attr_name)
                # Don't include methods or complex objects
                if not callable(value) and isinstance(value, (bool, int, float, str, type(None))):
                    state["local_attrs"][attr_name] = value
                else:
                    state["local_attrs"][attr_name] = f"{type(value).__name__} instance"
            except Exception as e:
                state["local_attrs"][attr_name] = f"<error accessing: {e}>"
    
    return state
