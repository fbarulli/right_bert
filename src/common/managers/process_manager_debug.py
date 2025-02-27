"""Utility module for debugging process-aware managers."""
import os
import logging
import threading
import multiprocessing as mp
from typing import Dict, Any, Optional, Set
from tabulate import tabulate

logger = logging.getLogger(__name__)

# Global registry of managers per process
_process_managers = {}  # pid -> {manager_name -> manager_instance}
_registry_lock = threading.RLock()

def register_manager(name: str, manager_instance: Any) -> None:
    """Register a manager instance with the debug system.
    
    Args:
        name: Manager name
        manager_instance: Manager instance
    """
    pid = os.getpid()
    with _registry_lock:
        if pid not in _process_managers:
            _process_managers[pid] = {}
        _process_managers[pid][name] = manager_instance
        logger.debug(f"Registered manager '{name}' for process {pid}")

def unregister_manager(name: str) -> None:
    """Unregister a manager instance.
    
    Args:
        name: Manager name
    """
    pid = os.getpid()
    with _registry_lock:
        if pid in _process_managers and name in _process_managers[pid]:
            del _process_managers[pid][name]
            logger.debug(f"Unregistered manager '{name}' for process {pid}")

def get_process_managers() -> Dict[int, Dict[str, Any]]:
    """Get all registered managers per process.
    
    Returns:
        Dict mapping process IDs to dictionaries of manager instances
    """
    with _registry_lock:
        return {pid: managers.copy() for pid, managers in _process_managers.items()}

def process_debugger() -> None:
    """Log the current state of all managers across processes."""
    with _registry_lock:
        process_count = len(_process_managers)
        if process_count == 0:
            logger.info("No processes registered in process debugger")
            return
            
        table_data = []
        for pid, managers in _process_managers.items():
            for manager_name, manager in managers.items():
                initialized = "Yes" if hasattr(manager, "is_initialized") and manager.is_initialized() else "No"
                table_data.append([pid, manager_name, initialized])
        
        headers = ["Process ID", "Manager Name", "Initialized"]
        table = tabulate(table_data, headers=headers, tablefmt="grid")
        
        logger.info(f"Process Manager Registry: {process_count} processes\n{table}")

def cleanup_process_managers() -> None:
    """Clean up all managers for the current process in the correct order."""
    pid = os.getpid()
    with _registry_lock:
        if pid not in _process_managers:
            logger.debug(f"No managers registered for process {pid}")
            return
            
        # Get managers for this process
        managers = _process_managers[pid]
        
        # Define cleanup ordering based on dependencies
        manager_order = [
            # High-level managers first
            "WandbManager", "MetricsManager", "BatchManager", "DataManager",
            # Mid-level managers
            "ModelManager", "DataLoaderManager", "StorageManager",
            # Core managers last
            "TokenizerManager", "DirectoryManager", "CUDAManager", 
            "AMPManager", "TensorManager", "ParameterManager"
        ]
        
        # Clean up in order
        for name in manager_order:
            if name in managers:
                try:
                    logger.debug(f"Cleaning up manager '{name}' for process {pid}")
                    manager = managers[name]
                    if hasattr(manager, "cleanup"):
                        manager.cleanup()
                    del managers[name]
                except Exception as e:
                    logger.error(f"Error cleaning up manager '{name}': {str(e)}")
        
        # Clean up any remaining managers
        remaining = list(managers.keys())
        for name in remaining:
            try:
                logger.debug(f"Cleaning up remaining manager '{name}' for process {pid}")
                manager = managers[name]
                if hasattr(manager, "cleanup"):
                    manager.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up manager '{name}': {str(e)}")
                
        # Clear the registry for this process
        del _process_managers[pid]
        logger.info(f"Cleaned up all managers for process {pid}")
