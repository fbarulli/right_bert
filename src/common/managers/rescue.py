"""Rescue utilities for recovering from manager initialization failures."""
import os
import tempfile
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def rescue_directory_manager(manager, config: Dict[str, Any]) -> bool:
    """
    Force-repair a DirectoryManager that failed to initialize.
    
    Args:
        manager: The DirectoryManager instance
        config: Configuration dictionary
        
    Returns:
        bool: True if rescue was successful
    """
    logger.critical(f"Attempting emergency rescue of DirectoryManager in process {os.getpid()}")
    
    try:
        # Get paths from config or use defaults
        paths_config = config.get('paths', {})
        base_dir = paths_config.get('base_dir', os.getcwd())
        output_dir = os.path.join(base_dir, paths_config.get('output_dir', 'output'))
        data_dir = os.path.join(base_dir, paths_config.get('data_dir', 'data'))
        cache_dir = os.path.join(base_dir, paths_config.get('cache_dir', '.cache'))
        model_dir = os.path.join(base_dir, paths_config.get('model_dir', 'models'))
        temp_dir = tempfile.mkdtemp()
        
        # Reset the thread-local object
        manager._local = threading.local()
        manager._local.pid = os.getpid()
        manager._local.initialized = True
        
        # Set directory paths
        manager._local.base_dir = Path(base_dir)
        manager._local.output_dir = Path(output_dir) 
        manager._local.data_dir = Path(data_dir)
        manager._local.cache_dir = Path(cache_dir)
        manager._local.model_dir = Path(model_dir)
        manager._local.temp_dir = Path(temp_dir)
        
        # Create all directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        logger.warning("DirectoryManager emergency rescue completed")
        logger.warning(f"DirectoryManager state after rescue: {manager._local.__dict__}")
        
        return True
    except Exception as e:
        logger.error(f"DirectoryManager rescue failed: {str(e)}")
        return False

def rescue_any_manager(manager, name: str) -> bool:
    """
    Generic rescue for any manager by setting the initialized flag.
    
    Args:
        manager: Any manager instance
        name: Manager name for logging
        
    Returns:
        bool: True if rescue was attempted
    """
    try:
        if not hasattr(manager, '_local'):
            manager._local = threading.local()
            
        manager._local.pid = os.getpid()
        manager._local.initialized = True
        
        logger.warning(f"Emergency rescue attempted for {name}")
        return True
    except Exception as e:
        logger.error(f"Failed to rescue {name}: {str(e)}")
        return False
