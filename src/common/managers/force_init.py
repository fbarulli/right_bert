"""Utilities for forcing manager initialization in emergency situations."""
import os
import logging
import threading
import tempfile
from pathlib import Path
from typing import Dict, Any

from src.common.managers.directory_manager import DirectoryManager

logger = logging.getLogger(__name__)

def force_directory_manager_init(config: Dict[str, Any] = None):
    """
    Force initialize the directory manager when normal initialization fails.
    
    Args:
        config: Configuration dictionary or None to use defaults
        
    Returns:
        The force-initialized directory manager
    """
    # Import here to avoid circular imports
    from src.common.managers import get_directory_manager
    
    logger.critical("FORCE INITIALIZING DIRECTORY MANAGER")
    directory_manager = get_directory_manager()
    
    # Create fresh thread-local storage
    directory_manager._local = threading.local()
    
    # Set basic attributes
    directory_manager._local.pid = os.getpid()
    directory_manager._local.initialized = True
    
    # Set directory attributes with safe defaults
    base_dir = os.getcwd()
    directory_manager._local.base_dir = base_dir
    directory_manager._local.output_dir = os.path.join(base_dir, 'output')
    directory_manager._local.data_dir = os.path.join(base_dir, 'data')
    directory_manager._local.cache_dir = os.path.join(base_dir, 'cache')
    directory_manager._local.model_dir = os.path.join(base_dir, 'model')
    directory_manager._local.temp_dir = tempfile.mkdtemp()
    
    # Create directories
    for attr_name in ['output_dir', 'data_dir', 'cache_dir', 'model_dir']:
        dir_path = getattr(directory_manager._local, attr_name)
        os.makedirs(dir_path, exist_ok=True)
        logger.critical(f"Force-created directory: {dir_path}")
    
    return directory_manager

def force_all_managers_init(config: Dict[str, Any] = None):
    """
    Force initialize all critical managers.
    
    Args:
        config: Configuration dictionary or None to use defaults
    """
    from src.common.managers import (
        get_cuda_manager, get_directory_manager, get_parameter_manager, 
        get_tokenizer_manager, get_model_manager, get_amp_manager
    )
    
    logger.critical("FORCE INITIALIZING ALL MANAGERS")
    
    # Start with most critical managers
    force_directory_manager_init(config)
    
    # Initialize other managers
    managers = [
        get_cuda_manager(),
        get_parameter_manager(),
        get_tokenizer_manager(),
        get_model_manager(),
        get_amp_manager()
    ]
    
    # Set initialized flags for all managers
    for manager in managers:
        if not hasattr(manager, '_local'):
            manager._local = threading.local()
        manager._local.pid = os.getpid()  
        manager._local.initialized = True
        
        logger.critical(f"Force-initialized {manager.__class__.__name__}")

__all__ = ['force_directory_manager_init', 'force_all_managers_init']
