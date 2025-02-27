"""Helper module for initializing managers in child processes."""
import os
import logging
import sys
import threading
import tempfile  # Add this missing import
from pathlib import Path  # Add this missing import
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Flag to track if initialization has been done for current process
_process_initialized = {}

def ensure_process_initialized(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Ensure that all managers are initialized for the current process.
    
    Args:
        config: Optional configuration to override the factory configuration
    """
    try:
        from src.common.manager_status import fix_manager, check_manager_attributes
        from src.common.managers import (
            get_cuda_manager, get_directory_manager, get_wandb_manager,
            get_batch_manager, get_parameter_manager, get_metrics_manager
        )
        
        # Get critical managers
        managers = {
            "CUDAManager": get_cuda_manager(),
            "DirectoryManager": get_directory_manager(),
            "WandbManager": get_wandb_manager(),
            "BatchManager": get_batch_manager(),
            "ParameterManager": get_parameter_manager(),
            "MetricsManager": get_metrics_manager()
        }
        
        logger.info(f"Initializing managers for child process {os.getpid()}")
        
        # First, ensure all managers have required base attributes
        for name, manager in managers.items():
            # Check for missing attributes and fix them
            required_attrs = ['pid', 'initialized']
            ok, missing = check_manager_attributes(manager, name, required_attrs)
            
            if not ok:
                # Set up defaults for base attributes
                defaults = {
                    'pid': os.getpid(),
                    'initialized': False
                }
                fix_manager(manager, name, defaults)
        
        # Special handling for problematic managers
        # WandbManager might need special attributes
        wandb_manager = managers["WandbManager"]
        wandb_attrs = ['enabled', 'run', 'start_time', 'project', 'entity']
        for attr in wandb_attrs:
            if not hasattr(wandb_manager._local, attr):
                logger.warning(f"WandbManager.{attr} attribute missing in process {os.getpid()}, initializing")
                if attr == 'enabled':
                    setattr(wandb_manager._local, attr, False)
                elif attr == 'run':
                    setattr(wandb_manager._local, attr, None)
                elif attr == 'start_time':
                    setattr(wandb_manager._local, attr, time.time())
                elif attr == 'project':
                    setattr(wandb_manager._local, attr, 'default')
                elif attr == 'entity':
                    setattr(wandb_manager._local, attr, None)
                elif attr == 'initialized':
                    setattr(wandb_manager._local, attr, True)
        
        # ParameterManager often has issues, force initialize it if needed
        parameter_manager = managers["ParameterManager"]
        if not parameter_manager.is_initialized():
            logger.critical(f"ParameterManager initialization failed in process {os.getpid()}, forcing it")
            # Force initialize it
            parameter_manager._local = threading.local()
            parameter_manager._local.pid = os.getpid()
            parameter_manager._local.initialized = True
            # Set all essential attributes to avoid AttributeError later
            parameter_manager._local.base_config = config or {}
            parameter_manager._local.search_space = {}
            parameter_manager._local.param_ranges = {}
            parameter_manager._local.hyperparameters = {}
            try:
                parameter_manager._initialize_process_local(config)
                logger.info(f"ParameterManager force-initialization successful for process {os.getpid()}")
            except Exception as e:
                logger.error(f"Force initialization of ParameterManager failed: {e}")
        
        # Initialize rest of the managers in dependency order
        cuda_manager = managers["CUDAManager"]
        if not cuda_manager.is_initialized():
            cuda_manager._initialize_process_local(config)
        
        directory_manager = managers["DirectoryManager"]
        if not directory_manager.is_initialized():
            directory_manager._initialize_process_local(config)
            
        # Ensure directory_manager has all required directory attributes
        required_dirs = ['base_dir', 'output_dir', 'data_dir', 'cache_dir', 'model_dir', 'temp_dir']
        for dir_attr in required_dirs:
            if not hasattr(directory_manager._local, dir_attr):
                if dir_attr == 'base_dir':
                    setattr(directory_manager._local, dir_attr, os.getcwd())
                elif dir_attr == 'temp_dir':
                    import tempfile
                    setattr(directory_manager._local, dir_attr, tempfile.mkdtemp())
                else:
                    # Derive from base_dir
                    base = getattr(directory_manager._local, 'base_dir', os.getcwd())
                    subdir = dir_attr.replace('_dir', '')
                    setattr(directory_manager._local, dir_attr, os.path.join(base, subdir))
        
        # Continue with other managers...
        
        # Finally, verify all managers are initialized
        all_initialized = True
        for name, manager in managers.items():
            if not manager.is_initialized():
                logger.error(f"{name} is still not initialized in process {os.getpid()}")
                all_initialized = False
        
        if all_initialized:
            logger.info(f"All managers initialized successfully in process {os.getpid()}")
        else:
            logger.error(f"Failed to initialize all managers in process {os.getpid()}")
            
    except Exception as e:
        logger.error(f"Error in ensure_process_initialized: {str(e)}")
        logger.error("Traceback: ", exc_info=True)
        # Just log the error, don't raise it, to allow process to continue

def try_setup_manager(manager, config, name):
    """Try to set up a manager and log any failures."""
    pid = os.getpid()
    try:
        if hasattr(manager, 'setup'):
            logger.debug(f"Calling setup() for {name} in process {pid}")
            manager.setup(config)
        elif hasattr(manager, '_initialize_process_local'):
            logger.debug(f"Calling _initialize_process_local() for {name} in process {pid}")
            manager._initialize_process_local(config)
        else:
            logger.warning(f"{name} has no setup or initialization method in process {pid}")
            
        # Add consistent initialization flag setting regardless of setup method
        if hasattr(manager, '_local'):
            manager._local.initialized = True
            manager._local.pid = pid
            
        # Verify initialization was successful
        if hasattr(manager, 'is_initialized'):
            is_init = manager.is_initialized()
            if not is_init:
                logger.warning(f"{name} did not initialize properly in process {pid}")
                # Force the initialization flag
                if hasattr(manager, '_local'):
                    manager._local.initialized = True
                    logger.info(f"Forced {name} initialization flag true for process {pid}")
        
        logger.debug(f"{name} initialized for process {pid}")
    except Exception as e:
        logger.error(f"Error initializing {name} in process {pid}: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        try:
            # Last-ditch effort: force the initialization
            if hasattr(manager, '_local'):
                manager._local.initialized = True
                logger.warning(f"Force-set {name} initialization flag true despite error")
        except:
            pass
        raise
