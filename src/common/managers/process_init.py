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
        import time  # Add import for time.time()
        from src.common.managers.verify_managers import check_manager_health_and_repair
        
        # Get managers - but enforce strict initialization order for dependencies
        logger.info(f"Initializing managers for child process {os.getpid()}")
        
        # First, get and initialize CUDA manager (top dependency)
        from src.common.managers import get_cuda_manager
        cuda_manager = get_cuda_manager()
        
        # Check if CUDA manager was initialized in this process
        if not cuda_manager.is_initialized():
            logger.info(f"Initializing CUDA manager for process {os.getpid()}")
            cuda_manager._initialize_process_local(config)
        
        # Check if initialization worked
        if not cuda_manager.is_initialized():
            logger.error(f"CUDA manager failed to initialize - using repair utility")
            from src.common.managers.manager_repair import repair_manager
            repair_manager(cuda_manager, "CUDAManager")
        else:
            logger.info(f"CUDA manager initialized for process {os.getpid()}")
            
        # Now get and initialize directory manager
        from src.common.managers import get_directory_manager
        directory_manager = get_directory_manager()
        if not directory_manager.is_initialized():
            logger.info(f"Initializing directory manager for process {os.getpid()}")
            directory_manager._initialize_process_local(config)
            
        # Now get and initialize tokenizer manager (needed by data manager)
        from src.common.managers import get_tokenizer_manager
        tokenizer_manager = get_tokenizer_manager()
        if not tokenizer_manager.is_initialized():
            logger.info(f"Initializing tokenizer manager for process {os.getpid()}")
            tokenizer_manager._initialize_process_local(config)
            
        # Now get and initialize dataloader manager (needed by data manager)
        from src.common.managers import get_dataloader_manager
        dataloader_manager = get_dataloader_manager()
        if not dataloader_manager.is_initialized():
            logger.info(f"Initializing dataloader manager for process {os.getpid()}")
            dataloader_manager._initialize_process_local(config)
            
        # Now get and initialize data manager
        from src.common.managers import get_data_manager
        data_manager = get_data_manager()
        if not data_manager.is_initialized():
            logger.info(f"Initializing data manager for process {os.getpid()}")
            try:
                data_manager._initialize_process_local(config)
            except Exception as e:
                logger.error(f"Error initializing data manager: {e}")
                logger.error(traceback.format_exc())
                # Try emergency repair
                from src.common.managers.manager_repair import repair_manager
                repair_manager(data_manager, "DataManager")
                
        # Now initialize metrics manager (depends on CUDA)
        from src.common.managers import get_metrics_manager
        metrics_manager = get_metrics_manager()
        if not metrics_manager.is_initialized():
            logger.info(f"Initializing metrics manager for process {os.getpid()}")
            try:
                metrics_manager._initialize_process_local(config)
            except Exception as e:
                logger.error(f"Error initializing metrics manager: {e}")
                logger.error(traceback.format_exc())
                # Try emergency repair
                from src.common.managers.manager_repair import repair_manager
                repair_manager(metrics_manager, "MetricsManager")
        
        # Finally, check all managers for health and repair if needed
        logger.info(f"Verifying manager health for process {os.getpid()}")
        manager_status = check_manager_health_and_repair(config)
        
        # Log any remaining issues
        failed_managers = [name for name, status in manager_status.items() if not status]
        if failed_managers:
            logger.error(f"The following managers could not be repaired: {failed_managers}")
        else:
            logger.info(f"All managers are properly initialized for process {os.getpid()}")
            
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
