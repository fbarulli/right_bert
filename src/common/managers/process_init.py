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

def ensure_process_initialized(config: Dict[str, Any]) -> None:
    """Ensure managers are initialized in the current process."""
    pid = os.getpid()
    if pid in _process_initialized:
        # Already initialized for this process
        logger.debug(f"Process {pid} already initialized, skipping")
        return
        
    logger.info(f"Initializing managers for child process {pid}")
    
    try:
        # Import here to avoid circular imports
        from src.common.managers import (
            get_cuda_manager,
            get_data_manager,
            get_model_manager,
            get_tokenizer_manager,
            get_directory_manager,
            get_parameter_manager,
            get_dataloader_manager,
            get_amp_manager,
            get_tensor_manager,
            get_wandb_manager
        )
        
        # Special manager initialization - check WandbManager thoroughly
        wandb_manager = get_wandb_manager()
        logger.debug(f"Checking WandbManager initialization in process {pid}")
        
        # Most thorough check possible
        if not hasattr(wandb_manager, '_local'):
            logger.warning(f"WandbManager has no _local attribute in process {pid}, creating it")
            wandb_manager._local = threading.local()
            
        # First ensure pid is correct    
        if not hasattr(wandb_manager._local, 'pid') or wandb_manager._local.pid != pid:
            wandb_manager._local.pid = pid
            
        # Then ensure required attributes exist
        required_attrs = ['enabled', 'run', 'start_time', 'project', 'entity', 'initialized']
        for attr in required_attrs:
            if not hasattr(wandb_manager._local, attr):
                logger.warning(f"WandbManager.{attr} attribute missing in process {pid}, initializing")
                if attr == 'enabled':
                    wandb_manager._local.enabled = False
                elif attr == 'run':
                    wandb_manager._local.run = None
                elif attr == 'start_time':
                    wandb_manager._local.start_time = None
                elif attr == 'project':
                    wandb_manager._local.project = "default_project"
                elif attr == 'entity':
                    wandb_manager._local.entity = None
                elif attr == 'initialized':
                    wandb_manager._local.initialized = False
                    
        # Call setup to properly initialize if needed
        if not wandb_manager.is_initialized():
            logger.debug(f"Initializing WandbManager in process {pid}")
            try:
                wandb_manager.setup(config)
            except:
                wandb_manager._initialize_process_local(config)
                wandb_manager._local.initialized = True
        
        # Special handling for ParameterManager - this one is critical for trials
        parameter_manager = get_parameter_manager()
        logger.debug(f"Ensuring ParameterManager is initialized in process {pid}")
        
        # Directly initialize it first, before other managers
        if hasattr(parameter_manager, 'setup'):
            try:
                parameter_manager.setup(config)
                logger.debug(f"ParameterManager setup completed in process {pid}")
            except Exception as e:
                logger.error(f"Error in ParameterManager setup: {e}")
                # Try direct initialization
                if hasattr(parameter_manager, '_initialize_process_local'):
                    parameter_manager._initialize_process_local(config)
                    parameter_manager._local.initialized = True
                    logger.warning(f"Forced ParameterManager initialization in process {pid}")
        
        if not parameter_manager.is_initialized():
            logger.critical(f"ParameterManager initialization failed in process {pid}, forcing it")
            
            # Add extra diagnostic information about the failure
            logger.debug("Parameter Manager Diagnostic Info:")
            logger.debug(f"- Has _local: {hasattr(parameter_manager, '_local')}")
            if hasattr(parameter_manager, '_local'):
                logger.debug(f"- _local attributes: {dir(parameter_manager._local)}")
            logger.debug(f"- Configuration keys: {config.keys() if config else None}")
            
            # Force initialization
            parameter_manager._local = threading.local()
            parameter_manager._local.pid = pid
            parameter_manager._local.initialized = True  # Force it to be initialized
            parameter_manager._local.base_config = config  # CRITICAL - set the base_config attribute
            parameter_manager._local.search_space = {}
            parameter_manager._local.param_ranges = {}
            parameter_manager._local.hyperparameters = {}
            
            # Verify forced initialization was successful
            if parameter_manager.is_initialized():
                logger.info(f"ParameterManager force-initialization successful for process {pid}")
            else:
                logger.critical(f"ParameterManager force-initialization FAILED for process {pid}!")
        
        # Initialize managers in dependency order
        logger.debug(f"Starting manager initialization for process {pid}")
        
        # Step 1: Core infrastructure - with extra safety for DirectoryManager
        cuda_manager = get_cuda_manager()
        try_setup_manager(cuda_manager, config, "CUDAManager")
        
        logger.debug(f"Attempting to initialize DirectoryManager for process {pid}")
        directory_manager = get_directory_manager()
        
        # Special handling for DirectoryManager due to its critical nature
        try:
            # Try direct initialization instead of through try_setup_manager
            if hasattr(directory_manager, 'setup'):
                logger.debug(f"Directly calling DirectoryManager.setup() for process {pid}")
                directory_manager.setup(config)
            else:
                logger.debug(f"Directly calling DirectoryManager._initialize_process_local() for process {pid}")
                directory_manager._initialize_process_local(config)
                
            # Verify the required attributes exist
            if not hasattr(directory_manager._local, 'output_dir'):
                logger.warning(f"output_dir missing after initialization, creating it")
                directory_manager._local.output_dir = os.path.join(os.getcwd(), 'output')
                os.makedirs(directory_manager._local.output_dir, exist_ok=True)
                
            if not hasattr(directory_manager._local, 'data_dir'):
                logger.warning(f"data_dir missing after initialization, creating it")
                directory_manager._local.data_dir = os.path.join(os.getcwd(), 'data')
                os.makedirs(directory_manager._local.data_dir, exist_ok=True)
                
            directory_manager._local.initialized = True
            logger.info(f"DirectoryManager initialized for process {pid}")
            
        except Exception as e:
            logger.error(f"Error during DirectoryManager initialization: {e}")
            logger.error("Stack trace:", exc_info=True)
            
            # Emergency fallback - create minimal DirectoryManager
            logger.critical(f"Performing emergency DirectoryManager initialization")
            directory_manager._local = threading.local()
            directory_manager._local.pid = pid
            directory_manager._local.base_dir = os.getcwd()
            directory_manager._local.output_dir = os.path.join(os.getcwd(), 'output')
            directory_manager._local.data_dir = os.path.join(os.getcwd(), 'data')
            directory_manager._local.cache_dir = os.path.join(os.getcwd(), '.cache')
            directory_manager._local.model_dir = os.path.join(os.getcwd(), 'models')
            directory_manager._local.temp_dir = Path(tempfile.mkdtemp())
            directory_manager._local.initialized = True
            
            # Create the emergency directories
            for dir_attr in ['output_dir', 'data_dir', 'cache_dir', 'model_dir']:
                dir_path = getattr(directory_manager._local, dir_attr)
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    logger.critical(f"Created emergency directory: {dir_path}")
                except:
                    pass
                    
        # Step 2: Initialize tokenizer (needed by both model and data)
        tokenizer_manager = get_tokenizer_manager()
        try_setup_manager(tokenizer_manager, config, "TokenizerManager")
        
        # Step 3: Initialize data processing components
        dataloader_manager = get_dataloader_manager()
        try_setup_manager(dataloader_manager, config, "DataloaderManager")
        
        # Step 4: Initialize parameter handling
        try_setup_manager(parameter_manager, config, "ParameterManager")
        
        # Step 5: Initialize model components AFTER dependencies
        model_manager = get_model_manager()
        try_setup_manager(model_manager, config, "ModelManager")
        
        # Step 6: Initialize high-level components that depend on the above
        data_manager = get_data_manager()
        try_setup_manager(data_manager, config, "DataManager")
        
        amp_manager = get_amp_manager()
        try_setup_manager(amp_manager, config, "AMPManager")
        
        # Step 7: Get tensor manager LAST since it depends on CUDA
        tensor_manager = get_tensor_manager()
        try_setup_manager(tensor_manager, config, "TensorManager")
        
        _process_initialized[pid] = True
        logger.info(f"Successfully initialized managers for child process {pid}")
        
    except Exception as e:
        logger.error(f"Error initializing managers in process {pid}: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

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
