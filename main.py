# main.py
import logging
import os
import traceback
import threading  # Add this import
import tempfile   # Add this for tempfile.mkdtemp() used elsewhere
from pathlib import Path
from typing import Dict, Any

import optuna
from src.common.config_utils import load_yaml_config
from src.common.utils import setup_logging, seed_everything
from src.common.resource.resource_initializer import ResourceInitializer

# Manager imports
from src.common.managers import (
    initialize_factory,
    get_cuda_manager,
    get_data_manager,
    get_model_manager,
    get_tokenizer_manager,
    get_directory_manager,
    get_amp_manager,
    get_storage_manager,
    get_parameter_manager,
    cleanup_managers
)
from src.common.managers.wandb_manager import WandbManager
from src.common.managers.optuna_manager import OptunaManager
from src.common.managers.parameter_manager import ParameterManager

# Embedding imports
from src.embedding.model import embedding_model_factory
from src.embedding.embedding_training import train_embeddings

logger = logging.getLogger(__name__)

# Define HARDCODED_FIELDS variable
HARDCODED_FIELDS = {
    'training': {
        'cuda_graph': {
            'enabled': False,
            'warmup_steps': 0
        },
        'profiler': {
            'enabled': False,
            'activities': [],
            'schedule': {},
            'record_shapes': False,
            'profile_memory': False,
            'with_stack': False,
            'with_flops': False,
            'export_chrome_trace': False
        }
    },
    'model': {
        'config': {
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072
        }
    }
}

def train_model(config: Dict[str, Any], wandb_manager=None) -> None:
    """Train the model using managers."""
    try:
        from src.common.managers import (
            get_cuda_manager,
            get_data_manager,
            get_model_manager,
            get_tokenizer_manager,
            get_directory_manager
        )
        from src.common.managers.process_init import ensure_process_initialized

        # Ensure all managers are initialized in this process
        logger.debug(f"Ensuring process {os.getpid()} is properly initialized in train_model()")
        ensure_process_initialized(config)

        seed_everything(config['training']['seed'])
        output_dir = Path(config['output']['dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get managers - they should now be properly initialized
        cuda_manager = get_cuda_manager()
        tokenizer_manager = get_tokenizer_manager()
        data_manager = get_data_manager()
        model_manager = get_model_manager()
        directory_manager = get_directory_manager()
        
        # Ensure the configuration contains the required paths
        if 'paths' not in config:
            logger.warning(f"No 'paths' section found in config, adding default paths")
            config['paths'] = {
                'base_dir': os.getcwd(),
                'output_dir': 'output',
                'data_dir': 'data',
                'cache_dir': '.cache',
                'model_dir': 'models'
            }

        # Handle DirectoryManager initialization with maximum robustness
        if not directory_manager.is_initialized():
            logger.warning(f"DirectoryManager not initialized after ensure_process_initialized. Trying fallbacks.")
            
            # First try setup
            try:
                directory_manager.setup(config)
                # Force initialize the flag if needed
                directory_manager._local.initialized = True
                directory_manager._local.pid = os.getpid()
                logger.info("DirectoryManager setup completed and initialized flag set")
            except Exception as e:
                logger.error(f"DirectoryManager setup failed: {e}")
                
                # Second try: force-initialization
                try:
                    from src.common.managers.force_init import force_directory_manager_init
                    directory_manager = force_directory_manager_init(config)
                    logger.warning("Using force-initialized DirectoryManager")
                except Exception as e2:
                    logger.error(f"Force initialization failed: {e2}")
                    
                    # Last resort: Use persistent directory manager
                    from src.common.managers.persistent_directory_manager import PersistentDirectoryManager
                    directories = PersistentDirectoryManager.get_directories(config)
                    metrics_dir = str(directories['output_dir'] / "metrics")
                    os.makedirs(metrics_dir, exist_ok=True)
                    logger.warning(f"Using persistent directory fallback: {metrics_dir}")
                    
                    # ...rest of persistent directory handling...
                    return
        
        # Check one more time
        if not directory_manager.is_initialized():
            # EMERGENCY HACK: Override initialization flag
            logger.critical("EMERGENCY: Forcing DirectoryManager initialized flag")
            directory_manager._local = threading.local()
            directory_manager._local.pid = os.getpid()
            directory_manager._local.initialized = True
            directory_manager._local.base_dir = Path(os.getcwd())
            directory_manager._local.output_dir = Path(os.getcwd()) / "output"
            directory_manager._local.data_dir = Path(os.getcwd()) / "data"
            directory_manager._local.cache_dir = Path(os.getcwd()) / ".cache"
            directory_manager._local.model_dir = Path(os.getcwd()) / "models"
            directory_manager._local.temp_dir = Path(tempfile.mkdtemp())
            
            # Create these directories
            os.makedirs(directory_manager._local.output_dir, exist_ok=True)
            os.makedirs(directory_manager._local.data_dir, exist_ok=True)
            os.makedirs(directory_manager._local.cache_dir, exist_ok=True)
            os.makedirs(directory_manager._local.model_dir, exist_ok=True)
        
        # Skip is_initialized check by using direct attribute access for metrics_dir
        metrics_dir = None
        try:
            metrics_dir = str(Path(directory_manager._local.output_dir) / "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            logger.info(f"Created metrics directory: {metrics_dir}")
        except Exception as e:
            # Ultimate fallback
            metrics_dir = os.path.join(os.getcwd(), "output", "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            logger.warning(f"Using fallback metrics directory: {metrics_dir}")

        if config['model']['stage'] == 'embedding':
            logger.info("\n=== Starting Embedding Training ===")

            # Initialize tokenizer
            tokenizer = tokenizer_manager.get_worker_tokenizer(0, config['model']['name'])
            tokenizer_manager.set_shared_tokenizer(tokenizer)

            # Create data loaders
            train_loader, val_loader, train_dataset, val_dataset = data_manager.create_dataloaders(config)

            # Create model
            model = embedding_model_factory(config)

            # Train model
            train_embeddings(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                metrics_dir=metrics_dir,  # Use our reliably created metrics_dir
                is_trial=False,
                trial=None,
                wandb_manager=wandb_manager,
                job_id=0,
                train_dataset=train_dataset,
                val_dataset=val_dataset
            )

        else:
            raise ValueError(f"Unknown training stage: {config['model']['stage']}")

    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise

def initialize_managers(config):
    """Initialize all necessary managers."""
    try:
        parameter_manager = ParameterManager(config)
        parameter_manager._initialize_process_local()  # Ensure initialization
        study_name = config['training']['study_name']  # Extract study_name from config
        wandb_manager = WandbManager(config, study_name)  # Pass study_name to WandbManager
        return parameter_manager, wandb_manager
    except Exception as e:
        logger.error(f"Failed to initialize managers: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def objective(trial, config):
    """Objective function for Optuna optimization."""
    # Register this process as a trial worker
    from src.common.process_registry import register_process
    register_process(process_type='trial')
    
    logger.debug(f"Starting objective function for trial {trial.number} in process {os.getpid()}")
    logger.debug(f"Process ID: {os.getpid()}, Parent Process ID: {os.getppid()}")
    
    # Ensure all managers are initialized in this child process
    from src.common.managers.process_init import ensure_process_initialized
    ensure_process_initialized(config)
    
    # Get needed managers directly from the factory
    from src.common.managers import get_parameter_manager, get_wandb_manager
    parameter_manager = get_parameter_manager()
    
    # Double check parameter manager initialization - critical for trials
    if not parameter_manager.is_initialized():
        logger.critical(f"ParameterManager not initialized in process {os.getpid()} despite ensure_process_initialized")
        # Force initialize it
        parameter_manager._local = threading.local()
        parameter_manager._local.pid = os.getpid()
        parameter_manager._local.initialized = True
        # Set all essential attributes to avoid AttributeError later
        parameter_manager._local.base_config = config
        parameter_manager._local.search_space = {}
        parameter_manager._local.param_ranges = {}
        parameter_manager._local.hyperparameters = {}  # Add missing attribute
        parameter_manager._initialize_process_local(config)
        logger.warning("Forced ParameterManager initialization in objective function")
    
    # Even if initialized, make sure base_config exists
    if not hasattr(parameter_manager._local, 'base_config'):
        logger.warning("ParameterManager missing base_config attribute, setting it now")
        parameter_manager._local.base_config = config
    
    # Get trial configuration with safety
    try:
        trial_config = parameter_manager.get_trial_config(trial)
        logger.debug(f"Trial configuration obtained for trial {trial.number}")
    except Exception as e:
        logger.error(f"Error getting trial config: {str(e)}")
        # Use original config as fallback
        trial_config = config.copy()
        logger.warning(f"Using original config for trial {trial.number} due to error")
    
    # Initialize wandb for this trial
    wandb_manager = get_wandb_manager()
    if config["training"]["num_trials"] > 1:
        wandb_manager.init_trial(trial.number)
        logger.debug(f"Wandb initialized for trial {trial.number}")
    
    # Train model with this trial's configuration
    try:
        train_model(trial_config, wandb_manager=wandb_manager)
    except Exception as e:
        logger.error(f"Error during trial {trial.number}: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

    # Return trial result
    best_val_loss = trial.user_attrs.get('best_val_loss', float('inf'))
    logger.debug(f"Trial {trial.number} completed with best_val_loss: {best_val_loss}")
    return float(best_val_loss)

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate the configuration before starting training."""
    try:
        # Check required top-level sections
        required_sections = {'training', 'data', 'model', 'output', 'resources'}
        missing_sections = required_sections - set(config.keys())
        if missing_sections:
            logger.error(f"Missing required configuration sections: {missing_sections}")
            return False

        # Validate output section specifically
        output_config = config.get('output', {})
        required_output_fields = {'dir', 'storage_dir', 'wandb'}
        missing_output_fields = required_output_fields - set(output_config.keys())
        if missing_output_fields:
            logger.error(f"Missing required output configuration fields: {missing_output_fields}")
            return False

        # Create output directories
        output_dir = Path(output_config['dir'])
        storage_dir = output_dir / output_config['storage_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        storage_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Configuration validation successful")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Storage directory: {storage_dir}")
        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False

def apply_hardcoded_fields(config: Dict[str, Any]) -> None:
    """Apply hardcoded fields to the configuration."""
    for section, fields in HARDCODED_FIELDS.items():
        if section not in config:
            config[section] = {}
        for field, value in fields.items():
            if field not in config[section]:
                config[section][field] = value

def main(config_file="config/embedding_config.yaml"):
    try:
        # Enable debug logging
        logging.getLogger('src.common.managers').setLevel(logging.DEBUG)
        
        # Add debug info
        logger.debug("=== Main Process Debug ===")
        logger.debug(f"Process ID: {os.getpid()}")
        logger.debug(f"Config file: {config_file}")
        
        from src.common.config_utils import load_yaml_config
        logger.info(f"Main Process ID: {os.getpid()}")

        # Load and validate configuration
        logger.info(f"Loading configuration from {config_file}...")
        config = load_yaml_config(config_file)
        logger.debug(f"Loaded configuration: {config}")
        if not config:
            logger.error("Failed to load configuration. Exiting.")
            return

        # Apply hardcoded fields to the configuration
        apply_hardcoded_fields(config)
        logger.debug(f"Configuration after applying hardcoded fields: {config}")

        if not validate_config(config):
            logger.error("Configuration validation failed. Exiting.")
            return

        # Setup logging after config is loaded and validated
        setup_logging(config=config)

        # Initialize the manager factory
        initialize_factory(config)

        # Display prominent message about which configuration file is being used
        logger.info("\n" + "="*80)
        logger.info(f"USING CONFIGURATION: {config_file}")
        logger.info(f"Model Stage: {config['model']['stage']}")
        logger.info(f"Model Name: {config['model']['name']}")
        if 'training' in config:
            logger.info(f"Training Epochs: {config['training'].get('num_epochs', 'N/A')}")
            logger.info(f"Batch Size: {config['training'].get('batch_size', 'N/A')}")
        logger.info("="*80 + "\n")

        logger.info("Configuration loaded and factory initialized")
        logger.info("\n=== Starting Training ===")

        if config['training']['num_trials'] > 1:
            # Get initialized managers from factory
            cuda_manager = get_cuda_manager()
            amp_manager = get_amp_manager()
            storage_manager = get_storage_manager()
            parameter_manager = get_parameter_manager()
            
            # Initialize Optuna
            optuna_manager = OptunaManager(config, storage_manager, parameter_manager)
            study = optuna_manager.study
            
            logger.info("Launching Optuna Study")
            logger.debug(f"Starting Optuna study with {config['training']['n_jobs']} jobs")
            logger.debug(f"Starting Optuna study with {config['training']['n_jobs']} jobs")
            study.optimize(
                lambda trial: objective(trial, config),
                n_trials=config["training"]["num_trials"],
                n_jobs=config["training"]["n_jobs"]
            )
        else:
            train_model(config)

    except Exception as e:
        logger.error("=== Main Process Error ===")
        logger.error(f"Error: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise
    finally:
        logger.info("Cleaning up resources...")
        try:
            from src.common.managers import cleanup_managers
            cleanup_managers()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    # Register main process explicitly
    from src.common.process_registry import register_process
    register_process(process_type='main', parent_pid=0)
    
    main()
