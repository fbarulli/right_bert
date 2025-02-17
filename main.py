# main.py
# main.py
import logging
import os
import traceback
from pathlib import Path
from typing import Dict, Any

import optuna

from src.common.utils import setup_logging, seed_everything
from src.common.config_utils import load_yaml_config
from src.common import (
    get_data_manager,
    get_model_manager,
    get_tokenizer_manager,
    get_directory_manager,
    get_parameter_manager,
    get_wandb_manager,
    get_optuna_manager,
    get_shared_tokenizer,
    set_shared_tokenizer,
    get_amp_manager,
    get_cuda_manager,
    get_tensor_manager,
    get_batch_manager,
    get_metrics_manager,
    get_dataloader_manager,
    get_storage_manager,
    get_resource_manager,
    get_worker_manager
)
from src.embedding.model import embedding_model_factory
from src.embedding.embedding_training import train_embeddings


logger = logging.getLogger(__name__)

def train_model(wandb_manager = None) -> None:
    """Train the model using managers from the factory."""
    try:
        from src.common.config_utils import load_yaml_config
        from src.common.managers import (
            get_factory,
            get_cuda_manager,
            get_data_manager,
            get_model_manager,
            get_tokenizer_manager,
            get_directory_manager
        )
        
        seed_everything(config['training']['seed'])
        output_dir = Path(config['output']['dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get required managers from factory
        cuda_manager = get_cuda_manager()
        data_manager = get_data_manager()
        model_manager = get_model_manager()
        tokenizer_manager = get_tokenizer_manager()
        directory_manager = get_directory_manager()


        # Setup CUDA
        cuda_manager.setup(config)

        if config['model']['stage'] == 'embedding':
            logger.info("\n=== Starting Embedding Training ===")

            # Initialize tokenizer
            tokenizer = tokenizer_manager.get_worker_tokenizer(0, config['model']['name'])
            set_shared_tokenizer(tokenizer)

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
                metrics_dir=str(directory_manager.base_dir / "metrics"),
                is_trial=False,
                trial=None,
                wandb_manager=wandb_manager,
                job_id=0,
                train_dataset=train_dataset,
                val_dataset=val_dataset
            )

        elif config['model']['stage'] == 'classification':
            logger.info("\n=== Starting Classification Training ===")
            pass

        else:
            raise ValueError(f"Unknown training stage: {config['model']['stage']}")

    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise

def objective(trial):
    """Objective function for Optuna optimization."""
    from src.common.config_utils import load_yaml_config
    from src.common.managers import get_factory
    
    # Get managers from factory
    factory = get_factory()
    parameter_manager = factory.get_parameter_manager()
    wandb_manager = factory.get_wandb_manager()
    factory = get_factory()
    
    # Get trial configuration
    config = parameter_manager.get_trial_config(trial)
    
    # Initialize wandb for this trial
    if factory.config["training"]["num_trials"] > 1:
        wandb_manager.init_trial(trial.number)

    # Train model with this trial's configuration
    train_model(wandb_manager=wandb_manager)

    # Return trial result
    best_val_loss = trial.user_attrs.get('best_val_loss', float('inf'))
    return best_val_loss

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

def main():
    try:
        from src.common.config_utils import load_yaml_config
        logger.info(f"Main Process ID: {os.getpid()}")

        
        # Load and validate configuration
        logger.info("Loading configuration...")
        config = load_yaml_config("config/embedding_config.yaml")
        if not config:
            logger.error("Failed to load configuration. Exiting.")
            return

        if not validate_config(config):
            logger.error("Configuration validation failed. Exiting.")
            return
        
        # Initialize manager factory
        from src.common.managers import initialize_factory
        config = load_yaml_config("config/embedding_config.yaml")
        initialize_factory(config)
        
        # Setup logging after config is loaded and validated
        setup_logging(config=config)
        
        logger.info("Configuration loaded and factory initialized")
        logger.info("\n=== Starting Training ===")
        
        if config['training']['num_trials'] > 1:
            from src.common.managers import get_optuna_manager
            optuna_manager = get_optuna_manager()
            study = optuna_manager.study
            logger.info("Launching Optuna Study")
            study.optimize(
                objective, 
                n_trials=config["training"]["num_trials"], 
                n_jobs=config["training"]["n_jobs"]
            )
        else:
            train_model()  # No need to pass config, it's in the factory

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        # Clean up managers and resources
        from src.common.managers import cleanup_managers
        from src.common.resource.resource_initializer import ResourceInitializer
        logger.info("Cleaning up resources...")
        #cleanup_managers() #removed
        ResourceInitializer.cleanup_process()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()