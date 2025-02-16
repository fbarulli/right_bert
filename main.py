# main.py
import logging
import os
import traceback
from pathlib import Path
from typing import Dict, Any
import multiprocessing as mp

from src.common.utils import setup_logging, seed_everything, load_yaml_config
from src.common.managers import (
    get_data_manager,
    get_model_manager,
    get_tokenizer_manager,
    get_directory_manager,
    get_parameter_manager,
    get_wandb_manager,
    set_shared_tokenizer,
    get_shared_tokenizer
)
from src.embedding.models import embedding_model_factory
from src.embedding.embedding_training import train_embeddings, validate_embeddings
from src.common.managers import get_optuna_manager
from src.common.managers.worker_manager import WorkerManager

logger = logging.getLogger(__name__)

def train_model(config: Dict[str, Any]) -> None:
    try:
        seed_everything(config['training']['seed'])
        output_dir = Path(config['output']['dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        data_manager = get_data_manager()
        model_manager = get_model_manager()
        tokenizer_manager = get_tokenizer_manager()
        directory_manager = get_directory_manager()
        parameter_manager = get_parameter_manager()
        wandb_manager = get_wandb_manager()

        if config['model']['stage'] == 'embedding':
            logger.info("\n=== Starting Embedding Training ===")
            tokenizer = tokenizer_manager.get_worker_tokenizer(0, config['model']['name'])
            set_shared_tokenizer(tokenizer)
            train_loader, val_loader, train_dataset, val_dataset = data_manager.create_dataloaders(
                data_path=Path(config['data']['csv_path']),
                tokenizer=tokenizer,
                max_length=config['data']['max_length'],
                batch_size=config['training']['batch_size'],
                train_ratio=config['data']['train_ratio'],
                is_embedding=True,
                mask_prob=config['data']['embedding_mask_probability'],
                max_predictions=config['data']['max_predictions'],
                max_span_length=config['data']['max_span_length'],
                num_workers=config['training']['num_workers']
            )

            model = embedding_model_factory(config)

            train_embeddings(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                metrics_dir= str(directory_manager.base_dir / "metrics"),
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
    parameter_manager = get_parameter_manager()
    config = parameter_manager.get_trial_config(trial)
    # wandb_manager = get_wandb_manager(config, "embedding_study")
    # wandb_manager.init_trial(trial.number)

    train_model(config)

    # Get validation metrics from the trial
    # Assuming best_val_loss is stored as a user attribute
    best_val_loss = trial.user_attrs.get('best_val_loss', float('inf'))
    return best_val_loss  # Optuna minimizes the objective

def main():
    """Main entry point."""
    try:
        from src.common.tensorflow_init import init
    except Exception as e:
        print("tf init failed", e)
    try:
        logger.info(f"Main Process ID: {os.getpid()}")
        setup_logging(log_config = {'level': 'INFO'})
        logger.info("Loading configuration...")

        config = load_yaml_config("config/embedding_config.yaml")
        if not config:
            logger.error("Failed to load configuration. Exiting.")
            return
        
        # Get the Optuna Manager
        optuna_manager = get_optuna_manager()
        # Create/Load the study
        study = optuna_manager.study
        logger.info("Launching Optuna Study")
        study.optimize(objective, n_trials=config["training"]["num_trials"], n_jobs=config["training"]["n_jobs"])

        logger.info("Configuration loaded successfully")
        logger.info("\n=== Starting Training ===")
        # train_model(config)  # Removed for now Optuna takes over
        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        from src.common.resource.resource_initializer import ResourceInitializer
        logger.info("Cleaning up resources...")
        ResourceInitializer.cleanup_process()

if __name__ == "__main__":
    # Use spawn method
    mp.set_start_method('spawn', force=True)
    main()