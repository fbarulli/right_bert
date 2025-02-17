main.py
import logging
import os
import traceback
from pathlib import Path
from typing import Dict, Any

import optuna

from src.common.utils import setup_logging, seed_everything, load_yaml_config
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

def train_model(config: Dict[str, Any], wandb_manager = None) -> None:
    try:
        seed_everything(config['training']['seed'])
        output_dir = Path(config['output']['dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        cuda_manager = get_cuda_manager()
        cuda_manager.setup(config)

        data_manager = get_data_manager()
        model_manager = get_model_manager()
        tokenizer_manager = get_tokenizer_manager()
        directory_manager = get_directory_manager()
        parameter_manager = get_parameter_manager()
        amp_manager = get_amp_manager()
        tensor_manager = get_tensor_manager()
        batch_manager = get_batch_manager()
        metrics_manager = get_metrics_manager()
        dataloader_manager = get_dataloader_manager()
        storage_manager = get_storage_manager()
        resource_manager = get_resource_manager()
        optuna_manager = get_optuna_manager()
        worker_manager = get_worker_manager()


        if config['model']['stage'] == 'embedding':
            logger.info("\n=== Starting Embedding Training ===")

            tokenizer = tokenizer_manager.get_worker_tokenizer(0, config['model']['name'])
            set_shared_tokenizer(tokenizer)
            train_loader, val_loader, train_dataset, val_dataset = data_manager.create_dataloaders(
                config
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
    wandb_manager = get_wandb_manager()
    if config["training"]["num_trials"] > 1:
        wandb_manager.init_trial(trial.number)

    train_model(config, wandb_manager=wandb_manager)

    best_val_loss = trial.user_attrs.get('best_val_loss', float('inf'))
    return best_val_loss

def main():
    global _config

    try:
        logger.info(f"Main Process ID: {os.getpid()}")
        setup_logging(config = load_yaml_config("config/embedding_config.yaml"))
        logger.info("Loading configuration...")

        config = load_yaml_config("config/embedding_config.yaml")
        if not config:
            logger.error("Failed to load configuration. Exiting.")
            return
        _config = config

        logger.info("Configuration loaded successfully")
        logger.info("\n=== Starting Training ===")
        if config['training']['num_trials'] > 1:
            optuna_manager = get_optuna_manager()
            study = optuna_manager.study
            logger.info("Launching Optuna Study")
            study.optimize(objective, n_trials=config["training"]["num_trials"], n_jobs=config["training"]["n_jobs"])
        else:
            train_model(config)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        from src.common.resource.resource_initializer import ResourceInitializer
        logger.info("Cleaning up resources...")
        ResourceInitializer.cleanup_process()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()