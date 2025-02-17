# src/studies.py
# src/studies.py
import logging
import os
from pathlib import Path
from typing import Dict, Any

import optuna

from src.common import (
    get_parameter_manager,
    get_data_manager,
    get_tokenizer_manager,
    get_directory_manager,
    get_wandb_manager,
    get_optuna_manager,
    get_shared_tokenizer
)
from src.embedding.models import embedding_model_factory
from src.embedding.embedding_training import train_embeddings, validate_embeddings
from src.common.study.objective_factory import ObjectiveFactory

logger = logging.getLogger(__name__)

def create_embedding_study(config: Dict[str, Any], output_path: Path):
    """Creates and runs the Optuna study for embedding model training."""

    logger.info("Creating embedding study")

    optuna_manager = get_optuna_manager()
    parameter_manager = get_parameter_manager()
    directory_manager = get_directory_manager()
    data_manager = get_data_manager()
    tokenizer_manager = get_tokenizer_manager()
    study = optuna_manager.study

    def objective(trial):
        parameter_manager = get_parameter_manager()
        config = parameter_manager.get_trial_config(trial)
        if config["training"]["num_trials"] > 1:
            wandb_manager = get_wandb_manager()
            wandb_manager.init_trial(trial.number)
        objective_factory = ObjectiveFactory(config, output_path)
        return objective_factory.objective(trial)

    try:
        if config["training"]["num_trials"] > 1:
            wandb_manager = get_wandb_manager()
            wandb_manager.init_optimization()
            logger.info("Initializin optuna optimize")
            best_trial = optuna_manager.optimize(config, output_path)
            if best_trial:
                best_params = best_trial.params
                best_value = best_trial.value
                logger.info(f"Best trial: {best_trial.number} with value: {best_value} and params: {best_params}")
            else:
                logger.error("Optimization failed or no trials completed successfully.")
                return
        else:
            trial = optuna.trial.FixedTrial(config['hyperparameters'])
            objective_factory = ObjectiveFactory(config, output_path)
            objective_factory.objective(trial)
            best_params = config['hyperparameters']
    except Exception as e:
        logger.error(f"Study optimization failed: {e}")
        return
    finally:
        if config["training"]["num_trials"] > 1:
            wandb_manager.finish()
    if config["training"]["num_trials"] > 1:
        if best_trial is None:
            logger.error("No best trial found. Cannot retrain.")
            return
        logger.info(f"Retraining with best parameters: {best_params}")
        final_config = parameter_manager.copy_base_config()
        parameter_manager.apply_fixed_params(final_config, best_params)
    else:
        final_config = parameter_manager.copy_base_config()
        parameter_manager.apply_fixed_params(final_config, config['hyperparameters'])

    final_config['data']['train_ratio'] = 1.0

    tokenizer = get_shared_tokenizer()
    train_loader, _, train_dataset, _ = data_manager.create_dataloaders(
        config = final_config
    )

    final_model = embedding_model_factory(final_config)

    output_dir = Path(final_config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    if config["training"]["num_trials"] > 1:
        wandb_manager = get_wandb_manager()
        wandb_manager.init_final_training()

    train_embeddings(
        final_model,
        train_loader,
        None,
        final_config,
        metrics_dir=str(output_dir),
        is_trial=False,
        trial=None,
        wandb_manager= wandb_manager if config["training"]["num_trials"] > 1 else None,
        job_id=0,
        train_dataset=train_dataset,
        val_dataset = None
    )
    if config["training"]["num_trials"] > 1:
        wandb_manager.finish()

    logger.info("Validating final embeddings...")
    validate_embeddings(
        model_path=str(output_dir),
        tokenizer_name= config['model']['name'],
        output_dir=str(output_dir / "validation"),
        words_to_check=["good", "bad", "movie", "film", "actor", "director", "terrible", "excellent"]
    )


def main():
    """Main function to run the embedding study."""
    from src.common.utils import load_yaml_config
    config = load_yaml_config("config/embedding_config.yaml")
    output_path = Path(config['output']['dir'])
    create_embedding_study(config, output_path)

if __name__ == "__main__":
    main()