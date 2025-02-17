# src/common/study/objective_factory.py
# src/common/study/objective_factory.py (CORRECTED)
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Any

import optuna

# Corrected import: get_shared_tokenizer is in src.common, not src.common.managers
from src.common import (
    get_data_manager,
    get_model_manager,
    get_parameter_manager,
    get_directory_manager,
    get_wandb_manager,
    get_shared_tokenizer,  # Corrected import
    set_shared_tokenizer
)

from src.embedding.embedding_training import train_embeddings
from src.classification.classification_training import train_final_model

logger = logging.getLogger(__name__)

class ObjectiveFactory:
    """Factory for creating objective functions for Optuna optimization."""

    def __init__(self, config: Dict[str, Any], output_path: Path):
        """
        Initialize the ObjectiveFactory.

        Args:
            config: Configuration dictionary.
            output_path: Base output path.
        """
        self.config = config
        self.output_path = output_path

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object.

        Returns:
            float: The objective value (e.g., validation loss).
        """
        try:
            parameter_manager = get_parameter_manager()
            data_manager = get_data_manager()
            model_manager = get_model_manager()
            tokenizer_manager = get_tokenizer_manager()
            directory_manager = get_directory_manager()
            if config["training"]["num_trials"] > 1:
                wandb_manager = get_wandb_manager()

            config = parameter_manager.get_trial_config(trial)
            if config["training"]["num_trials"] > 1:
                wandb_manager.init_trial(trial.number)

            if config['model']['stage'] == 'embedding':
                logger.info("\n=== Starting Embedding Training ===")
                tokenizer = tokenizer_manager.get_worker_tokenizer(trial.number, config['model']['name'])
                set_shared_tokenizer(tokenizer)
                train_loader, val_loader, train_dataset, val_dataset = data_manager.create_dataloaders(
                    config=config
                )
                from src.embedding.model import embedding_model_factory

                model = embedding_model_factory(config, trial)

                train_embeddings(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=config,
                    metrics_dir=str(directory_manager.base_dir / "trials" / f"trial_{trial.number}" / "metrics"),
                    is_trial=True,
                    trial=trial,
                    wandb_manager= wandb_manager if config["training"]["num_trials"] > 1 else None,
                    job_id=trial.number,
                    train_dataset=train_dataset,
                    val_dataset = val_dataset
                )
            elif config['model']['stage'] == 'classification':
                logger.info("\n=== Starting Classification Training ===")
                # Assuming you have similar functions for classification
                pass  # Replace with your classification training logic
            else:
                raise ValueError(f"Unknown stage: {config['model']['stage']}")

            best_val_loss = trial.user_attrs.get('best_val_loss', float('inf'))

            return best_val_loss

        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            raise