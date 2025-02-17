# src/common/study/objective_factory.py
"""Factory for creating Optuna objectives."""

from __future__ import annotations

import logging
import os
import gc
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
import optuna
from torch.utils.data import DataLoader, Dataset
from optuna.trial import FixedTrial
from torch.nn import Module
from torch.optim import Optimizer

from src.common.managers import (
    get_cuda_manager,
    get_dataloader_manager,
    get_tokenizer_manager,
    get_model_manager,
    get_resource_manager,
    get_parameter_manager,
    get_wandb_manager,
    get_shared_tokenizer
)
from src.embedding.dataset import EmbeddingDataset
from src.embedding.embedding_trainer import EmbeddingTrainer
from src.common.utils import create_directory

logger = logging.getLogger(__name__)

class ObjectiveFactory:
    """Factory for creating Optuna objective functions."""

    def __init__(self, config: Dict[str, Any], output_dir: str):
        """
        Initialize the factory.

        Args:
            config: Training configuration
            output_dir: Directory for saving outputs.  This is the *main* output
                directory, not the trial-specific one.
        """
        self.config = config
        self.output_dir = output_dir
        self.pid = os.getpid()
        logger.info(f"ObjectiveFactory initialized for process {self.pid}")

    def objective(self, trial: optuna.Trial) -> float:
        """Process-local objective function for Optuna optimization."""

        current_pid = os.getpid()
        logger.info(f"Running trial {trial.number} in process {current_pid}")

        try:
            parameter_manager = get_parameter_manager()
            trial_config = parameter_manager.get_trial_config(trial)
            if trial_config["training"]["num_trials"] > 1:
                wandb_manager = get_wandb_manager()
                wandb_manager.init_trial(trial.number)

            cuda_manager = get_cuda_manager()
            data_manager = get_data_manager()
            tokenizer_manager = get_tokenizer_manager()
            directory_manager = get_directory_manager()
            model_manager = get_model_manager()

            device = cuda_manager.get_device()

            tokenizer = get_shared_tokenizer()
            train_loader, val_loader, train_dataset, val_dataset = data_manager.create_dataloaders(
                config = trial_config
            )
            logger.info(f"Created dataloaders in process {self.pid}")

            from src.embedding.models import embedding_model_factory
            model = embedding_model_factory(trial_config, trial=trial)
            model = model.to(device)
            logger.info(f"Created model in process {self.pid}")

            trial_output_dir = directory_manager.base_dir / "trials" / f"trial_{trial.number}"
            metrics_dir = create_directory(trial_output_dir / "metrics")
            from src.common.study.trial_state_manager import TrialStateManager
            trial_state_manager = TrialStateManager(trial, trial_config)

            trainer = EmbeddingTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=trial_config,
                metrics_dir=metrics_dir,
                is_trial=True,
                trial=trial,
                wandb_manager=wandb_manager if trial_config["training"]["num_trials"] > 1 else None,
                job_id=trial.number,
                train_dataset=train_dataset,
                val_dataset=val_dataset
            )
            logger.info(f"Created trainer in process {self.pid}")

            trainer.train(trial_config['training']['num_epochs'])
            trial_state_manager.update_state(TrialStatus.COMPLETED)
            return trainer.best_val_loss

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}", exc_info=True)
            if 'trial_state_manager' in locals():
                trial_state_manager.update_state(TrialStatus.FAILED)
            raise optuna.TrialPruned()

__all__ = ['ObjectiveFactory']