# src/common/study/objective_factory.py
from __future__ import annotations
"""Factory for creating Optuna objectives."""
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
    get_shared_tokenizer # For sharing tokenizer
)
from src.embedding.dataset import EmbeddingDataset
from src.embedding.embedding_trainer import EmbeddingTrainer
from src.common.utils import create_directory
from src.common.study.trial_state_manager import TrialStateManager

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
            # --- Get Managers ---
            parameter_manager = get_parameter_manager()
            cuda_manager = get_cuda_manager()
            data_manager = get_data_manager()
            tokenizer_manager = get_tokenizer_manager()
            directory_manager = get_directory_manager()
            model_manager = get_model_manager()

            # --- Trial-Specific Configuration ---
            trial_config = parameter_manager.get_trial_config(trial)

            # --- Wandb Setup (per trial) ---
            if trial_config["training"]["num_trials"] > 1:
                wandb_manager = get_wandb_manager()
                wandb_manager.init_trial(trial.number)

            # --- Device Setup ---
            device = cuda_manager.get_device()

            # --- Data Loading ---
            tokenizer = get_shared_tokenizer()  # Use the SHARED tokenizer
            train_loader, val_loader, train_dataset, val_dataset = data_manager.create_dataloaders(
                data_path=Path(trial_config['data']['csv_path']),
                tokenizer=tokenizer,  # Pass the shared tokenizer
                max_length=trial_config['data']['max_length'],
                batch_size=trial_config['training']['batch_size'],
                train_ratio=trial_config['data']['train_ratio'],
                is_embedding=True,
                mask_prob=trial_config['data']['embedding_mask_probability'],
                max_predictions=trial_config['data']['max_predictions'],
                max_span_length=trial_config['data']['max_span_length'],
                num_workers=trial_config['training']['num_workers']
            )
            logger.info(f"Created dataloaders in process {self.pid}")

            # --- Model Creation ---
            from src.embedding.models import embedding_model_factory
            model = embedding_model_factory(trial_config, trial=trial)
            model = model.to(device)  # Ensure model is on the correct device
            logger.info(f"Created model in process {self.pid}")

            # --- Trial Output Directory ---
            trial_output_dir = directory_manager.base_dir / "trials" / f"trial_{trial.number}"
            metrics_dir = create_directory(trial_output_dir / "metrics")

            # --- Trial State Manager ---
            trial_state_manager = TrialStateManager(trial, trial_config)

            # --- Trainer Setup ---
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

            # --- Training ---
            trainer.train(trial_config['training']['num_epochs'])
            trial_state_manager.update_state(TrialStatus.COMPLETED)  # Mark as completed
            return trainer.best_val_loss  # Or best_val_acc, depending on optimization goal

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}", exc_info=True)
            if 'trial_state_manager' in locals():
                trial_state_manager.update_state(TrialStatus.FAILED)  # Mark as failed
            raise optuna.TrialPruned() # Make sure optuna prunes it

__all__ = ['ObjectiveFactory']