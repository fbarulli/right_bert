# src/classification/classification_trainer.py
from __future__ import annotations

import logging
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader, Dataset

from src.common.managers import get_metrics_manager
from src.training.base_trainer import BaseTrainer


logger = logging.getLogger(__name__)

class ClassificationTrainer(BaseTrainer):
    """Trainer for fine-tuning BERT for classification tasks."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        metrics_dir: Optional[Path] = None,
        is_trial: bool = False,
        trial: Optional['optuna.Trial'] = None,
        wandb_manager: Optional['WandbManager'] = None,
        job_id: Optional[int] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None
    ) -> None:
        """Initialize classification trainer."""
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            metrics_dir=metrics_dir,
            is_trial=is_trial,
            trial=trial,
            wandb_manager=wandb_manager,
            job_id=job_id,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
        self.best_accuracy = 0.0
        self._optimizer = self.create_optimizer()  # Corrected: create optimizer here

    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute classification-specific metrics."""
        try:
            metrics_manager = get_metrics_manager()
            metrics = metrics_manager.compute_classification_metrics(outputs, batch)

            if metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = metrics['accuracy']

            return metrics

        except Exception as e:
            logger.error(f"Error computing classification metrics: {str(e)}")
            raise

    def get_current_lr(self) -> float:
        return self._optimizer.param_groups[0]['lr'] if self._optimizer else 0.0

    def cleanup_memory(self, aggressive: bool = False) -> None:
        super().cleanup_memory(aggressive)
        if aggressive:
            self.best_accuracy = 0.0