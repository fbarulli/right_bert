# src/classification/classification_trainer.py
from __future__ import annotations
import logging
import torch
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader, Dataset

from src.common.managers.cuda_manager import CUDAManager
from src.common.managers.batch_manager import BatchManager
from src.common.managers.amp_manager import AMPManager
from src.common.managers.tokenizer_manager import TokenizerManager
from src.common.managers.metrics_manager import MetricsManager
from src.common.managers.storage_manager import StorageManager
from src.common.managers.wandb_manager import WandbManager
from src.training.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

class ClassificationTrainer(BaseTrainer):
    """
    Trainer for fine-tuning BERT for classification tasks.
    
    This trainer extends BaseTrainer with:
    - Classification-specific metrics tracking
    - Best accuracy tracking
    - Classification-specific memory cleanup
    """

    def __init__(
        self,
        cuda_manager: CUDAManager,
        batch_manager: BatchManager,
        amp_manager: AMPManager,
        tokenizer_manager: TokenizerManager,
        metrics_manager: MetricsManager,
        storage_manager: StorageManager,
        model: torch.nn.Module,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        metrics_dir: Optional[Path] = None,
        is_trial: bool = False,
        trial: Optional['optuna.Trial'] = None,
        wandb_manager: Optional[WandbManager] = None,
        job_id: Optional[int] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None
    ) -> None:
        """
        Initialize classification trainer with dependency injection.

        Args:
            cuda_manager: Injected CUDAManager instance
            batch_manager: Injected BatchManager instance
            amp_manager: Injected AMPManager instance
            tokenizer_manager: Injected TokenizerManager instance
            metrics_manager: Injected MetricsManager instance
            storage_manager: Injected StorageManager instance
            model: The model to train
            train_loader: Optional training data loader
            val_loader: Optional validation data loader
            config: Configuration dictionary
            metrics_dir: Optional directory for saving metrics
            is_trial: Whether this is an Optuna trial
            trial: Optional Optuna trial object
            wandb_manager: Optional WandbManager for logging
            job_id: Optional job ID
            train_dataset: Optional training dataset (for memory cleanup)
            val_dataset: Optional validation dataset (for memory cleanup)
        """
        # Initialize parent class
        super().__init__(
            cuda_manager=cuda_manager,
            batch_manager=batch_manager,
            amp_manager=amp_manager,
            tokenizer_manager=tokenizer_manager,
            metrics_manager=metrics_manager,
            storage_manager=storage_manager,
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

        # Initialize classification-specific tracking
        self.best_accuracy = 0.0

        # Create optimizer
        self._optimizer = self.create_optimizer()

        logger.info(
            f"Initialized ClassificationTrainer:\n"
            f"- Model: {type(model).__name__}\n"
            f"- Train batches: {len(train_loader) if train_loader else 0}\n"
            f"- Val batches: {len(val_loader) if val_loader else 0}"
        )

    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute classification-specific metrics.

        Args:
            outputs: Model outputs
            batch: Input batch

        Returns:
            Dict[str, float]: Computed metrics

        Raises:
            Exception: If error computing metrics
        """
        try:
            # Compute metrics using metrics manager
            metrics = self._metrics_manager.compute_classification_metrics(
                outputs,
                batch
            )

            # Update best accuracy
            if metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = metrics['accuracy']
                if self.trial:
                    self.trial.set_user_attr('best_accuracy', self.best_accuracy)

            # Log debug metrics
            logger.debug(
                f"Batch Metrics:\n"
                f"- Loss: {metrics['loss']:.4f}\n"
                f"- Accuracy: {metrics['accuracy']:.4%}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Error computing classification metrics: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_current_lr(self) -> float:
        """
        Get current learning rate from optimizer.

        Returns:
            float: Current learning rate
        """
        return (
            self._optimizer.param_groups[0]['lr']
            if self._optimizer else 0.0
        )

    def cleanup_memory(self, aggressive: bool = False) -> None:
        """
        Clean up classification-specific memory resources.

        Args:
            aggressive: Whether to perform aggressive cleanup
        """
        try:
            # Call parent cleanup
            super().cleanup_memory(aggressive)

            # Reset best accuracy if aggressive cleanup
            if aggressive:
                self.best_accuracy = 0.0

        except Exception as e:
            logger.error(f"Error during memory cleanup: {str(e)}")
            logger.error(traceback.format_exc())
            raise
