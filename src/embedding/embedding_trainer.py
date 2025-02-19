
# src/embedding/embedding_trainer.py
from __future__ import annotations
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from src.common.managers.cuda_manager import CUDAManager
from src.common.managers.batch_manager import BatchManager
from src.common.managers.amp_manager import AMPManager
from src.common.managers.tokenizer_manager import TokenizerManager
from src.common.managers.metrics_manager import MetricsManager
from src.common.managers.storage_manager import StorageManager
from src.common.managers.wandb_manager import WandbManager
from src.training.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

class EmbeddingTrainer(BaseTrainer):
    """
    Trainer for learning embeddings through masked language modeling.

    This trainer extends BaseTrainer with:
    - Learning rate scaling based on batch size
    - Linear warmup scheduler
    - Embedding-specific metrics tracking
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
        metrics_dir: Optional[str] = None,
        is_trial: bool = False,
        trial: Optional['optuna.Trial'] = None,
        wandb_manager: Optional[WandbManager] = None,
        job_id: Optional[int] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None
    ) -> None:
        """
        Initialize embedding trainer with dependency injection.

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
        # Store max_grad_norm before parent initialization
        self.max_grad_norm = config['training']['max_grad_norm']

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

        # Initialize embedding-specific tracking
        self.best_embedding_loss = float('inf')
        self.best_val_acc = 0.0

        # Scale learning rate based on batch size
        base_batch_size = 32
        current_batch_size = config['training']['batch_size']
        effective_batch_size = (
            current_batch_size * config['training']['gradient_accumulation_steps']
        )

        if effective_batch_size != base_batch_size:
            scale_factor = effective_batch_size / base_batch_size
            config['training']['learning_rate'] *= scale_factor
            logger.info(
                f"Scaled learning rate by {scale_factor:.3f}:\n"
                f"- Batch size: {current_batch_size}\n"
                f"- Gradient accumulation: {config['training']['gradient_accumulation_steps']}\n"
                f"- Effective batch size: {effective_batch_size}"
            )

        # Create optimizer
        self._optimizer = self.create_optimizer()

        # Create scheduler if enabled
        if config['training']['scheduler']['use_scheduler']:
            num_training_steps = len(train_loader) * config['training']['num_epochs']
            num_warmup_steps = int(
                num_training_steps * config['training']['scheduler']['warmup_ratio']
            )

            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self._optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            logger.info(
                f"Created linear scheduler with warmup:\n"
                f"- Warmup steps: {num_warmup_steps}\n"
                f"- Total steps: {num_training_steps}"
            )

    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute metrics and track best values.

        Args:
            outputs: Model outputs
            batch: Input batch

        Returns:
            Dict[str, float]: Computed metrics
        """
        try:
            # Compute metrics using metrics manager
            metrics = self._metrics_manager.compute_embedding_metrics(outputs, batch)

            # Update best values
            if metrics['embedding_loss'] < self.best_embedding_loss:
                self.best_embedding_loss = metrics['embedding_loss']

            if metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = metrics['accuracy']
                if self.trial:
                    self.trial.set_user_attr('best_val_acc', float(self.best_val_acc))
                    self.trial.set_user_attr(
                        'epoch_metrics',
                        self.metrics_logger.epoch_metrics
                    )

            # Log debug metrics
            logger.debug(
                f"Batch Metrics:\n"
                f"- Loss: {metrics['loss']:.4f}\n"
                f"- PPL: {metrics.get('ppl', float('inf')):.2f}\n"
                f"- Accuracy: {metrics['accuracy']:.4%}\n"
                f"- Top-5 Accuracy: {metrics['top5_accuracy']:.4%}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
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
        Clean up embedding-specific memory resources.

        Args:
            aggressive: Whether to perform aggressive cleanup
        """
        try:
            # Call parent cleanup
            super().cleanup_memory(aggressive)

            # Reset best values if aggressive cleanup
            if aggressive:
                self.best_embedding_loss = float('inf')
                self.best_val_acc = 0.0

        except Exception as e:
            logger.error(f"Error during memory cleanup: {str(e)}")
            logger.error(traceback.format_exc())
            raise