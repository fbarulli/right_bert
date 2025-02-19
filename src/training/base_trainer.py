
# src/training/base_trainer.py
from __future__ import annotations
import logging
import gc
import os
import traceback
import datetime
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from contextlib import contextmanager

import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from torch.optim import Optimizer
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup

from src.common.managers.cuda_manager import CUDAManager
from src.common.managers.batch_manager import BatchManager
from src.common.managers.amp_manager import AMPManager
from src.common.managers.tokenizer_manager import TokenizerManager
from src.common.managers.metrics_manager import MetricsManager
from src.common.managers.storage_manager import StorageManager
from src.common.managers.wandb_manager import WandbManager

logger = logging.getLogger(__name__)

class BaseTrainer:
    """Base trainer class with common functionality."""

    def __init__(
        self,
        cuda_manager: CUDAManager,
        batch_manager: BatchManager,
        amp_manager: AMPManager,
        tokenizer_manager: TokenizerManager,
        metrics_manager: MetricsManager,
        storage_manager: StorageManager,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        metrics_dir: Path,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        is_trial: bool = False,
        trial: Optional[optuna.Trial] = None,
        wandb_manager: Optional[WandbManager] = None,
        job_id: Optional[int] = None,
        train_dataset: Optional[Any] = None,
        val_dataset: Optional[Any] = None
    ):
        """
        Initialize trainer with dependency injection.

        Args:
            cuda_manager: Injected CUDAManager instance
            batch_manager: Injected BatchManager instance
            amp_manager: Injected AMPManager instance
            tokenizer_manager: Injected TokenizerManager instance
            metrics_manager: Injected MetricsManager instance
            storage_manager: Injected StorageManager instance
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Configuration dictionary
            metrics_dir: Directory to save metrics
            scheduler: Optional learning rate scheduler
            is_trial: Whether this is an Optuna trial
            trial: Optional Optuna trial object
            wandb_manager: Optional WandbManager for logging
            job_id: Optional job ID
            train_dataset: Optional training dataset (for memory cleanup)
            val_dataset: Optional validation dataset (for memory cleanup)
        """
        # Store injected managers
        self._cuda_manager = cuda_manager
        self._batch_manager = batch_manager
        self._amp_manager = amp_manager
        self._tokenizer_manager = tokenizer_manager
        self._metrics_manager = metrics_manager
        self._storage_manager = storage_manager
        self._wandb_manager = wandb_manager

        # Initialize model and config
        self.model = model
        self.config = config
        self.metrics_dir = metrics_dir
        self.is_trial = is_trial
        self.trial = trial
        self.job_id = job_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Training configuration
        training_config = config['training']
        self.grad_norm = training_config['max_grad_norm']
        self.gradient_accumulation = training_config['gradient_accumulation_steps']
        self.log_interval = training_config['logging_steps']
        self.eval_interval = training_config['eval_steps']
        self.use_amp = training_config['fp16']

        # Profiler configuration
        profiler_config = training_config['profiler']
        self.use_profiler = profiler_config['enabled']
        self.profiler_config = profiler_config

        # CUDA graph configuration
        cuda_graph_config = training_config['cuda_graph']
        self.use_cuda_graph = cuda_graph_config['enabled']
        self.cuda_graph_config = cuda_graph_config
        self.static_batch = None
        self.static_graph = None

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device setup
        self.device = self._cuda_manager.get_device()
        self.model = self.model.to(self.device)

        # Initialize weights
        self._initialize_weights()

        # Initialize metrics logger
        self.metrics_logger = metrics_manager.MetricsLogger(
            metrics_dir=metrics_dir,
            is_trial=is_trial,
            trial=trial,
            wandb_manager=wandb_manager,
            job_id=job_id
        )

        # Training state
        self.current_step = 0
        self.accumulation_step = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.scheduler = scheduler
        self.pbar = None
        self.global_step = 0
        self.train_steps = len(train_loader)
        self.val_steps = len(val_loader)

        # Early stopping configuration
        self.early_stopping_triggered = False
        self.early_stopping_patience = training_config['early_stopping_patience']
        self.early_stopping_min_delta = training_config['early_stopping_min_delta']

        # Initialize CUDA graph if enabled
        if self.use_cuda_graph:
            self._cuda_manager.create_graph(self.forward_static)

        logger.info(
            f"Initialized trainer with:\n"
            f"- Device: {self.device}\n"
            f"- AMP: {self.use_amp}\n"
            f"- Gradient accumulation: {self.gradient_accumulation}\n"
            f"- CUDA graph: {self.use_cuda_graph}"
        )

    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        def _init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.model.apply(_init_weights)

    def create_optimizer(self, params=None) -> Optimizer:
        """
        Create optimizer.

        Args:
            params: Optional model parameters. If None, uses self.model.named_parameters()

        Returns:
            Optimizer: The created optimizer

        Raises:
            ValueError: If unsupported optimizer type specified
        """
        try:
            # Group parameters
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in self.model.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.config['training']['weight_decay']
                },
                {
                    'params': [p for n, p in self.model.named_parameters()
                            if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0
                }
            ]

            # Create optimizer based on type
            optimizer_type = self.config['training']['optimizer_type'].lower()
            if optimizer_type == 'adamw':
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=self.config['training']['learning_rate'],
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
            elif optimizer_type == 'adam':
                optimizer = torch.optim.Adam(
                    optimizer_grouped_parameters,
                    lr=self.config['training']['learning_rate'],
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
            elif optimizer_type == 'sgd':
                optimizer = torch.optim.SGD(
                    optimizer_grouped_parameters,
                    lr=self.config['training']['learning_rate'],
                    momentum=0.9
                )
            else:
                raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

            logger.info(
                f"Created optimizer: {optimizer_type}\n"
                f"Learning rate: {self.config['training']['learning_rate']}\n"
                f"Weight decay: {self.config['training']['weight_decay']}"
            )

            return optimizer

        except Exception as e:
            logger.error(f"Error creating optimizer: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[Union[Dict[str, torch.Tensor], 'transformers.modeling_outputs.ModelOutput'],
               Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            batch: Input batch

        Returns:
            Tuple containing:
            - Model outputs
            - Device batch

        Raises:
            RuntimeError: If model outputs missing loss or forward pass fails
        """
        try:
            with record_function("batch_to_device"):
                device_batch = self._batch_manager.prepare_batch(batch, self.device)

            if self.use_cuda_graph:
                with record_function("cuda_graph_replay"):
                    self._cuda_manager.replay_graph()
                    return self.static_outputs, device_batch

            with record_function("model_forward"), self._amp_manager.autocast():
                outputs = self.model(**device_batch)

                # Verify loss exists in outputs
                if not (hasattr(outputs, 'loss') or
                       (isinstance(outputs, dict) and 'loss' in outputs)):
                    logger.error(f"Model outputs missing loss: {type(outputs)}")
                    if hasattr(outputs, '__dict__'):
                        logger.error(f"Available attributes: {outputs.__dict__.keys()}")
                    elif isinstance(outputs, dict):
                        logger.error(f"Available keys: {outputs.keys()}")
                    raise RuntimeError("Model outputs missing loss")

                # Convert to dictionary if needed
                if not isinstance(outputs, dict):
                    outputs = {
                        'loss': outputs.loss,
                        'logits': outputs.logits,
                        'hidden_states': (
                            outputs.hidden_states
                            if hasattr(outputs, 'hidden_states') else None
                        ),
                        'attentions': (
                            outputs.attentions
                            if hasattr(outputs, 'attentions') else None
                        )
                    }

                return outputs, device_batch

        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            logger.error(f"Batch keys: {batch.keys()}")
            logger.error(f"Model type: {type(self.model)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Forward pass failed: {str(e)}")

    def backward_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> None:
        """
        Backward pass.

        Args:
            loss: Loss value
            optimizer: Optimizer
        """
        with record_function("loss_scale"):
            loss = loss / self.gradient_accumulation

        with record_function("backward_step"):
            self._amp_manager.backward_step(
                loss=loss,
                model=self.model,
                optimizer=optimizer,
                grad_norm=self.grad_norm
            )

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: Optimizer
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Input batch
            optimizer: Optimizer

        Returns:
            Dict[str, float]: Training metrics

        Raises:
            ValueError: If unexpected output type
        """
        try:
            outputs, device_batch = self.forward(batch)

            # Get loss from outputs
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            elif isinstance(outputs, dict):
                loss = outputs['loss']
            else:
                raise ValueError(f"Unexpected output type: {type(outputs)}")

            # Backward pass
            scaled_loss = loss / self.gradient_accumulation
            self.backward_step(scaled_loss, optimizer)

            # Update weights if needed
            self.accumulation_step = (
                self.accumulation_step + 1
            ) % self.gradient_accumulation
            if self.accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Compute metrics
            metrics = self.compute_metrics(outputs, device_batch)
            metrics['loss'] = loss.item()
            metrics['learning_rate'] = optimizer.param_groups[0]['lr']

            return metrics

        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dict[str, float]: Validation metrics
        """
        self.model.eval()
        val_metrics = []

        try:
            with torch.no_grad():
                for batch in self.val_loader:
                    outputs, device_batch = self.forward(batch)
                    metrics = self.compute_metrics(outputs, device_batch)
                    val_metrics.append(metrics)

            # Compute average metrics
            avg_metrics = {}
            for key in val_metrics[0].keys():
                values = [m[key] for m in val_metrics]
                avg_metrics[key] = sum(values) / len(values)

            return avg_metrics

        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        finally:
            self.model.train()

    def train_epoch(
        self,
        epoch: int,
        optimizer: Optimizer
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
            optimizer: Optimizer

        Returns:
            Dict[str, float]: Average training metrics for the epoch
        """
        try:
            self.current_val_metrics = {'loss': float('inf'), 'acc': 0.0}

            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/1 [Train]",
                dynamic_ncols=True
            )

            train_metrics = []
            profiler = self._create_profiler() if self.use_profiler else nullcontext()

            with profiler as prof:
                for batch in pbar:
                    with record_function("train_step"):
                        # Training step
                        metrics = self.train_step(batch, optimizer)
                        train_metrics.append(metrics)

                        # Update progress bar
                        display_metrics = {
                            'loss': f"{metrics['loss']:.4f}",
                            'ppl': f"{metrics['ppl']:.2f}",
                            'acc': f"{metrics['accuracy']:.2%}",
                            'val_loss': f"{self.current_val_metrics['loss']:.4f}",
                            'val_acc': f"{self.current_val_metrics['accuracy']:.2%}"
                        }
                        pbar.set_postfix(display_metrics)

                        # Log metrics if needed
                        if self.current_step % self.log_interval == 0:
                            self.log_metrics(metrics, 'train')

                        # Validate if needed
                        if self.current_step % self.eval_interval == 0:
                            self.current_val_metrics = self.validate()
                            self.log_metrics(self.current_val_metrics, 'val')

                            # Save best model
                            if self.current_val_metrics['loss'] < self.best_val_loss:
                                self.best_val_loss = self.current_val_metrics['loss']
                                if not self.is_trial:
                                    self.save_model()

                        self.current_step += 1

                        if self.use_profiler:
                            prof.step()

            # Log profiler results
            if self.use_profiler:
                logger.info(
                    prof.key_averages().table(
                        sort_by="cuda_time_total",
                        row_limit=10
                    )
                )

            # Compute average metrics
            avg_metrics = {}
            for key in train_metrics[0].keys():
                values = [m[key] for m in train_metrics]
                avg_metrics[key] = sum(values) / len(values)

            # Step scheduler if needed
            if self.scheduler is not None:
                self.scheduler.step()
                avg_metrics['learning_rate'] = self.scheduler.get_last_lr()[0]

            return avg_metrics

        except Exception as e:
            logger.error(f"Error in epoch {epoch}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def train(self, num_epochs: int) -> None:
        """
        Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
        """
        try:
            with record_function("train_setup"):
                # Get or create optimizer
                optimizer = getattr(self, '_optimizer', None)
                if optimizer is None:
                    optimizer = self.create_optimizer()
                scheduler = getattr(self, 'scheduler', None)

                logger.info(
                    "Starting training with profiling enabled"
                    if self.use_profiler else "Starting training"
                )

            for epoch in range(num_epochs):
                with record_function(f"epoch_{epoch}"):
                    # Train epoch
                    train_metrics = self.train_epoch(epoch, optimizer)
                    self.log_metrics(train_metrics, f'train_epoch_{epoch}')

                    # Validate
                    with record_function("validation"):
                        val_metrics = self.validate()
                        self.log_metrics(val_metrics, f'val_epoch_{epoch}')

                    # Save best model
                    with record_function("model_saving"):
                        if val_metrics['loss'] < self.best_val_loss:
                            self.best_val_loss = val_metrics['loss']
                            if not self.is_trial:
                                self.save_model()

                    # Step scheduler
                    if scheduler is not None:
                        scheduler.step()
                        logger.debug(
                            f"Stepped scheduler, new LR: "
                            f"{scheduler.get_last_lr()[0]:.2e}"
                        )

                    # Clean up memory
                    with record_function("memory_cleanup"):
                        self.cleanup_memory()

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        finally:
            if self.use_profiler:
                self._cleanup_profiler()

    def log_metrics(
        self,
        metrics: Dict[str, float],
        phase: str
    ) -> None:
        """
        Log metrics.

        Args:
            metrics: Metrics to log
            phase: Phase (e.g., 'train', 'val')
        """
        try:
            self.metrics_logger.log_metrics(
                metrics,
                phase,
                step=self.current_step
            )

            if self.is_trial and self.trial and phase.startswith('val'):
                self.trial.report(float(metrics['loss']), step=self.current_step)

                if self.trial.should_prune():
                    raise optuna.TrialPruned()

        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _create_profiler(self):
        """
        Create PyTorch profiler.

        Returns:
            torch.profiler.profile: Configured profiler
        """
        try:
            schedule_config = self.profiler_config['schedule']
            activities = []
            if 'cpu' in self.profiler_config['activities']:
                activities.append(ProfilerActivity.CPU)
            if 'cuda' in self.profiler_config['activities']:
                activities.append(ProfilerActivity.CUDA)

            profiler_dir = self._storage_manager.get_profiler_dir(
                self.trial.number if self.is_trial else None
            )

            profiler = profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=schedule_config['wait'],
                    warmup=schedule_config['warmup'],
                    active=schedule_config['active'],
                    repeat=schedule_config['repeat']
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(profiler_dir)
                ),
                record_shapes=self.profiler_config['record_shapes'],
                profile_memory=self.profiler_config['profile_memory'],
                with_stack=self.profiler_config['with_stack'],
                with_flops=self.profiler_config['with_flops']
            )

            if self.profiler_config['export_chrome_trace']:
                def export_chrome_trace(prof):
                    prof.export_chrome_trace(
                        str(profiler_dir / 'trace.json')
                    )
                profiler.on_trace_ready = export_chrome_trace

            return profiler

        except Exception as e:
            logger.error(f"Error creating profiler: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _cleanup_profiler(self) -> None:
        """Clean up profiler."""
        try:
            if hasattr(self, '_profiler') and self._profiler is not None:
                self._profiler.stop()
                del self._profiler
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if self.is_trial:
                self._storage_manager.cleanup_profiler(self.trial.number)
            else:
                self._storage_manager.cleanup_profiler()

        except Exception as e:
            logger.error(f"Error cleaning up profiler: {str(e)}")
            logger.error(traceback.format_exc())

    def compute_metrics(
        self,
        outputs: Union[Dict[str, torch.Tensor], 'transformers.modeling_outputs.ModelOutput'],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute metrics.

        Args:
            outputs: Model outputs
            batch: Input batch

        Returns:
            Dict[str, float]: Computed metrics
        """
        return self._metrics_manager.compute_embedding_metrics(outputs, batch)

    def save_model(self) -> None:
        """Save model."""
        if not self.is_trial:
            try:
                save_dir = self.metrics_dir.parent / 'model'
                save_dir.mkdir(exist_ok=True)

                self.model.save_pretrained(save_dir, safe_serialization=False)
                logger.info(f"Saved model to {save_dir}")

            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
                logger.error(traceback.format_exc())
                raise

    def cleanup_memory(self, aggressive: bool = False) -> None:
        """
        Clean up memory.

        Args:
            aggressive: Whether to perform aggressive cleanup
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            gc.collect()

            if aggressive and not hasattr(self, '_cleaned_up'):
                if hasattr(self, 'model') and self.model is not None:
                    self.model.cpu()

                if hasattr(self, 'train_loader'):
                    del self.train_loader
                if hasattr(self, 'val_loader'):
                    del self.val_loader

                if hasattr(self, 'train_dataset'):
                    del self.train_dataset
                if hasattr(self, 'val_dataset'):
                    del self.val_dataset

                if self.is_trial and self.trial:
                    self._tokenizer_manager.cleanup_worker(self.trial.number)

                gc.collect()
                self._cleaned_up = True

        except Exception as e:
            logger.error(f"Error during memory cleanup: {str(e)}")
            logger.error(traceback.format_exc())

    def plot_metrics(self) -> None:
        """Plot metrics (use trial_analyzer)."""
        pass


@contextmanager
def nullcontext():
    """Context manager that does nothing."""
    yield