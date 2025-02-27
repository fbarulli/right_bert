# src/common/managers/metrics_manager.py
from __future__ import annotations
import logging
import torch
import torch.nn as nn
import os
import threading  # Add the missing import
import traceback  # Add for better error reporting
from typing import Dict, Any, Optional, List
import math
import json
from pathlib import Path

from src.common.managers.base_manager import BaseManager
from src.common.managers.cuda_manager import CUDAManager
from src.common.managers.wandb_manager import WandbManager
from src.common import get_tensor_manager

logger = logging.getLogger(__name__)

class MetricsLogger:
    """Logs and manages metrics during training."""

    def __init__(
        self,
        metrics_dir: Path,
        is_trial: bool = False,
        trial: Optional['optuna.Trial'] = None,
        wandb_manager: Optional[WandbManager] = None,
        job_id: Optional[int] = None
    ):
        """
        Initialize the MetricsLogger.

        Args:
            metrics_dir: Directory to save metrics to
            is_trial: If the logger is for an Optuna trial
            trial: The optuna trial object
            wandb_manager: WandbManager for logging to W&B
            job_id: Optional job ID
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.is_trial = is_trial
        self.trial = trial
        self.wandb_manager = wandb_manager
        self.job_id = job_id
        self.epoch_metrics: List[Dict[str, float]] = []
        self.batch_metrics: List[Dict[str, float]] = []

        logger.info(
            f"MetricsLogger initialized:\n"
            f"- Metrics dir: {self.metrics_dir}\n"
            f"- Is trial: {self.is_trial}\n"
            f"- Job ID: {self.job_id}"
        )

    def log_metrics(
        self,
        metrics: Dict[str, float],
        phase: str,
        step: Optional[int] = None
    ) -> None:
        """
        Log metrics to console, file, and optionally WandB.

        Args:
            metrics: Dictionary of metrics to log
            phase: Phase of training (e.g., 'train', 'val')
            step: Current training step (global step)
        """
        step = step or 0

        # Log to console
        log_str = f"[{phase}] Step {step}:" + "".join(
            f" {k}={v:.4f}," for k, v in metrics.items()
        )
        logger.info(log_str)

        # Log to WandB
        if self.wandb_manager and self.wandb_manager.enabled:
            try:
                self.wandb_manager.log_metrics(metrics, step=step)
            except Exception as e:
                logger.error(f"Error logging to WandB: {e}", exc_info=True)

        # Store metrics internally
        metrics['phase'] = phase
        metrics['step'] = step
        if 'epoch' in phase:
            self.epoch_metrics.append(metrics)
        else:
            self.batch_metrics.append(metrics)

        # Log to file
        try:
            if self.is_trial and self.trial:
                file_name = f"trial_{self.trial.number}_metrics.json"
            else:
                file_name = (
                    "final_metrics.json" if self.job_id is None
                    else f"job_{self.job_id}_metrics.json"
                )

            file_path = self.metrics_dir / file_name
            with open(file_path, 'a') as f:
                json.dump(metrics, f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Error writing metrics to file: {e}", exc_info=True)


class MetricsManager(BaseManager):
    """Manager for tracking and computing metrics."""
    
    def __init__(
        self, 
        cuda_manager: CUDAManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize metrics manager.
        
        Args:
            cuda_manager: CUDA manager instance
            config: Optional configuration dictionary
        """
        self._cuda_manager = cuda_manager
        
        # Initialize thread-local storage immediately to avoid cleanup errors
        self._local = threading.local()
        self._local.pid = os.getpid()
        self._local.initialized = False
        self._local.metrics = {}
        
        # Now call super to complete initialization
        super().__init__(config)
    
    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local state."""
        try:
            super()._initialize_process_local(config)
            
            # Ensure basic attributes exist
            if not hasattr(self._local, 'metrics'):
                self._local.metrics = {}
            
            self._local.device = self._cuda_manager.get_device()
            logger.info(f"Metrics manager initialized for process {self._local.pid}")
            
        except Exception as e:
            logger.error(f"Error initializing MetricsManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def cleanup(self) -> None:
        """Clean up metrics manager resources."""
        try:
            # First check if _local exists before trying to access/clean it
            if hasattr(self, '_local'):
                # Get PID for logging
                pid = getattr(self._local, 'pid', os.getpid())
                
                # Reset metrics
                if hasattr(self._local, 'metrics'):
                    self._local.metrics = {}
                
                logger.info(f"Cleaned up MetricsManager for process {pid}")
            
            # Always call parent cleanup
            super().cleanup()
            
        except Exception as e:
            logger.error(f"Error cleaning up MetricsManager: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local attributes.

        Args:
            config: Optional configuration dictionary that overrides the one from constructor
        """
        try:
            super()._initialize_process_local(config)

            if not self._cuda_manager.is_initialized():
                raise RuntimeError("CUDAManager must be initialized before MetricsManager")

            self._local.device = self._cuda_manager.get_device()
            self._local.loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100,
                reduction='sum'
            ).to(self._local.device)

            logger.info(
                f"MetricsManager initialized for process {self._local.pid} "
                f"using device {self._local.device}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize MetricsManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_device(self) -> torch.device:
        """
        Get the current device.

        Returns:
            torch.device: The current device (CUDA or CPU)
        """
        self.ensure_initialized()
        return self._local.device

    def get_loss_fct(self) -> nn.Module:
        """
        Get the loss function.

        Returns:
            nn.Module: The loss function
        """
        self.ensure_initialized()
        return self._local.loss_fct

    def compute_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        k: int = 1
    ) -> Dict[str, float]:
        """
        Compute top-k accuracy for masked tokens.

        Args:
            logits: Predicted logits from the model
            labels: Ground truth labels
            k: Value for top-k accuracy

        Returns:
            Dict[str, float]: Dictionary with 'top1' and 'top{k}' accuracy
        """
        self.ensure_initialized()

        try:
            # Flatten logits and labels
            active_preds = logits.view(-1, logits.size(-1))
            active_labels = labels.view(-1)

            # Consider only positions where labels != -100
            active_mask = active_labels != -100

            # If no valid positions, return 0 accuracy
            if not active_mask.any():
                logger.warning("No valid positions for accuracy calculation")
                return {'top1': 0.0, f'top{k}': 0.0}

            # Filter predictions and labels based on the mask
            active_preds = active_preds[active_mask]
            active_labels = active_labels[active_mask]

            # Get top-k predictions
            _, pred_indices = active_preds.topk(k, dim=1)

            # Check if correct predictions are in top-k
            correct_k = pred_indices.eq(active_labels.unsqueeze(1).expand_as(pred_indices))

            # Calculate accuracy
            total = active_mask.sum().item()
            top1 = correct_k[:, 0].sum().item() / total
            topk = correct_k.any(dim=1).sum().item() / total

            logger.debug(
                f"Accuracy Stats:\n"
                f"- Total predictions: {total}\n"
                f"- Top-1 correct: {int(top1 * total)}\n"
                f"- Top-{k} correct: {int(topk * total)}"
            )

            return {'top1': top1, f'top{k}': topk}

        except Exception as e:
            logger.error(f"Error computing accuracy: {str(e)}")
            logger.error(traceback.format_exc())
            return {'top1': 0.0, f'top{k}': 0.0}

    def compute_embedding_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute embedding metrics.

        Args:
            outputs: Model outputs
            batch: The input batch

        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        self.ensure_initialized()

        try:
            logits = outputs['logits']
            labels = batch['labels']

            # Calculate loss only on masked tokens
            loss = self.get_loss_fct()(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            # Get number of masked tokens
            valid_predictions = (labels.view(-1) != -100).sum().item()
            if valid_predictions == 0:
                logger.warning("No valid predictions found in batch")
                return {
                    'loss': 0.0,
                    'embedding_loss': 0.0,
                    'ppl': 1.0,
                    'accuracy': 0.0,
                    'top5_accuracy': 0.0,
                    'mask_ratio': 0.0
                }

            # Normalize loss
            normalized_loss = loss / valid_predictions

            with torch.no_grad():
                # Calculate perplexity
                try:
                    ppl = math.exp(normalized_loss.item())
                except OverflowError:
                    ppl = float('inf')

                # Compute accuracy
                accuracy = self.compute_accuracy(logits, labels)['top1']
                top5_accuracy = self.compute_accuracy(logits, labels, k=5)['top5']

                # Calculate mask ratio
                mask = labels != -100
                total_masked = mask.sum().item()
                total_tokens = labels.numel()

                return {
                    'loss': normalized_loss.item(),
                    'embedding_loss': normalized_loss.item(),
                    'ppl': ppl,
                    'accuracy': accuracy,
                    'top5_accuracy': top5_accuracy,
                    'mask_ratio': total_masked / total_tokens
                }

        except Exception as e:
            logger.error(f"Error computing embedding metrics: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'loss': float('inf'),
                'embedding_loss': float('inf'),
                'ppl': float('inf'),
                'accuracy': 0.0,
                'top5_accuracy': 0.0,
                'mask_ratio': 0.0
            }

    def compute_classification_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            outputs: Model outputs
            batch: The input batch

        Returns:
            Dict[str, float]: Dictionary with metrics
        """
        self.ensure_initialized()

        try:
            logits = outputs['logits']
            labels = batch['labels']

            # Use standard cross-entropy for classification
            loss_fct = nn.CrossEntropyLoss().to(self.get_device())
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Calculate accuracy
            _, preds = torch.max(logits, dim=1)
            correct = (preds == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total if total > 0 else 0.0

            return {
                'loss': loss.item(),
                'accuracy': accuracy
            }

        except Exception as e:
            logger.error(f"Error computing classification metrics: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'loss': float('inf'),
                'accuracy': 0.0
            }

    def cleanup(self) -> None:
        """Clean up metrics manager resources."""
        try:
            self._local.loss_fct = None
            self._local.device = None
            logger.info(f"Cleaned up MetricsManager for process {self._local.pid}")
            super().cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up MetricsManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise


__all__ = ['MetricsManager', 'MetricsLogger']