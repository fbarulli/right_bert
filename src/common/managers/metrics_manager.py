# src/common/managers/metrics_manager.py
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import math
import json
from pathlib import Path

from src.common.managers.base_manager import BaseManager
from src.common.managers import get_cuda_manager

logger = logging.getLogger(__name__)

class MetricsLogger:
    """Logs and manages metrics during training."""
    def __init__(self,
                metrics_dir: Path,
                is_trial: bool = False,
                trial: Optional['optuna.Trial'] = None,
                wandb_manager: Optional['src.common.managers.wandb_manager.WandbManager'] = None, #Corrected type
                job_id: Optional[int] = None):
        """
        Initialize the MetricsLogger

        Args:
            metrics_dir: Directory to save metrics to
            is_trial: If the logger is for an Optuna trial
            trial: The optuna trial object
            wandb_manager: WandbManager for logging to W&B
            job_id: Optional job ID.
        """

        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        self.is_trial = is_trial
        self.trial = trial
        self.wandb_manager = wandb_manager
        self.job_id = job_id
        self.epoch_metrics: List[Dict[str, float]] = []
        self.batch_metrics: List[Dict[str, float]] = []

        logger.info(f"MetricsLogger initialized:\n"
            f"- Metrics dir: {self.metrics_dir}\n"
            f"- Is trial: {self.is_trial}\n"
            f"- Job ID: {self.job_id}")


    def log_metrics(self, metrics: Dict[str, float], phase: str, step: Optional[int] = None) -> None:
        """
        Logs metrics to console, file, and optionally WandB.  Also stores
        metrics internally for later analysis/plotting.

        Args:
            metrics: Dictionary of metrics to log.
            phase: Phase of training (e.g., 'train', 'val').
            step: Current training step (global step).
        """

        if step is None:
            step = 0  # Provide a default value

        log_str = f"[{phase}] Step {step}:" + "".join(f" {k}={v:.4f}," for k, v in metrics.items())
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
                file_name = "final_metrics.json" if self.job_id is None else f"job_{self.job_id}_metrics.json"

            file_path = self.metrics_dir / file_name
            with open(file_path, 'a') as f:
                json.dump(metrics, f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Error writing metrics to file: {e}", exc_info=True)

class MetricsManager(BaseManager):
    """Manages metric computation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
      super().__init__(config)


    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        super()._initialize_process_local(config)
        cuda_manager = get_cuda_manager()
        cuda_manager.ensure_initialized()

        self._local.device = None
        self._local.loss_fct = None


    def get_device(self) -> torch.device:
        """Gets the device (CUDA or CPU)."""
        if self._local.device is None:
            cuda_manager = get_cuda_manager()
            if cuda_manager.is_available():
                self._local.device = torch.device('cuda')
            else:
                self._local.device = torch.device('cpu')
        return self._local.device

    def get_loss_fct(self) -> nn.Module:
        """Gets the loss function."""
        if self._local.loss_fct is None:
            self._local.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum').to(self.get_device())
        return self._local.loss_fct

    def compute_accuracy(self, logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> Dict[str, float]:
        """Compute top-k accuracy for masked tokens.

        Args:
            logits: Predicted logits from the model.
            labels: Ground truth labels.
            k: Value for top-k accuracy.

        Returns:
            Dict[str, float]: Dictionary with 'top1' and 'top{k}' accuracy.

        """
        self.ensure_initialized()

        # Flatten logits and labels
        active_preds = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
        active_labels = labels.view(-1)  # [batch_size * seq_len]

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
        _, pred_indices = active_preds.topk(k, dim=1)  # [num_valid_positions, k]

        # Check if correct predictions are in top-k
        correct_k = pred_indices.eq(active_labels.unsqueeze(1).expand_as(pred_indices))

        # Calculate accuracy
        total = active_mask.sum().item()
        top1 = correct_k[:, 0].sum().item() / total  # Corrected: Use .item()
        topk = correct_k.any(dim=1).sum().item() / total  # Corrected: Use .item()

        logger.debug(
            f"Accuracy Stats:\n"
            f"- Total predictions: {total}\n"
            f"- Top-1 correct: {int(top1 * total)}\n"
            f"- Top-{k} correct: {int(topk * total)}"
        )
        return {'top1': top1, f'top{k}': topk}


    def compute_embedding_metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute embedding metrics.

        Args:
          outputs: Model outputs.
          batch: The input batch

        Returns:
          Dict[str, float]: Dictionary of metrics (loss, embedding_loss, ppl, accuracy, top5_accuracy, mask_ratio)
        """
        self.ensure_initialized()
        try:
            logits = outputs['logits']
            labels = batch['labels']

            # Calculate loss *only* on masked tokens (-100 is ignored by CrossEntropyLoss)
            loss_fct = self.get_loss_fct()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Get the number of masked tokens (where label is not -100)
            valid_predictions = (labels.view(-1) != -100).sum().item()
            if valid_predictions == 0:
                logger.warning("No valid predictions found in batch")
                return {  # Return default values
                    'loss': 0.0,
                    'embedding_loss': 0.0,
                    'ppl': 1.0,  # Perplexity is 1 when loss is 0
                    'accuracy': 0.0,
                    'top5_accuracy': 0.0,
                    'mask_ratio': 0.0  # No masked tokens
                }

            # Normalize the loss by the number of *valid* predictions
            normalized_loss = loss / valid_predictions

            with torch.no_grad():
                # Calculate perplexity
                try:
                    ppl = math.exp(normalized_loss.item())  # Use normalized loss
                except OverflowError:
                    ppl = float('inf')
                # Compute accuracy
                accuracy = self.compute_accuracy(logits, labels)['top1']  # Use helper
                top5_accuracy = self.compute_accuracy(logits, labels, k=5)['top5'] # Use helper

                # Get the number of masked tokens
                mask = labels != -100
                total_masked = mask.sum().item()
                total_tokens = labels.numel()  # Total number of tokens in the batch

                metrics = {
                    'loss': normalized_loss.item(),  # Use normalized loss
                    'embedding_loss': normalized_loss.item(), #Same
                    'ppl': ppl,
                    'accuracy': accuracy,
                    'top5_accuracy': top5_accuracy,
                    'mask_ratio': total_masked / total_tokens
                }
                return metrics

        except Exception as e:
            logger.error(f"Error computing embedding metrics: {str(e)}", exc_info=True)
            return {  # Return default values on error
                'loss': float('inf'),
                'embedding_loss': float('inf'),
                'ppl': float('inf'),
                'accuracy': 0.0,
                'top5_accuracy': 0.0,
                'mask_ratio': 0.0
            }



    def compute_classification_metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute classification metrics.

        Args:
            outputs: Model outputs
            batch: The input batch.

        Returns:
            Dict[str, float]: Dictionary with 'loss' and 'accuracy'.

        """

        self.ensure_initialized()

        try:
            logits = outputs['logits']
            labels = batch['labels']

            loss_fct = nn.CrossEntropyLoss()  # Use standard cross-entropy
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            _, preds = torch.max(logits, dim=1)
            correct = (preds == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total if total > 0 else 0.0

            metrics = {
                'loss': loss.item(),
                'accuracy': accuracy
            }
            # You could add more metrics here (F1, precision, recall, etc.)
            return metrics

        except Exception as e:
            logger.error(f"Error computing classification metrics: {str(e)}")
            return {
                'loss': float('inf'),
                'accuracy': 0.0
            }

__all__ = ['MetricsManager', 'MetricsLogger']