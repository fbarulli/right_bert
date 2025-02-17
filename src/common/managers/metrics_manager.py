# src/common/managers/metrics_manager.py
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import math

from src.common.managers.base_manager import BaseManager
from src.common.managers import get_cuda_manager

logger = logging.getLogger(__name__)

class MetricsManager(BaseManager):

    def __init__(self):
        super().__init__()

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        super()._initialize_process_local(config)
        cuda_manager = get_cuda_manager()
        cuda_manager.ensure_initialized()

        self._local.device = None
        self._local.loss_fct = None
        self._local.pad_token_id = 0  # You might get this from the tokenizer config

    def get_device(self) -> torch.device:
        if self._local.device is None:
            cuda_manager = get_cuda_manager()
            if cuda_manager.is_available():
                self._local.device = torch.device('cuda')
            else:
                self._local.device = torch.device('cpu')
        return self._local.device

    def get_loss_fct(self) -> nn.Module:
        if self._local.loss_fct is None:
            self._local.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum').to(self.get_device())
        return self._local.loss_fct

    def compute_accuracy(self, logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> Dict[str, float]:
        """Compute top-k accuracy for masked tokens."""
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
        """Compute embedding metrics."""
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

    def compute_classification_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute classification metrics."""

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
__all__ = ['MetricsManager']