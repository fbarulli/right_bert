# src/common/managers/metrics_manager.py
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import math

from src.common.managers import get_cuda_manager
from .base_manager import BaseManager

logger = logging.getLogger(__name__)

class MetricsManager(BaseManager):
    """Manager for computing and tracking metrics."""
    def __init__(self):
        super().__init__()

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes."""
        super()._initialize_process_local(config)

        cuda_manager = get_cuda_manager()
        cuda_manager.ensure_initialized()

        self._local.device = None
        self._local.loss_fct = None
        self._local.pad_token_id = 0

    def get_device(self) -> torch.device:
        """Get current device."""
        self.ensure_initialized()
        if self._local.device is None:
            cuda_manager = get_cuda_manager()
            if cuda_manager.is_available():
                self._local.device = torch.device('cuda')
            else:
                self._local.device = torch.device('cpu')
        return self._local.device

    def get_loss_fct(self) -> nn.Module:
        """Get or create loss function."""
        self.ensure_initialized()
        if self._local.loss_fct is None:
            self._local.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum').to(self.get_device())
        return self._local.loss_fct

    def compute_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        k: int = 1
    ) -> Dict[str, float]:
        """Compute top-k accuracy for masked tokens only."""
        self.ensure_initialized()

        active_preds = logits.view(-1, logits.size(-1))
        active_labels = labels.view(-1)
        active_mask = active_labels != -100

        if not active_mask.any():
            logger.warning("No valid positions for accuracy calculation")
            return {'top1': 0.0, f'top{k}': 0.0}

        active_preds = active_preds[active_mask]
        active_labels = active_labels[active_mask]

        _, pred_indices = active_preds.topk(k, dim=1)
        correct_k = pred_indices.eq(active_labels.unsqueeze(1).expand_as(pred_indices))

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

    def compute_embedding_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute embedding metrics with proper loss normalization."""

        self.ensure_initialized()

        try:
            logits = outputs['logits']
            labels = batch['labels']
            input_ids = batch['input_ids']

            mask = labels != -100
            total_masked = mask.sum().item()
            total_tokens = labels.numel()

            if total_masked == 0:
                logger.warning("No masked tokens found in batch")
                return {
                    'loss': 0.0,
                    'embedding_loss': 0.0,
                    'ppl': 1.0,
                    'accuracy': 0.0,
                    'top5_accuracy': 0.0,
                    'mask_ratio': 0.0
                }

            logits_view = logits.view(-1, logits.size(-1))
            labels_view = labels.view(-1)

            loss = self.get_loss_fct()(logits_view, labels_view)

            valid_predictions = (labels_view != -100).sum().item()
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

            normalized_loss = loss / valid_predictions

            predictions = logits_view.argmax(dim=-1)

            valid_mask = labels_view != -100
            if valid_mask.any():
                correct = (predictions[valid_mask] == labels_view[valid_mask]).float().sum()
                accuracy = (correct / valid_mask.sum()).item()
            else:
                accuracy = 0.0

            _, top_k = logits_view.topk(5, dim=-1)
            correct_k = top_k.eq(labels_view.unsqueeze(-1).expand_as(top_k))

            correct_k = correct_k & valid_mask.unsqueeze(-1)
            top5_accuracy = correct_k.any(dim=-1).float().sum().item() / valid_mask.sum().item()

            try:
                ppl = math.exp(normalized_loss.item())
                logger.debug(f"Calculated perplexity: exp({normalized_loss.item()}) = {ppl:.2f}")
            except OverflowError:
                logger.warning(f"Overflow computing perplexity for loss {normalized_loss.item()}")
                ppl = float('inf')

            metrics = {
                'loss': normalized_loss.item(),
                'embedding_loss': normalized_loss.item(),  # Same as normalized loss
                'ppl': ppl,
                'accuracy': accuracy,
                'top5_accuracy': top5_accuracy,
                'mask_ratio': total_masked / total_tokens
            }

            logger.debug(
                f"Embedding Metrics:\n"
                f"- Total tokens: {total_tokens}\n"
                f"- Masked tokens: {total_masked}\n"
                f"- Mask ratio: {metrics['mask_ratio']:.2%}\n"
                f"- Raw loss: {loss.item():.4f}\n"
                f"- Normalized loss: {normalized_loss.item():.4f}\n"
                f"- Perplexity: {ppl:.2f}\n"
                f"- Masked accuracy: {accuracy:.4%}\n"
                f"- Top-5 Accuracy: {top5_accuracy:.4%}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Error computing embedding metrics: {str(e)}")
            logger.error(f"Output type: {type(outputs)}")
            logger.error(f"Output keys: {outputs.keys() if isinstance(outputs, dict) else 'Not a dict'}")
            logger.error(f"Batch type: {type(batch)}")
            logger.error(f"Batch keys: {batch.keys() if isinstance(batch, dict) else 'Not a dict'}")
            import traceback
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
        """Compute classification metrics."""

        self.ensure_initialized()
        try:
            logits = outputs['logits']
            labels = batch['labels']
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))


            _, preds = torch.max(logits, dim=1)
            correct = (preds == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total if total > 0 else 0.0

            metrics = {
                'loss': loss.item(),
                'accuracy': accuracy
            }
            return metrics

        except Exception as e:
            logger.error(f"Error computing classification metrics: {str(e)}")
            return {
                'loss': float('inf'),
                'accuracy': 0.0
            }
__all__ = ['MetricsManager']