# src/embedding/losses.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import traceback
from typing import Dict, Optional
import math

from src.common.managers import get_cuda_manager, get_tensor_manager

logger = logging.getLogger(__name__)

class InfoNCELoss(nn.Module):
    """InfoNCE loss function."""

    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = 'mean',
        contrast_mode: str = 'all',
        chunk_size: int = 256
    ) -> None:
        """
        Initialize InfoNCE Loss.

        Args:
            temperature: Temperature for scaling similarities.
            reduction: Reduction method ('mean' or 'sum').
            contrast_mode: Contrast mode ('all' or 'one').  Currently only 'all' is implemented.
            chunk_size: Size of chunks for similarity matrix computation.
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.contrast_mode = contrast_mode  # Added for future flexibility
        self.chunk_size = chunk_size
        if self.contrast_mode != 'all':
            raise NotImplementedError("Only contrast_mode='all' is currently supported.")
        logger.info(
            f"InfoNCE Loss initialized with temperature={temperature}, "
            f"reduction={reduction}, contrast_mode={contrast_mode}"
        )

    def compute_similarity_chunk(
        self,
        features: torch.Tensor,
        chunk_start: int,
        chunk_size: int
    ) -> torch.Tensor:
        """Compute similarity between a chunk and all features."""
        try:
            chunk_end = min(chunk_start + chunk_size, features.size(0))
            chunk_features = features[chunk_start:chunk_end]
            sim_chunk = torch.matmul(chunk_features, features.T)
            return sim_chunk

        except Exception as e:
            logger.error(f"Error computing similarity chunk: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute InfoNCE loss."""
        logger.debug("Computing InfoNCE loss")
        try:
            device = features.device
            if labels is not None:
                labels = labels.to(device)
            if mask is not None:
                mask = mask.to(device)

            features = F.normalize(features, dim=1)
            batch_size = features.size(0)

            total_loss = 0.0
            total_pairs = 0

            for i in range(0, batch_size, self.chunk_size):
                chunk_start = i
                chunk_end = min(i + self.chunk_size, batch_size)
                chunk_size = chunk_end - chunk_start

                chunk_features = features[chunk_start:chunk_end]
                chunk_labels = labels[chunk_start:chunk_end] if labels is not None else None

                try:
                    chunk_features = torch.clamp(chunk_features, min=-1e3, max=1e3)
                    features_clipped = torch.clamp(features, min=-1e3, max=1e3)
                    sim_chunk = torch.matmul(chunk_features, features_clipped.T)

                    temperature = max(self.temperature, 1e-4)
                    sim_chunk = sim_chunk / temperature

                    chunk_mask_self = torch.ones_like(sim_chunk, dtype=torch.bool, device=device)
                    chunk_mask_self[:, chunk_start:chunk_end].fill_diagonal_(False)

                    if chunk_labels is not None:
                        chunk_labels = chunk_labels.contiguous().view(-1, 1).to(device)
                        chunk_mask_pos = chunk_labels == labels.view(1, -1).to(device)
                        chunk_mask_pos = chunk_mask_pos & chunk_mask_self
                        if not chunk_mask_pos.any():
                            chunk_mask_pos = chunk_mask_self
                    else:
                        chunk_mask_pos = chunk_mask_self

                    sim_max, _ = torch.max(sim_chunk, dim=1, keepdim=True)
                    sim_chunk = sim_chunk - sim_max.detach()
                    sim_chunk = torch.clamp(sim_chunk, min=-1e3, max=1e3)

                    exp_sim = torch.exp(sim_chunk)
                    exp_sim = torch.clamp(exp_sim, min=1e-8)
                    exp_sim = exp_sim * chunk_mask_self
                    log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

                    log_prob = sim_chunk - log_sum_exp
                    log_prob = torch.clamp(log_prob, min=-1e3)

                    pos_pairs = chunk_mask_pos.sum(1)
                    chunk_loss = -(chunk_mask_pos * log_prob).sum(1)
                    valid_pairs = pos_pairs > 0
                    if valid_pairs.any():
                        chunk_loss = chunk_loss[valid_pairs] / pos_pairs[valid_pairs]
                    else:
                        chunk_loss = torch.zeros(1, device=device)
                except Exception as e:
                    logger.error(f"Error in chunk computation: {str(e)}")
                    chunk_loss = torch.zeros(1, device=device)

                total_loss += chunk_loss.sum()
                total_pairs += (pos_pairs > 0).sum()

            mean_loss = total_loss / (total_pairs + 1e-8)

            if self.reduction == 'mean':
                loss = mean_loss
            elif self.reduction == 'sum':
                loss = mean_loss * total_pairs
            else:
                loss = mean_loss

            return {
                'loss': loss,
                'num_pairs': total_pairs,
                'mean_loss': mean_loss
            }

        except Exception as e:
            logger.error(f"Error computing InfoNCE loss: {str(e)}")
            logger.error(traceback.format_exc())
            raise

def info_nce_loss_factory(config: Dict[str, Any]) -> InfoNCELoss:
    """
    Factory function for creating an InfoNCELoss instance.

    Args:
        config: Configuration dictionary with 'temperature' and 'reduction' keys.

    Returns:
        An instance of InfoNCELoss.
    """
    logger.info("Creating InfoNCELoss with factory")
    try:
        temperature = config['model']['loss_temperature']
        reduction = config['model'].get('loss_reduction', 'mean') # Provide default
        contrast_mode = config['model'].get('contrast_mode', 'all')
        chunk_size = config['model'].get('chunk_size', 256)

        loss = InfoNCELoss(temperature=temperature, reduction=reduction, contrast_mode=contrast_mode, chunk_size=chunk_size)
        logger.info("InfoNCELoss created successfully")
        return loss
    except Exception as e:
        logger.error(f"Error creating InfoNCELoss: {str(e)}")
        logger.error(traceback.format_exc())
        raise

__all__ = ['InfoNCELoss', 'info_nce_loss_factory']