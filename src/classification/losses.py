# losses.py
# src/classification/losses.py
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss for classification to focus on hard examples."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        """
        Initialize Focal Loss.

        Args:
            gamma: Focusing parameter.  Higher values focus more on hard examples.
            alpha: Optional weighting factor for each class.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if self.alpha is not None:
            self.alpha = self.alpha.clone().detach()


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss.

        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Focal loss value
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()

def focal_loss_factory(config: Dict[str, Any]) -> FocalLoss:
    """
    Factory function for creating a FocalLoss instance.

    Args:
        config: Configuration dictionary.

    Returns:
        A FocalLoss instance.
    """
    gamma = config.get('focal_loss_gamma', 2.0)  # Default gamma = 2.0
    alpha = config.get('focal_loss_alpha', None)  # Default alpha = None

    # If alpha is provided, it should be a list in the config. Convert to tensor.
    if alpha is not None:
        if not isinstance(alpha, list):
            raise ValueError("focal_loss_alpha must be a list or None")
        alpha = torch.tensor(alpha, dtype=torch.float)

    return FocalLoss(gamma=gamma, alpha=alpha)

__all__ = ['FocalLoss', 'focal_loss_factory']