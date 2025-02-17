# src/common/scheduler/dynamic_scheduler.py
# src/common/scheduler/dynamic_scheduler.py
from __future__ import annotations
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional, List
import math
import logging

logger = logging.getLogger(__name__)

class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine learning rate scheduler with warmup and optional hard restarts.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total number of training steps
            num_cycles: Number of cycles for cosine decay
            min_lr_ratio: Minimum learning rate ratio
            last_epoch: Last epoch (-1 for fresh start)
        """
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rate based on current step."""
        try:
            if self.last_epoch < self.num_warmup_steps:
                # Linear warmup
                return [
                    base_lr * float(self.last_epoch) / float(max(1, self.num_warmup_steps))
                    for base_lr in self.base_lrs
                ]
            else:
                # Cosine decay with optional hard restarts
                progress = float(self.last_epoch - self.num_warmup_steps) / float(
                    max(1, self.num_training_steps - self.num_warmup_steps)
                )

                if progress >= 1.0:
                    return [self.min_lr_ratio * base_lr for base_lr in self.base_lrs]

                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress))
                decayed = (1.0 - self.min_lr_ratio) * cosine_decay + self.min_lr_ratio

                return [base_lr * decayed for base_lr in self.base_lrs]

        except Exception as e:
            logger.error(f"Error computing learning rate: {str(e)}")
            raise

class WarmupLinearScheduler(_LRScheduler):
    """
    Linear learning rate scheduler with warmup.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total number of training steps
            min_lr_ratio: Minimum learning rate ratio
            last_epoch: Last epoch (-1 for fresh start)
        """
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rate based on current step."""
        try:
            if self.last_epoch < self.num_warmup_steps:
                # Linear warmup
                return [
                    base_lr * float(self.last_epoch) / float(max(1, self.num_warmup_steps))
                    for base_lr in self.base_lrs
                ]
            else:
                # Linear decay
                progress = float(self.last_epoch - self.num_warmup_steps) / float(
                    max(1, self.num_training_steps - self.num_warmup_steps)
                )

                if progress >= 1.0:
                    return [self.min_lr_ratio * base_lr for base_lr in self.base_lrs]

                decay = (1.0 - progress) * (1.0 - self.min_lr_ratio) + self.min_lr_ratio

                return [base_lr * decay for base_lr in self.base_lrs]

        except Exception as e:
            logger.error(f"Error computing learning rate: {str(e)}")
            raise

def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr_ratio: float = 0.0,
        num_cycles: float = 0.5
        ) -> Optional[_LRScheduler]:
    """
    Create learning rate scheduler based on configuration.

    Args:
        optimizer: The optimizer to wrap
        config: Configuration dictionary containing scheduler settings

    Returns:
        Learning rate scheduler or None if not configured
    """
    try:

        if scheduler_type == 'cosine':
            return WarmupCosineScheduler(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
                min_lr_ratio=min_lr_ratio
            )
        elif scheduler_type == 'linear':
            return WarmupLinearScheduler(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                min_lr_ratio=min_lr_ratio
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    except Exception as e:
        logger.error(f"Error creating scheduler: {str(e)}")
        raise