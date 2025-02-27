# src/common/managers/amp_manager.py
from __future__ import annotations
import torch
import logging
import traceback
import os  # Add this import
from typing import Optional, Dict, Any
from contextlib import contextmanager
from .base_manager import BaseManager
from src.common.managers.cuda_manager import CUDAManager

logger = logging.getLogger(__name__)


class AMPManager(BaseManager):
    """Manager for Automatic Mixed Precision (AMP) training.
    
    This manager handles the GradScaler and automatically sets up AMP
    based on CUDA availability.
    """

    def __init__(self, cuda_manager: CUDAManager, config: Optional[Dict[str, Any]] = None):
        """Initialize the AMP Manager.
        
        Args:
            cuda_manager: The CUDA manager to determine device availability
            config: Application configuration
        """
        self._cuda_manager = cuda_manager
        super().__init__(config)  # Single call to initialize

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local state."""
        try:
            super()._initialize_process_local(config)
            effective_config = config if config is not None else self._config

            if not self._cuda_manager.is_initialized():
                raise RuntimeError("CUDA must be initialized before AMPManager")

            self._use_amp = effective_config.get('training', {}).get('use_amp', False) and self._cuda_manager.is_available()

            if self._use_amp:
                self._local.scaler = torch.cuda.amp.GradScaler()
                logger.info(f"Initialized GradScaler for process {self._local.pid}")
            else:
                self._local.scaler = None
                logger.info("FP16 not enabled, AMP disabled")

            logger.info(f"AMPManager initialized for process {self._local.pid}")

        except Exception as e:
            logger.error(f"Failed to initialize GradScaler: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def is_enabled(self) -> bool:
        """
        Check if AMP is enabled.

        Returns:
            bool: True if AMP is enabled, False otherwise
        """
        self.ensure_initialized()
        return hasattr(self._local, 'scaler') and self._local.scaler is not None

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for AMP.

        Args:
            loss: The loss tensor to scale

        Returns:
            torch.Tensor: Scaled loss tensor
        """
        self.ensure_initialized()
        if not self.is_enabled():
            return loss
        return self._local.scaler.scale(loss)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Perform optimizer step with AMP scaler if enabled.

        Args:
            optimizer: The optimizer to step
        """
        self.ensure_initialized()
        if self.is_enabled():
            self._local.scaler.step(optimizer)
            self._local.scaler.update()
        else:
            optimizer.step()

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Unscale gradients for AMP.

        Args:
            optimizer: The optimizer containing gradients to unscale
        """
        self.ensure_initialized()
        if self.is_enabled():
            self._local.scaler.unscale_(optimizer)

    def backward_step(
        self,
        loss: torch.Tensor,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        grad_norm: Optional[float] = None
    ) -> None:
        """
        Perform backward step, handling gradient scaling and clipping with AMP.

        Args:
            loss: The loss tensor
            model: The model containing parameters
            optimizer: The optimizer
            grad_norm: Optional gradient norm for clipping
        """
        self.ensure_initialized()

        try:
            # Check if loss is None - critical safety check
            if loss is None:
                logger.error("Loss is None in backward_step - skipping backward pass")
                return
                
            # Validate loss is a proper tensor with requires_grad
            if not isinstance(loss, torch.Tensor):
                logger.error(f"Loss is not a tensor (got {type(loss)}) - skipping backward pass")
                return
                
            if not loss.requires_grad:
                logger.warning("Loss tensor doesn't have requires_grad=True - setting it now")
                loss.requires_grad_(True)

            if self.is_enabled():
                scaled_loss = self.scale_loss(loss)
                scaled_loss.backward()

                self.unscale_gradients(optimizer)

                if grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)

                self.step(optimizer)
            else:
                loss.backward()

                if grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)

                optimizer.step()

        except Exception as e:
            logger.error(f"Error in backward step: {str(e)}")
            logger.error(traceback.format_exc())
            # Don't re-raise, just log the error to allow training to continue
            # with the next batch

    @contextmanager
    def autocast(self) -> None:
        """
        Context manager for AMP autocasting.

        Usage:
            with amp_manager.autocast():
                output = model(input)
        """
        self.ensure_initialized()
        if self.is_enabled():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                yield
        else:
            yield

    def cleanup(self) -> None:
        """Clean up AMP manager resources."""
        try:
            if hasattr(self, '_local'):
                if hasattr(self._local, 'scaler'):
                    self._local.scaler = None
                
                # Add a safety check for the pid attribute
                pid = getattr(self._local, 'pid', os.getpid())
                logger.info(f"Cleaned up AMPManager for process {pid}")
                
            super().cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up AMPManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise


def get_amp_manager(cuda_manager: CUDAManager, config: Optional[Dict[str, Any]] = None) -> AMPManager:
    """
    Get an instance of AMPManager.

    Args:
        cuda_manager: An instance of CUDAManager
        config: Optional configuration dictionary

    Returns:
        AMPManager: An instance of AMPManager
    """
    return AMPManager(cuda_manager, config)


__all__ = ['AMPManager']