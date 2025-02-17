# src/common/managers/amp_manager.py
from __future__ import annotations
import torch
import logging
import traceback
from typing import Optional, Dict, Any
from contextlib import contextmanager
from .base_manager import BaseManager

logger = logging.getLogger(__name__)

class AMPManager(BaseManager):

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        try:
            super()._initialize_process_local(config)

            from src.common.managers import get_cuda_manager
            cuda_manager = get_cuda_manager()

            if not cuda_manager.is_initialized():
                raise RuntimeError("CUDA must be initialized before AMPManager")

            if cuda_manager.is_available():
                training_config = self.get_config_section(config, 'training')
                if training_config['fp16']:
                    self._local.scaler = torch.cuda.amp.GradScaler()
                    logger.info(f"Initialized GradScaler for process {self._local.pid}")
                else:
                    self._local.scaler = None
                    logger.info("FP16 not enabled, AMP disabled")
            else:
                self._local.scaler = None
                logger.warning("CUDA not available, AMP disabled")

            logger.info(f"AMPManager initialized for process {self._local.pid}")

        except Exception as e:
            logger.error(f"Failed to initialize GradScaler: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def is_enabled(self) -> bool:
        self.ensure_initialized()
        return hasattr(self._local, 'scaler') and self._local.scaler is not None

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        self.ensure_initialized()
        if not self.is_enabled():
            return loss
        return self._local.scaler.scale(loss)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.ensure_initialized()
        if self.is_enabled():
            self._local.scaler.step(optimizer)
            self._local.scaler.update()
        else:
            optimizer.step()

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
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

        self.ensure_initialized()

        try:
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
            raise

    @contextmanager
    def autocast(self) -> None:
        self.ensure_initialized()
        if self.is_enabled():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                yield
        else:
            yield

__all__ = ['AMPManager']