
# src/common/managers/batch_manager.py
from __future__ import annotations
import torch
import logging
import traceback
from typing import Dict, Any, Optional, Union, Set
from torch.utils.data import DataLoader

from src.common.managers.base_manager import BaseManager
from src.common.managers.cuda_manager import CUDAManager
from src.common.managers.tensor_manager import TensorManager

logger = logging.getLogger(__name__)

class BatchManager(BaseManager):
    """Process-local batch manager for device placement and memory management."""

    # Define model fields as a class constant
    MODEL_FIELDS: Set[str] = {
        'input_ids',
        'attention_mask',
        'token_type_ids',
        'position_ids',
        'labels'
    }

    def __init__(
        self,
        cuda_manager: CUDAManager,
        tensor_manager: TensorManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize BatchManager.

        Args:
            cuda_manager: Injected CUDAManager instance
            tensor_manager: Injected TensorManager instance
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self._cuda_manager = cuda_manager
        self._tensor_manager = tensor_manager
        self._local.device = None

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local attributes.

        Args:
            config: Optional configuration dictionary that overrides the one from constructor
        """
        try:
            super()._initialize_process_local(config)

            if not self._cuda_manager.is_initialized():
                raise RuntimeError("CUDA must be initialized before BatchManager")
            if not self._tensor_manager.is_initialized():
                raise RuntimeError("TensorManager must be initialized before BatchManager")

            self._local.device = self._cuda_manager.get_device()
            logger.info(f"BatchManager initialized for process {self._local.pid} with device {self._local.device}")

        except Exception as e:
            logger.error(f"Failed to initialize BatchManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def prepare_batch(
        self,
        batch: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch by moving tensors to the specified device.

        Args:
            batch: Dictionary containing batch tensors
            device: Optional target device. If None, uses default CUDA device

        Returns:
            Dict[str, torch.Tensor]: Batch with tensors moved to target device
        """
        self.ensure_initialized()
        try:
            target_device = device if device is not None else self._local.device

            return {
                k: v.to(device=target_device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k in self.MODEL_FIELDS
            }

        except Exception as e:
            logger.error(f"Error preparing batch: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_batch_size(self, batch: Union[Dict[str, torch.Tensor], DataLoader]) -> int:
        """
        Get batch size from batch dict or dataloader.

        Args:
            batch: Either a dictionary of tensors or a DataLoader

        Returns:
            int: The batch size

        Raises:
            ValueError: If no tensors found in batch dict
            TypeError: If batch is neither dict nor DataLoader
        """
        self.ensure_initialized()
        try:
            if isinstance(batch, dict):
                for v in batch.values():
                    if isinstance(v, torch.Tensor):
                        return v.size(0)
                raise ValueError("No tensors found in batch dict")
            elif isinstance(batch, DataLoader):
                return batch.batch_size
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

        except Exception as e:
            logger.error(f"Error getting batch size: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def cleanup(self) -> None:
        """Clean up batch manager resources."""
        try:
            if hasattr(self._local, 'device'):
                self._local.device = None
            super().cleanup()
            logger.info(f"Cleaned up BatchManager for process {self._local.pid}")
        except Exception as e:
            logger.error(f"Error cleaning up BatchManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise


__all__ = ['BatchManager']