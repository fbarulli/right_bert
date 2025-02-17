# src/common/managers/tensor_manager.py
import torch
import logging
import traceback
from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np

from src.common.managers.base_manager import BaseManager
from src.common.managers import get_cuda_manager #Correct import
from src.common.cuda_utils import (
    is_cuda_available,
    clear_cuda_memory
)

logger = logging.getLogger(__name__)

class TensorManager(BaseManager):
    """Process-local tensor manager for device placement and memory management."""

    def __init__(self):
        super().__init__()


    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes."""
        try:
            super()._initialize_process_local(config)
            if not get_cuda_manager().is_initialized():
                raise RuntimeError("CUDA must be initialized before TensorManager")

            self._local.device = None
            logger.info(f"TensorManager initialized for process {self._local.pid}")

        except Exception as e:
            logger.error(f"Failed to initialize TensorManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_tensor(
        self,
        data: Union[torch.Tensor, List, np.ndarray],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """Create tensor with proper device placement."""
        self.ensure_initialized()
        try:
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)

            if device is None:
                device = self.get_device()

            data = data.to(device=device, dtype=dtype)
            data.requires_grad = requires_grad

            return data

        except Exception as e:
            logger.error(f"Error creating tensor: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_device(self) -> torch.device:
        """Get current device."""
        self.ensure_initialized()
        if self._local.device is None:
            if get_cuda_manager().is_available():
                self._local.device = torch.device('cuda')
            else:
                self._local.device = torch.device('cpu')
        return self._local.device

    def create_cpu_tensor(
        self,
        data: Union[torch.Tensor, List, np.ndarray],
        dtype: Optional[torch.dtype] = None,
        pin_memory: bool = True
    ) -> torch.Tensor:
        """Create tensor on CPU with optional pinned memory."""
        self.ensure_initialized()
        try:
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)

            data = data.cpu()
            if dtype is not None:
                data = data.to(dtype=dtype)

            if pin_memory:
                if get_cuda_manager().is_available():
                    data = data.pin_memory()

            return data

        except Exception as e:
            logger.error(f"Error creating CPU tensor: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_random(
        self,
        size: Union[Tuple[int, ...], List[int]],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Create random tensor between 0 and 1."""
        self.ensure_initialized()
        try:
            if device is None:
                device = self.get_device()
            return torch.rand(size, device=device)
        except Exception as e:
            logger.error(f"Error creating random tensor: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_random_int(
        self,
        low: int,
        high: int,
        size: Union[Tuple[int, ...], List[int]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Create random integer tensor between low and high."""
        self.ensure_initialized()
        try:
            if device is None:
                device = self.get_device()
            return torch.randint(low, high, size, device=device, dtype=dtype)
        except Exception as e:
            logger.error(f"Error creating random int tensor: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def clear_memory(self) -> None:
        """Clear CUDA memory cache."""
        self.ensure_initialized()
        if is_cuda_available():
            clear_cuda_memory()

__all__ = ['TensorManager']