# src/common/managers/tensor_manager.py
from __future__ import annotations
import torch
import logging
import traceback
from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np

from src.common.managers.base_manager import BaseManager
from src.common.managers.cuda_manager import CUDAManager

logger = logging.getLogger(__name__)

class TensorManager(BaseManager):
    """
    Process-local tensor manager for device placement and memory management.
    
    This manager handles:
    - Tensor creation and device placement
    - Memory pinning and management
    - Random tensor generation
    - Device-specific operations
    """

    def __init__(
        self,
        cuda_manager: CUDAManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize TensorManager.

        Args:
            cuda_manager: Injected CUDAManager instance
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self._cuda_manager = cuda_manager
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
                raise RuntimeError("CUDAManager must be initialized before TensorManager")

            self._local.device = self._cuda_manager.get_device()

            logger.info(
                f"TensorManager initialized for process {self._local.pid} "
                f"using device {self._local.device}"
            )

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
        """
        Create tensor with device placement.

        Args:
            data: Input data to convert to tensor
            device: Optional target device
            dtype: Optional tensor dtype
            requires_grad: Whether tensor requires gradients

        Returns:
            torch.Tensor: Created tensor on specified device
        """
        self.ensure_initialized()
        try:
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)

            target_device = device if device is not None else self._local.device
            data = data.to(device=target_device, dtype=dtype)
            data.requires_grad = requires_grad
            return data

        except Exception as e:
            logger.error(f"Error creating tensor: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_device(self) -> torch.device:
        """
        Get current device.

        Returns:
            torch.device: Current device
        """
        self.ensure_initialized()
        return self._local.device

    def create_cpu_tensor(
        self,
        data: Union[torch.Tensor, List, np.ndarray],
        dtype: Optional[torch.dtype] = None,
        pin_memory: bool = True
    ) -> torch.Tensor:
        """
        Create tensor on CPU.

        Args:
            data: Input data to convert to tensor
            dtype: Optional tensor dtype
            pin_memory: Whether to pin memory for faster GPU transfer

        Returns:
            torch.Tensor: Created tensor on CPU
        """
        self.ensure_initialized()
        try:
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)

            data = data.cpu()
            if dtype is not None:
                data = data.to(dtype=dtype)
            if pin_memory and self._cuda_manager.is_available():
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
        """
        Create random tensor between 0 and 1.

        Args:
            size: Shape of the tensor to create
            device: Optional target device

        Returns:
            torch.Tensor: Random tensor
        """
        self.ensure_initialized()
        try:
            target_device = device if device is not None else self._local.device
            return torch.rand(size, device=target_device)

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
        """
        Create random integer tensor.

        Args:
            low: Lower bound (inclusive)
            high: Upper bound (exclusive)
            size: Shape of the tensor to create
            device: Optional target device
            dtype: Optional tensor dtype

        Returns:
            torch.Tensor: Random integer tensor
        """
        self.ensure_initialized()
        try:
            target_device = device if device is not None else self._local.device
            return torch.randint(
                low,
                high,
                size,
                device=target_device,
                dtype=dtype
            )

        except Exception as e:
            logger.error(f"Error creating random integer tensor: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def clear_memory(self) -> None:
        """Clear CUDA memory."""
        self.ensure_initialized()
        try:
            self._cuda_manager.cleanup()
            logger.debug("Cleared CUDA memory")

        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def cleanup(self) -> None:
        """Clean up tensor manager resources."""
        try:
            self.clear_memory()
            self._local.device = None
            logger.info(f"Cleaned up TensorManager for process {self._local.pid}")
            super().cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up TensorManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise


__all__ = ['TensorManager']
