# src/common/managers/tensor_manager.py (FINAL CORRECTED)
import torch
import logging
import traceback
from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np

from src.common.managers.base_manager import BaseManager
# DO NOT import get_cuda_manager here

logger = logging.getLogger(__name__)

class TensorManager(BaseManager):
    """Process-local tensor manager for device placement and memory management."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TensorManager."""
        super().__init__(config) # Initialize Base

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CUDA and set device."""
        super()._initialize_process_local(config)
        # NOW we can safely import and use get_cuda_manager
        from src.common.managers import get_cuda_manager
        cuda_manager = get_cuda_manager()
        cuda_manager.ensure_initialized()
        self.device = cuda_manager.get_device() # Store device


    def create_tensor(self, data: Union[torch.Tensor, List, np.ndarray], device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, requires_grad: bool = False) -> torch.Tensor:
        """Create tensor with device placement."""
        try:
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)

            if device is None:
                from src.common.managers import get_cuda_manager #DELAYED IMPORT
                device = get_cuda_manager().get_device()


            data = data.to(device=device, dtype=dtype)
            data.requires_grad = requires_grad
            return data

        except Exception as e:
            logger.error(f"Error creating tensor: {str(e)}")
            raise

    def get_device(self) -> torch.device:
        """Get current device."""
        if self.device is None:
             from src.common.managers import get_cuda_manager #DELAYED IMPORT
             self.device = get_cuda_manager().get_device()
        return self.device

    def create_cpu_tensor(self, data: Union[torch.Tensor, List, np.ndarray], dtype: Optional[torch.dtype] = None, pin_memory: bool = True) -> torch.Tensor:
        """Create tensor on CPU."""
        try:
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)

            data = data.cpu()
            if dtype is not None:
                data = data.to(dtype=dtype)
            if pin_memory:
                from src.common.managers import get_cuda_manager #DELAYED IMPORT
                if get_cuda_manager().is_available():
                    data = data.pin_memory()
            return data

        except Exception as e:
            logger.error(f"Error creating CPU tensor: {str(e)}")
            raise

    def create_random(self, size: Union[Tuple[int, ...], List[int]], device: Optional[torch.device] = None) -> torch.Tensor:
        """Create random tensor between 0 and 1."""
        if device is None:
            from src.common.managers import get_cuda_manager #DELAYED IMPORT
            device = get_cuda_manager().get_device()
        return torch.rand(size, device=device)

    def create_random_int(self, low: int, high: int, size: Union[Tuple[int, ...], List[int]], device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Create random integer tensor."""
        if device is None:
            from src.common.managers import get_cuda_manager #DELAYED IMPORT
            device = get_cuda_manager().get_device()
        return torch.randint(low, high, size, device=device, dtype=dtype)

    def clear_memory(self) -> None:
        """Clear CUDA memory."""
        from src.common.managers import get_cuda_manager #DELAYED IMPORT
        cuda_manager = get_cuda_manager()
        cuda_manager.cleanup()

__all__ = ['TensorManager']