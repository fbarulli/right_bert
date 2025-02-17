# cuda_utils.py
# src/common/cuda_utils.py
import torch
import gc
import logging
from typing import Dict, Any, Optional


logger = logging.getLogger(__name__)

def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()

def get_cuda_device(device_id: Optional[int] = None) -> torch.device:
    """Get the current CUDA device."""
    if is_cuda_available():
        return torch.device(f'cuda:{device_id if device_id is not None else 0}')
    return torch.device('cpu')

def get_cuda_device_count() -> int:
    """Get number of available CUDA devices."""
    return torch.cuda.device_count() if is_cuda_available() else 0

def get_cuda_device_properties(device_id: Optional[int] = None) -> Dict[str, Any]:
    """Get properties of CUDA device."""
    if not is_cuda_available():
        return {}

    device = device_id if device_id is not None else 0
    props = torch.cuda.get_device_properties(device)
    return {
        'name': props.name,
        'total_memory': props.total_memory / 1024**3,  # GB
        'major': props.major,
        'minor': props.minor,
        'multi_processor_count': props.multi_processor_count
    }

def clear_cuda_memory() -> None:
    """Clear CUDA memory cache."""
    if is_cuda_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def get_cuda_memory_stats() -> Dict[str, float]:
    """Get current CUDA memory statistics."""
    if not is_cuda_available():
        return {}

    return {
        'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
        'cached': torch.cuda.memory_reserved() / 1024**3,  # GB
        'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
    }
def reset_cuda_stats() -> None:
    """Reset CUDA memory statistics."""
    if is_cuda_available():
        torch.cuda.reset_peak_memory_stats()
        if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
            torch.cuda.reset_accumulated_memory_stats()

__all__ = [
    'is_cuda_available',
    'get_cuda_device',
    'get_cuda_device_count',
    'get_cuda_device_properties',
    'clear_cuda_memory',
    'get_cuda_memory_stats',
    'reset_cuda_stats'
]