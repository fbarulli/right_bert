"""CUDA utility functions to manage device placement."""
import torch
import gc
import logging
import os
from typing import Dict, Any, Optional, Union

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
        'total_memory': props.total_memory / 1024**3,
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
        'allocated': torch.cuda.memory_allocated() / 1024**3,
        'cached': torch.cuda.memory_reserved() / 1024**3,
        'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
    }

def reset_cuda_stats() -> None:
    """Reset CUDA memory statistics."""
    if is_cuda_available():
        torch.cuda.reset_peak_memory_stats()
        if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
            torch.cuda.reset_accumulated_memory_stats()

def get_optimal_device() -> torch.device:
    """
    Get the optimal device for model training.
    
    Returns:
        torch.device: The best available device (CUDA or CPU)
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available, using CPU")
        return torch.device("cpu")
        
    # Get device count
    device_count = torch.cuda.device_count()
    logger.info(f"Found {device_count} CUDA device(s)")
    
    if device_count == 0:
        logger.warning("No CUDA devices found despite torch.cuda.is_available() being True")
        return torch.device("cpu")
        
    # If only one device, use it
    if device_count == 1:
        return torch.device("cuda:0")
        
    # If multiple devices, select the one with most free memory
    try:
        free_memory = []
        for i in range(device_count):
            # torch.cuda.get_device_properties(i).total_memory returns total memory in bytes
            # torch.cuda.memory_reserved(i) returns current reserved memory in bytes
            # torch.cuda.memory_allocated(i) returns current allocated memory in bytes
            reserved = torch.cuda.memory_reserved(i)
            allocated = torch.cuda.memory_allocated(i) 
            free = torch.cuda.get_device_properties(i).total_memory - reserved
            free_memory.append((i, free))
            logger.info(f"Device {i}: {free / 1024**3:.2f} GB free out of {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            
        # Select device with most free memory
        best_device_idx = max(free_memory, key=lambda x: x[1])[0]
        logger.info(f"Selected device {best_device_idx} with most free memory")
        return torch.device(f"cuda:{best_device_idx}")
        
    except Exception as e:
        logger.error(f"Error selecting optimal CUDA device: {e}")
        # Default to cuda:0
        return torch.device("cuda:0")

def ensure_model_on_device(model: torch.nn.Module, device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    Ensure a model is on the specified device (or best available device).
    
    Args:
        model: PyTorch model
        device: Optional device to move model to
        
    Returns:
        torch.nn.Module: Model on the specified device
    """
    if device is None:
        device = get_optimal_device()
        
    logger.info(f"Ensuring model is on device: {device}")
    
    # Check current device
    current_device = next(model.parameters()).device
    
    if current_device == device:
        logger.info("Model is already on the correct device")
        return model
        
    # Move model to device
    logger.info(f"Moving model from {current_device} to {device}")
    model = model.to(device)
    
    # Verify move was successful
    new_device = next(model.parameters()).device
    if new_device != device:
        logger.error(f"Failed to move model to {device}, still on {new_device}")
        
    return model

__all__ = [
    'is_cuda_available',
    'get_cuda_device',
    'get_cuda_device_count',
    'get_cuda_device_properties',
    'clear_cuda_memory',
    'get_cuda_memory_stats',
    'reset_cuda_stats',
    'get_optimal_device',
    'ensure_model_on_device'
]