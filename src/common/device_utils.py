"""Utilities for handling device consistency in PyTorch."""
import torch
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

def check_tensor_device_consistency(tensors: Dict[str, torch.Tensor]) -> bool:
    """
    Check if all tensors are on the same device.
    
    Args:
        tensors: Dictionary of tensors to check
        
    Returns:
        bool: Whether all tensors are on the same device
    """
    devices = set()
    for name, tensor in tensors.items():
        if torch.is_tensor(tensor):
            devices.add(tensor.device)
            
    return len(devices) <= 1

def ensure_tensors_on_device(
    tensors: Dict[str, Any], 
    device: torch.device
) -> Dict[str, Any]:
    """
    Ensure all tensors in a dictionary are on the specified device.
    
    Args:
        tensors: Dictionary of tensors and other values
        device: Target device
        
    Returns:
        Dict[str, Any]: Dictionary with tensors moved to device
    """
    result = {}
    for name, value in tensors.items():
        if torch.is_tensor(value):
            if value.device != device:
                try:
                    result[name] = value.to(device)
                except Exception as e:
                    logger.error(f"Error moving tensor {name} to {device}: {e}")
                    # Keep original tensor as fallback
                    result[name] = value
            else:
                result[name] = value
        elif isinstance(value, list) and all(torch.is_tensor(t) for t in value):
            # Handle list of tensors
            try:
                result[name] = [t.to(device) if t.device != device else t for t in value]
            except Exception as e:
                logger.error(f"Error moving tensor list {name} to {device}: {e}")
                result[name] = value
        else:
            # Not a tensor, keep as is
            result[name] = value
            
    return result

def get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Get the device of a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        torch.device: Device where the model is located
    """
    try:
        # First try the model.device attribute if it exists
        if hasattr(model, 'device'):
            return model.device
            
        # Next try to get device from parameters
        try:
            return next(model.parameters()).device
        except StopIteration:
            # Model has no parameters
            pass
            
        # Try to get device from buffers
        try:
            return next(model.buffers()).device
        except StopIteration:
            # Model has no buffers
            pass
            
        # Default to CPU
        logger.warning(f"Could not determine device for model {type(model).__name__}, defaulting to CPU")
        return torch.device('cpu')
        
    except Exception as e:
        logger.error(f"Error getting model device: {e}")
        return torch.device('cpu')  # Default fallback
