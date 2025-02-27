"""Utilities for handling model device placement and recovery."""
import os
import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, Tuple

logger = logging.getLogger(__name__)

def ensure_model_device(
    model: nn.Module, 
    target_device: Optional[torch.device] = None,
    config: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Ensure model is on the target device, with fallbacks.
    
    Args:
        model: The model to check/move
        target_device: Target device (if None, will determine best device)
        config: Optional configuration dict
        
    Returns:
        nn.Module: Model on the correct device
    """
    try:
        # Determine target device if not specified
        if target_device is None:
            if torch.cuda.is_available():
                target_device = torch.device('cuda')
            else:
                target_device = torch.device('cpu')
                
        # Get current device of model
        try:
            current_device = next(model.parameters()).device
        except StopIteration:
            # Model has no parameters
            logger.warning("Model has no parameters, cannot determine device")
            return model
            
        logger.info(f"Model current device: {current_device}, target: {target_device}")
        
        # If already on target device, return
        if current_device == target_device:
            return model
            
        # Try to move model to target device
        try:
            model = model.to(target_device)
            new_device = next(model.parameters()).device
            logger.info(f"Moved model from {current_device} to {new_device}")
            if new_device != target_device:
                logger.warning(f"Model still not on target device after move")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"CUDA OOM when moving model: {e}")
                # Try to free memory
                torch.cuda.empty_cache()
                # If target was CUDA, fall back to CPU
                if target_device.type == 'cuda':
                    logger.warning("Falling back to CPU due to CUDA OOM")
                    model = model.cpu()
            else:
                logger.error(f"Error moving model to device: {e}")
                # Still try to ensure model is on a usable device
                if current_device.type != 'cpu' and target_device.type != 'cpu':
                    logger.warning("Falling back to CPU due to device error")
                    model = model.cpu()
        
        # Final verification
        try:
            final_device = next(model.parameters()).device
            logger.info(f"Final model device: {final_device}")
        except:
            logger.warning("Could not determine final model device")
            
        return model
    
    except Exception as e:
        logger.error(f"Critical error in ensure_model_device: {e}")
        # Last resort - don't modify the model to avoid further errors
        return model
        
def diagnose_model_device_issues(model: nn.Module) -> Dict[str, Any]:
    """
    Diagnose model device placement issues.
    
    Args:
        model: The model to diagnose
        
    Returns:
        Dict[str, Any]: Diagnostic information
    """
    info = {
        'has_parameters': True,
        'uniform_device': True,
        'cuda_available': torch.cuda.is_available(),
        'devices': set(),
        'parameter_count': 0,
        'total_size_mb': 0
    }
    
    try:
        # Check if model has parameters
        params = list(model.parameters())
        if not params:
            info['has_parameters'] = False
            return info
            
        # Check parameter devices
        devices = set()
        total_params = 0
        total_size_bytes = 0
        
        for param in params:
            devices.add(str(param.device))
            total_params += 1
            total_size_bytes += param.numel() * param.element_size()
            
        info['devices'] = devices
        info['uniform_device'] = len(devices) == 1
        info['parameter_count'] = total_params
        info['total_size_mb'] = total_size_bytes / (1024 * 1024)
        
        return info
        
    except Exception as e:
        logger.error(f"Error in diagnose_model_device_issues: {e}")
        return {**info, 'error': str(e)}
        
def get_available_device(required_memory_mb: Optional[int] = None) -> torch.device:
    """
    Get the best available device with optional memory requirements.
    
    Args:
        required_memory_mb: Optional minimum required memory in MB
        
    Returns:
        torch.device: Best available device 
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')
        
    if required_memory_mb is None:
        return torch.device('cuda')
        
    # Check memory requirements
    device_count = torch.cuda.device_count()
    best_device = None
    max_free_memory = 0
    
    for i in range(device_count):
        try:
            # Get memory info
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            
            # Calculate free memory
            total = torch.cuda.get_device_properties(i).total_memory
            reserved = torch.cuda.memory_reserved(i)
            allocated = torch.cuda.memory_allocated(i)
            free = (total - reserved) / 1024**2  # Convert to MB
            
            logger.info(f"Device {i}: {free:.2f} MB free")
            
            if free > max_free_memory:
                max_free_memory = free
                best_device = i
        except Exception as e:
            logger.error(f"Error checking device {i}: {e}")
            
    if best_device is not None and max_free_memory >= required_memory_mb:
        return torch.device(f'cuda:{best_device}')
    else:
        if max_free_memory < required_memory_mb:
            logger.warning(f"No CUDA device with {required_memory_mb} MB free memory found")
        return torch.device('cpu')
