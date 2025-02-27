"""Emergency repair utility for broken manager classes."""
import os
import sys
import threading
import logging
import traceback
from typing import Dict, Any, List, Tuple, Optional, Set, Type

import torch

# Import base manager class for type checking
from src.common.managers.base_manager import BaseManager

logger = logging.getLogger(__name__)

def repair_manager(manager: Any, name: str) -> bool:
    """
    Repair a manager that failed initialization or has broken state.
    
    Args:
        manager: The manager instance to repair
        name: The manager name for logging
        
    Returns:
        bool: Whether the repair was successful
    """
    if not isinstance(manager, BaseManager):
        logger.warning(f"{name} is not a BaseManager instance, cannot repair")
        return False
        
    logger.warning(f"Attempting to repair {name} in process {os.getpid()}")
    
    # Step 1: Check if _local exists, create if missing
    if not hasattr(manager, '_local'):
        logger.warning(f"{name} is missing _local attribute, creating it")
        manager._local = threading.local()
    
    # Step 2: Ensure required attributes exist in _local
    required_attributes = ['pid', 'initialized']
    for attr in required_attributes:
        if not hasattr(manager._local, attr):
            if attr == 'pid':
                manager._local.pid = os.getpid()
                logger.info(f"Set {name}._local.pid = {os.getpid()}")
            elif attr == 'initialized':
                manager._local.initialized = True
                logger.info(f"Set {name}._local.initialized = True")
    
    # Step 3: Add manager-specific attributes based on class name
    if "CUDAManager" in name:
        logger.info(f"Adding CUDA-specific attributes to {name}")
        if not hasattr(manager._local, 'device'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            manager._local.device = device
            logger.info(f"Set {name}._local.device = {device}")
            
    if "MetricsManager" in name:
        logger.info(f"Adding metrics-specific attributes to {name}")
        if not hasattr(manager._local, 'device'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            manager._local.device = device
            logger.info(f"Set {name}._local.device = {device}")
        if not hasattr(manager._local, 'metrics'):
            manager._local.metrics = {}
            logger.info(f"Set {name}._local.metrics = {{}}")
        if not hasattr(manager._local, 'loss_fct'):
            try:
                import torch.nn as nn
                manager._local.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                logger.info(f"Created loss function for {name}")
            except Exception as e:
                logger.error(f"Could not create loss function: {e}")
                manager._local.loss_fct = None
    
    if "DataManager" in name:
        logger.info(f"Adding data-specific attributes to {name}")
        if not hasattr(manager._local, 'datasets'):
            manager._local.datasets = {}
            logger.info(f"Set {name}._local.datasets = {{}}")
        if not hasattr(manager._local, 'tokenizer'):
            manager._local.tokenizer = None
            logger.info(f"Set {name}._local.tokenizer = None")
            
    if "TokenizerManager" in name:
        logger.info(f"Adding tokenizer-specific attributes to {name}")
        if not hasattr(manager._local, 'tokenizers'):
            manager._local.tokenizers = {}
            logger.info(f"Set {name}._local.tokenizers = {{}}")
        if not hasattr(manager._local, 'shared_tokenizer'):
            manager._local.shared_tokenizer = None
            logger.info(f"Set {name}._local.shared_tokenizer = None")
            
    if "DirectoryManager" in name:
        logger.info(f"Adding directory-specific attributes to {name}")
        # Add base_dir and other directory attributes if missing
        for dir_attr in ['base_dir', 'output_dir', 'data_dir', 'cache_dir', 'model_dir', 'temp_dir']:
            if not hasattr(manager._local, dir_attr):
                value = os.path.join(os.getcwd(), dir_attr.replace('_dir', ''))
                setattr(manager._local, dir_attr, value)
                logger.info(f"Set {name}._local.{dir_attr} = {value}")
                # Create directory if it doesn't exist
                os.makedirs(value, exist_ok=True)
                
    logger.info(f"Successfully repaired {name} in process {os.getpid()}")
    return True
    
def repair_all_managers() -> Dict[str, bool]:
    """
    Repair all available managers.
    
    Returns:
        Dict[str, bool]: Results of repair attempts
    """
    from src.common.managers import (
        get_cuda_manager, get_directory_manager, get_wandb_manager,
        get_batch_manager, get_parameter_manager, get_metrics_manager,
        get_tokenizer_manager, get_data_manager, get_dataloader_manager
    )
    
    managers = {
        "CUDAManager": get_cuda_manager,
        "DirectoryManager": get_directory_manager,
        "TokenizerManager": get_tokenizer_manager,
        "DataLoaderManager": get_dataloader_manager,
        "DataManager": get_data_manager,
        "BatchManager": get_batch_manager,
        "MetricsManager": get_metrics_manager,
        "ParameterManager": get_parameter_manager,
        "WandbManager": get_wandb_manager
    }
    
    results = {}
    for name, get_func in managers.items():
        try:
            manager = get_func()
            results[name] = repair_manager(manager, name)
        except Exception as e:
            logger.error(f"Error getting or repairing {name}: {e}")
            logger.error(traceback.format_exc())
            results[name] = False
            
    return results
