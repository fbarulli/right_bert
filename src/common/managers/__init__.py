# src/common/managers/__init__.py
from __future__ import annotations
"""Manager access layer using factory pattern."""
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .manager_factory import ManagerFactory

logger = logging.getLogger(__name__)

_factory: Optional[ManagerFactory] = None

def initialize_factory(config: Dict[str, Any]) -> None:
    """Initialize the manager factory with configuration."""
    global _factory
    if _factory is not None:
        logger.warning("Manager factory already initialized")
        return
    _factory = ManagerFactory(config)
    logger.info("Initialized manager factory")

def get_factory() -> ManagerFactory:
    """Get the manager factory instance."""
    if _factory is None:
        raise RuntimeError("Manager factory not initialized. Call initialize_factory first.")
    return _factory

def get_amp_manager():
    """Get the AMP manager instance."""
    return get_factory().get_amp_manager()

def get_batch_manager():
    """Get the batch manager instance."""
    return get_factory().get_batch_manager()

def get_cuda_manager():
    """Get the CUDA manager instance."""
    return get_factory().get_cuda_manager()

def get_data_manager():
    """Get the data manager instance."""
    return get_factory().get_data_manager()

def get_dataloader_manager():
    """Get the dataloader manager instance."""
    return get_factory().get_dataloader_manager()

def get_directory_manager():
    """Get the directory manager instance."""
    return get_factory().get_directory_manager()

def get_metrics_manager():
    """Get the metrics manager instance."""
    return get_factory().get_metrics_manager()

def get_model_manager():
    """Get the model manager instance."""
    return get_factory().get_model_manager()

def get_parameter_manager():
    """Get the parameter manager instance."""
    return get_factory().get_parameter_manager()

def get_resource_manager():
    """Get the resource manager instance."""
    return get_factory().get_resource_manager()

def get_storage_manager():
    """Get the storage manager instance."""
    return get_factory().get_storage_manager()

def get_tensor_manager():
    """Get the tensor manager instance."""
    return get_factory().get_tensor_manager()

def get_tokenizer_manager():
    """Get the tokenizer manager instance."""
    return get_factory().get_tokenizer_manager()

def get_worker_manager():
    """Get the worker manager instance."""
    return get_factory().get_worker_manager()

def get_wandb_manager():
    """Get the W&B manager instance."""
    return get_factory().get_wandb_manager()

def get_optuna_manager():
    """Get the Optuna manager instance."""
    return get_factory().get_optuna_manager()

def cleanup_managers():
    """Clean up all manager instances."""
    global _factory
    if _factory is not None:
        _factory.cleanup()
        _factory = None
        logger.info("Cleaned up all managers")

__all__ = [
    'get_amp_manager',
    'get_batch_manager',
    'get_cuda_manager',
    'get_data_manager',
    'get_dataloader_manager',
    'get_directory_manager',
    'get_metrics_manager',
    'get_model_manager',
    'get_parameter_manager',
    'get_resource_manager',
    'get_storage_manager',
    'get_tensor_manager',
    'get_tokenizer_manager',
    'get_worker_manager',
    'get_wandb_manager',
    'get_optuna_manager'
]
