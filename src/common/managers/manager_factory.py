# src/common/managers/manager_factory.py
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Type, TypeVar

from .base_manager import BaseManager
from .storage_manager import StorageManager
from .optuna_manager import OptunaManager
from .cuda_manager import CUDAManager
from .batch_manager import BatchManager
from .amp_manager import AMPManager
from .tokenizer_manager import TokenizerManager
from .metrics_manager import MetricsManager
from .model_manager import ModelManager
from .parameter_manager import ParameterManager
from .resource_manager import ProcessResourceManager
from .tensor_manager import TensorManager
from .wandb_manager import WandbManager
from .worker_manager import WorkerManager
from .data_manager import DataManager
from .dataloader_manager import DataLoaderManager
from .directory_manager import DirectoryManager

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseManager)

class ManagerFactory:
    """Central factory for creating and managing all manager instances."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the factory with configuration."""
        if not config:
            raise ValueError("Configuration cannot be empty")
        self.validate_config(config)
        self.config = config
        self._instances: Dict[str, BaseManager] = {}
        
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate the configuration."""
        required_sections = {'output', 'training', 'model', 'data', 'resources'}
        missing = required_sections - set(config.keys())
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")
            
        # Validate output section
        output_config = config.get('output', {})
        required_output = {'dir', 'storage_dir', 'wandb'}
        missing_output = required_output - set(output_config.keys())
        if missing_output:
            raise ValueError(f"Missing required output fields: {missing_output}")
    
    def _get_or_create(self, key: str, manager_class: Type[T], *args, **kwargs) -> T:
        """Get an existing manager instance or create a new one."""
        if key not in self._instances:
            logger.debug(f"Creating new {key} manager")
            self._instances[key] = manager_class(self, *args, **kwargs)
        return self._instances[key]  # type: ignore
    
    def get_storage_manager(self) -> StorageManager:
        """Get the storage manager instance."""
        return self._get_or_create('storage', StorageManager)
    
    def get_optuna_manager(self) -> OptunaManager:
        """Get the Optuna manager instance."""
        return self._get_or_create(
            'optuna',
            OptunaManager,
            study_name=self.config['training'].get('study_name', 'embedding_study')
        )
    
    def get_cuda_manager(self) -> CUDAManager:
        """Get the CUDA manager instance."""
        return self._get_or_create('cuda', CUDAManager)
    
    def get_batch_manager(self) -> BatchManager:
        """Get the batch manager instance."""
        return self._get_or_create('batch', BatchManager)
    
    def get_amp_manager(self) -> AMPManager:
        """Get the AMP manager instance."""
        return self._get_or_create('amp', AMPManager)
    
    def get_tokenizer_manager(self) -> TokenizerManager:
        """Get the tokenizer manager instance."""
        return self._get_or_create('tokenizer', TokenizerManager)
    
    def get_metrics_manager(self) -> MetricsManager:
        """Get the metrics manager instance."""
        return self._get_or_create('metrics', MetricsManager)
    
    def get_model_manager(self) -> ModelManager:
        """Get the model manager instance."""
        return self._get_or_create('model', ModelManager)
    
    def get_parameter_manager(self) -> ParameterManager:
        """Get the parameter manager instance."""
        return self._get_or_create('parameter', ParameterManager)
    
    def get_resource_manager(self) -> ProcessResourceManager:
        """Get the resource manager instance."""
        return self._get_or_create('resource', ProcessResourceManager)
    
    def get_tensor_manager(self) -> TensorManager:
        """Get the tensor manager instance."""
        return self._get_or_create('tensor', TensorManager)
    
    def get_wandb_manager(self) -> WandbManager:
        """Get the W&B manager instance."""
        return self._get_or_create('wandb', WandbManager)
    
    def get_worker_manager(self) -> WorkerManager:
        """Get the worker manager instance."""
        return self._get_or_create('worker', WorkerManager)
    
    def get_data_manager(self) -> DataManager:
        """Get the data manager instance."""
        return self._get_or_create('data', DataManager)
    
    def get_dataloader_manager(self) -> DataLoaderManager:
        """Get the dataloader manager instance."""
        return self._get_or_create('dataloader', DataLoaderManager)
    
    def get_directory_manager(self) -> DirectoryManager:
        """Get the directory manager instance."""
        base_dir = Path(self.config['output']['dir'])
        return self._get_or_create('directory', DirectoryManager, base_dir=base_dir)
    
    def cleanup(self) -> None:
        """Clean up all manager instances."""
        for manager in self._instances.values():
            try:
                if hasattr(manager, 'cleanup'):
                    manager.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up {type(manager).__name__}: {e}")
        self._instances.clear()

__all__ = ['ManagerFactory']
