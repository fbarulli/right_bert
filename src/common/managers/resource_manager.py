
# src/common/managers/resource_manager.py
from __future__ import annotations
import torch
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import logging
import os
import traceback
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from src.common.managers.base_manager import BaseManager
from src.common.managers.cuda_manager import CUDAManager
from src.common.managers.model_manager import ModelManager
from src.common.managers.tokenizer_manager import TokenizerManager
from src.common.managers.data_manager import DataManager
from src.common.resource.resource_initializer import ResourceInitializer
from src.common.resource import resource_factory
from src.common.resource.resource_pool import ResourcePool

logger = logging.getLogger(__name__)

@dataclass
class ResourceConfig:
    """
    Serializable configuration for resources.

    Attributes:
        dataset_params: Dataset configuration parameters
        dataloader_params: DataLoader configuration parameters
        model_params: Model configuration parameters
        device_id: Optional device ID for resource allocation
    """
    dataset_params: Dict[str, Any]
    dataloader_params: Dict[str, Any]
    model_params: Dict[str, Any]
    device_id: Optional[int] = None

class ProcessResourceManager(BaseManager):
    """
    Manages per-process resource creation and cleanup.

    This manager handles:
    - Process-specific resource initialization
    - Resource pooling and memory management
    - Dataset and DataLoader creation
    - Resource cleanup
    """

    RESOURCE_TYPES: Dict[str, str] = {
        'dataset': 'Dataset resources',
        'model': 'Model resources',
        'optimizer': 'Optimizer resources',
        'dataloader': 'DataLoader resources'
    }

    def __init__(
        self,
        cuda_manager: CUDAManager,
        model_manager: ModelManager,
        tokenizer_manager: TokenizerManager,
        data_manager: DataManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ProcessResourceManager.

        Args:
            cuda_manager: Injected CUDAManager instance
            model_manager: Injected ModelManager instance
            tokenizer_manager: Injected TokenizerManager instance
            data_manager: Injected DataManager instance
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self._cuda_manager = cuda_manager
        self._model_manager = model_manager
        self._tokenizer_manager = tokenizer_manager
        self._data_manager = data_manager
        self._local.process_id = None
        self._local.device_id = None
        self._local.resources = {}
        self._local.resource_pool = None

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local attributes.

        Args:
            config: Optional configuration dictionary that overrides the one from constructor
        """
        try:
            super()._initialize_process_local(config)

            if not self._cuda_manager.is_initialized():
                raise RuntimeError("CUDAManager must be initialized before ProcessResourceManager")
            if not self._model_manager.is_initialized():
                raise RuntimeError("ModelManager must be initialized before ProcessResourceManager")
            if not self._tokenizer_manager.is_initialized():
                raise RuntimeError("TokenizerManager must be initialized before ProcessResourceManager")
            if not self._data_manager.is_initialized():
                raise RuntimeError("DataManager must be initialized before ProcessResourceManager")

            effective_config = config if config is not None else self._config
            if effective_config:
                resources_config = self.get_config_section(effective_config, 'resources')
                self._local.resource_pool = ResourcePool(
                    cuda_manager=self._cuda_manager,
                    memory_limit_gb=resources_config['max_memory_gb']
                )

            logger.info(f"ProcessResourceManager initialized for process {self._local.pid}")

        except Exception as e:
            logger.error(f"Failed to initialize ProcessResourceManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_datasets(
        self,
        config: Dict[str, Any],
        stage: str = 'embedding'
    ) -> Tuple[Dataset, Dataset]:
        """
        Create train and validation datasets.

        Args:
            config: Configuration dictionary
            stage: Model stage ('embedding' or 'classification')

        Returns:
            Tuple[Dataset, Dataset]: Train and validation datasets
        """
        self.ensure_initialized()
        try:
            tokenizer_manager = get_tokenizer_manager()
            tokenizer = tokenizer_manager.get_worker_tokenizer(
                worker_id=os.getpid(), #TODO: Fix worker id here
                model_name=config['model']['name'],
                model_type=config['model']['stage']
            )
            train_dataset = resource_factory.create_resource(
                'dataset',
                config,
                tokenizer = tokenizer,
                split='train',
                stage=stage
            )
            val_dataset = resource_factory.create_resource(
                'dataset',
                config,
                tokenizer = tokenizer,
                split='val',
                stage=stage
            )
            return train_dataset, val_dataset

        except Exception as e:
            logger.error(f"Error creating datasets: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_dataloaders(
        self,
        config: Dict[str, Any],
        train_dataset: Dataset,
        val_dataset: Dataset
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation dataloaders.

        Args:
            config: Configuration dictionary
            train_dataset: Training dataset
            val_dataset: Validation dataset

        Returns:
            Tuple[DataLoader, DataLoader]: Train and validation dataloaders
        """
        self.ensure_initialized()
        try:
            train_loader = resource_factory.create_resource(
                'dataloader',
                config,
                dataset=train_dataset,
                split='train'
            )
            val_loader = resource_factory.create_resource(
                'dataloader',
                config,
                dataset=val_dataset,
                split='val'
            )
            return train_loader, val_loader

        except Exception as e:
            logger.error(f"Error creating dataloaders: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def initialize_process(
        self,
        process_id: int,
        device_id: Optional[int] = None
    ) -> None:
        """
        Initialize resources for this process.

        Args:
            process_id: Process ID
            device_id: Optional device ID
        """
        self.ensure_initialized()
        try:
            ResourceInitializer.initialize_process(self._config)
            self._local.process_id = process_id
            self._local.device_id = device_id
            self._create_process_resources()

        except Exception as e:
            logger.error(f"Error initializing process: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _create_process_resources(self) -> None:
        """Create new resources specific to this process."""
        try:
            device = self._cuda_manager.get_device()
            self._local.resources = {
                'device': device,
                **self._data_manager.init_process_resources(self._config)
            }

            for resource_type, resource in self._local.resources.items():
                logger.debug(
                    f"Created resource {resource_type} of type {type(resource).__name__} "
                    f"for process {self._local.process_id}"
                )

        except Exception as e:
            logger.error(f"Failed to create process resources: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_resource(self, name: str) -> Any:
        """
        Get a resource, creating it if needed.

        Args:
            name: Resource name

        Returns:
            Any: The requested resource

        Raises:
            RuntimeError: If process not initialized
            KeyError: If resource not available
        """
        self.ensure_initialized()
        if self._local.process_id is None:
            raise RuntimeError("Process not initialized")
        if name not in self._local.resources:
            raise KeyError(f"Resource {name} not available")
        return self._local.resources[name]

    def cleanup(self) -> None:
        """Clean up process resources."""
        try:
            # Clean up individual resources
            for resource_type, resource in self._local.resources.items():
                try:
                    if hasattr(resource, 'cleanup'):
                        resource.cleanup()
                    logger.debug(f"Cleaned up resource {resource_type}")
                except Exception as e:
                    logger.error(f"Error cleaning up resource {resource_type}: {str(e)}")
                    logger.error(traceback.format_exc())

            # Clear resources and cleanup process
            self._local.resources.clear()
            ResourceInitializer.cleanup_process()

            # Reset process state
            self._local.process_id = None
            self._local.device_id = None
            self._local.resource_pool = None

            logger.info(f"Cleaned up ProcessResourceManager for process {self._local.pid}")
            super().cleanup()

        except Exception as e:
            logger.error(f"Error cleaning up ProcessResourceManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise


__all__ = ['ProcessResourceManager', 'ResourceConfig']