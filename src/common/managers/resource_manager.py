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

from src.common.resource.resource_initializer import ResourceInitializer
from src.common.managers import get_cuda_manager
from src.common.resource import resource_factory
from .base_manager import BaseManager

cuda_manager = get_cuda_manager()

logger = logging.getLogger(__name__)

RESOURCE_TYPES = {
    'dataset': 'Dataset resources',
    'model': 'Model resources',
    'optimizer': 'Optimizer resources',
    'dataloader': 'DataLoader resources'
}

@dataclass
class ResourceConfig:
    """Serializable configuration for resources."""
    dataset_params: Dict[str, Any]
    dataloader_params: Dict[str, Any]
    model_params: Dict[str, Any]
    device_id: Optional[int] = None

class ProcessResourceManager(BaseManager):
    """Manages per-process resource creation and cleanup."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config # It will be initialized later
        self.resource_pool = None


    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes."""
        super()._initialize_process_local(config)
        self.process_id = None
        self.resources = {}
        from src.common.resource.resource_pool import ResourcePool
        self.resource_pool =  ResourcePool(memory_limit_gb=config['resources']['max_memory_gb'])

    def create_datasets(
        self,
        config: Dict[str, Any],
        stage: str = 'embedding'
    ) -> Tuple[Dataset, Dataset]:
        """Create train and validation datasets."""
        train_dataset = resource_factory.create_resource(
            'dataset',
            config,
            split='train',
            stage=stage
        )
        val_dataset = resource_factory.create_resource(
            'dataset',
            config,
            split='val',
            stage=stage
        )
        return train_dataset, val_dataset

    def create_dataloaders(
        self,
        config: Dict[str, Any],
        train_dataset: Dataset,
        val_dataset: Dataset
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
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

    def initialize_process(self, process_id: int, device_id: Optional[int] = None) -> None:
        """Initialize resources for this process."""
        ResourceInitializer.initialize_process(self.config)

        self.process_id = process_id
        self.device_id = device_id

        self._create_process_resources()

    def _create_process_resources(self) -> None:
        """Create new resources specific to this process."""
        try:
            from src.common.managers.data_manager import DataManager

            device = cuda_manager.get_device()

            from src.common.managers import get_data_manager
            data_mgr = get_data_manager()

            self.resources = {
                'device': device,
                **data_mgr.init_process_resources(self.config)
            }

            for resource_type, resource in self.resources.items():
                logger.debug(
                    f"Created resource {resource_type} of type {type(resource).__name__} "
                    f"for process {self.process_id}"
                )

        except Exception as e:
            logger.error(f"Failed to create process resources: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_resource(self, name: str) -> Any:
        """Get a resource, creating it if needed."""
        if self.process_id is None:
            raise RuntimeError("Process not initialized")

        if name not in self.resources:
            raise KeyError(f"Resource {name} not available")
        return self.resources[name]

    def cleanup(self) -> None:
        """Clean up process resources."""
        try:
            for resource_type, resource in self.resources.items():
                try:
                    if hasattr(resource, 'cleanup'):
                        resource.cleanup()
                    logger.debug(f"Cleaned up resource {resource_type}")
                except Exception as e:
                    logger.error(f"Error cleaning up resource {resource_type}: {str(e)}")

            self.resources.clear()
            ResourceInitializer.cleanup_process()

        except Exception as e:
            logger.error(f"Error during resource cleanup: {str(e)}")
            logger.error(traceback.format_exc())
            raise

__all__ = ['ProcessResourceManager', 'ResourceConfig']