# src/common/managers/dataloader_manager.py
from __future__ import annotations
import torch
from torch.utils.data import DataLoader, Dataset
import os
import logging
import traceback
from typing import Any, Optional, Dict, Callable

from src.common.managers.base_manager import BaseManager
from src.common.managers.cuda_manager import CUDAManager
from src.common.process.initialization import get_worker_init_fn
from src.common.resource.resource_initializer import ResourceInitializer

logger = logging.getLogger(__name__)

class DataLoaderManager(BaseManager):
    """
    Process-local dataloader manager.
    
    This manager handles:
    - DataLoader creation and configuration
    - Worker initialization and settings
    - Memory pinning optimization
    """

    def __init__(
        self,
        cuda_manager: CUDAManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize DataLoaderManager.

        Args:
            cuda_manager: Injected CUDAManager instance
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self._cuda_manager = cuda_manager
        self._local.num_workers = 0
        self._local.pin_memory = False

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local attributes.

        Args:
            config: Optional configuration dictionary that overrides the one from constructor
        """
        try:
            super()._initialize_process_local(config)

            if not self._cuda_manager.is_initialized():
                raise RuntimeError("CUDAManager must be initialized before DataLoaderManager")

            # Set default values
            effective_config = config if config is not None else self._config
            if effective_config:
                training_config = self.get_config_section(effective_config, 'training')
                self._local.num_workers = training_config.get('num_workers', 0)
            else:
                self._local.num_workers = 0

            # Set pin memory based on CUDA availability
            self._local.pin_memory = self._cuda_manager.is_available()

            logger.info(
                f"DataLoaderManager initialized for process {self._local.pid} "
                f"with {self._local.num_workers} workers and "
                f"pin_memory={self._local.pin_memory}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize DataLoaderManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> DataLoader:
        """
        Create a dataloader with specified settings.

        Args:
            dataset: The dataset to load
            batch_size: Batch size for the dataloader
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
            config: Optional configuration dictionary
            **kwargs: Additional arguments to pass to DataLoader

        Returns:
            DataLoader: The configured dataloader
        """
        self.ensure_initialized()

        try:
            # Use provided values or defaults
            effective_num_workers = num_workers if num_workers is not None else self._local.num_workers
            effective_pin_memory = pin_memory if pin_memory is not None else self._local.pin_memory

            # Handle prefetch factor
            dataloader_kwargs = dict(kwargs)
            if effective_num_workers > 0:
                dataloader_kwargs['prefetch_factor'] = kwargs.get('prefetch_factor', 2)
            elif 'prefetch_factor' in dataloader_kwargs:
                del dataloader_kwargs['prefetch_factor']

            # Set resource config for workers
            if effective_num_workers > 0:
                ResourceInitializer._config = config

            # Create worker init function
            worker_init_fn = get_worker_init_fn(effective_num_workers)

            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=effective_num_workers,
                pin_memory=effective_pin_memory,
                worker_init_fn=worker_init_fn,
                **dataloader_kwargs
            )

            logger.debug(
                f"Created DataLoader with batch_size={batch_size}, "
                f"num_workers={effective_num_workers}, "
                f"pin_memory={effective_pin_memory}"
            )

            return dataloader

        except Exception as e:
            logger.error(f"Error creating dataloader: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def set_worker_settings(
        self,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None
    ) -> None:
        """
        Update worker settings.

        Args:
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.ensure_initialized()
        try:
            self._local.num_workers = num_workers
            if pin_memory is not None:
                self._local.pin_memory = pin_memory and self._cuda_manager.is_available()

            logger.debug(
                f"Updated DataLoader settings: "
                f"num_workers={self._local.num_workers}, "
                f"pin_memory={self._local.pin_memory}"
            )

        except Exception as e:
            logger.error(f"Error updating worker settings: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def cleanup(self) -> None:
        """Clean up dataloader manager resources."""
        try:
            self._local.num_workers = 0
            self._local.pin_memory = False
            logger.info(f"Cleaned up DataLoaderManager for process {self._local.pid}")
            super().cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up DataLoaderManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise


__all__ = ['DataLoaderManager']
