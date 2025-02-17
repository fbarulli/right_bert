# src/common/managers/dataloader_manager.py (CORRECTED)
from __future__ import annotations
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import os
import logging
import traceback
from typing import Any, Optional, Dict, Callable
from torch.utils.data import Dataset, DataLoader

from src.common.managers.base_manager import BaseManager
# DELAYED IMPORTS
# from src.common.managers import get_cuda_manager
from src.common.resource.resource_initializer import ResourceInitializer
from src.common.process.initialization import get_worker_init_fn
# from .base_manager import BaseManager  # No longer needed, already imported at top
logger = logging.getLogger(__name__)

class DataLoaderManager(BaseManager):
    """Process-local dataloader manager."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DataLoaderManager."""
        super().__init__(config)  # Initialize base
        # self._local.num_workers = 0 #moved to initialize
        # self._local.pin_memory = False

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        super()._initialize_process_local(config)
        logger.info("Initializing DataLoaderManager for process %s", os.getpid())
        self._local.num_workers = 0  # Default
        self._local.pin_memory = False # Default
        from src.common.managers import get_cuda_manager #DELAYED
        cuda_manager = get_cuda_manager()
        cuda_manager.ensure_initialized()
        self._local.pin_memory = cuda_manager.is_available()

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
        """Create dataloader with settings."""
        # Ensure initialized before accessing _local attributes
        self.ensure_initialized()
        if num_workers is None:
            num_workers = self._local.num_workers

        # Handle prefetch_factor correctly.
        kwargs = dict(kwargs)  # Make a copy to avoid modifying the original
        if num_workers > 0:
            kwargs['prefetch_factor'] = 2  # Default prefetch_factor
        elif 'prefetch_factor' in kwargs:
            del kwargs['prefetch_factor'] # Remove if num_workers is 0

        if pin_memory is None:
            pin_memory = self._local.pin_memory

        if num_workers > 0:
            ResourceInitializer._config = config # type: ignore

        worker_init_fn = get_worker_init_fn(num_workers)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            **kwargs
        )

        logger.debug(
            f"Created DataLoader with batch_size={batch_size}, "
            f"num_workers={num_workers}, pin_memory={pin_memory}"
        )
        return dataloader


    def set_worker_settings(self, num_workers: int = 0, pin_memory: Optional[bool] = None) -> None:
        """Update worker settings."""
        from src.common.managers import get_cuda_manager
        self.ensure_initialized()  # Ensure manager is initialized
        self._local.num_workers = num_workers
        if pin_memory is not None:
            cuda_manager = get_cuda_manager()
            self._local.pin_memory = pin_memory and cuda_manager.is_available()

        logger.debug(f"Updated DataLoader settings: num_workers={self._local.num_workers}, pin_memory={self._local.pin_memory}")

__all__ = ['DataLoaderManager']