# src/common/managers/dataloader_manager.py (Refactored)
from __future__ import annotations
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import os
import logging
import traceback
from typing import Any, Optional, Dict, Callable
from torch.utils.data import Dataset, DataLoader

from src.common.managers.base_manager import BaseManager
from src.common.managers import get_cuda_manager  # Corrected import
from src.common.resource.resource_initializer import ResourceInitializer
from src.common.process.initialization import get_worker_init_fn

logger = logging.getLogger(__name__)

class DataLoaderManager(BaseManager):
    """Process-local dataloader manager."""

    def __init__(self):
        super().__init__()
    
    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes."""
        try:
            super()._initialize_process_local(config)
            
            logger.info("Initializing DataLoaderManager for process %s", os.getpid())
            
            if not hasattr(self._local, 'num_workers'):
                self._local.num_workers = 0
            if not hasattr(self._local, 'pin_memory'):
                self._local.pin_memory = False
                
            cuda_manager = get_cuda_manager()
            cuda_manager.ensure_initialized()
            self._local.pin_memory = cuda_manager.is_available()
                
            self._local.settings_initialized = True

            logger.info(
                "DataLoader settings initialized for process %s (pin_memory=%s)",
                self._local.pid,
                self._local.pin_memory,
            )
        except Exception as e:
            logger.error("Failed to initialize DataLoader settings: %s", str(e))
            logger.error(traceback.format_exc())
            if hasattr(self._local, 'settings_initialized'):
                delattr(self._local, 'settings_initialized')
            raise

    def is_initialized(self) -> bool:
        """Check if manager is fully initialized."""
        return (
            super().is_initialized() and
            hasattr(self._local, 'settings_initialized') and
            self._local.settings_initialized
        )
            
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
        """Create dataloader with proper settings."""
        self.ensure_initialized()
        try:
            if not hasattr(self._local, 'num_workers'):
                self._initialize_process_local()
            
            if num_workers is None:
                num_workers = self._local.num_workers
            
            kwargs = dict(kwargs)
            if num_workers > 0:
                prefetch = 2 
                if 'prefetch_factor' in kwargs:
                    prefetch = kwargs['prefetch_factor']
                kwargs['prefetch_factor'] = prefetch
            elif 'prefetch_factor' in kwargs:
                del kwargs['prefetch_factor']

            if pin_memory is None:
                pin_memory = hasattr(self._local, 'pin_memory') and self._local.pin_memory
                
            if num_workers > 0:
                ResourceInitializer._config = config
                
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
            
        except Exception as e:
            logger.error(f"Error creating dataloader: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def set_worker_settings(
        self,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None
    ) -> None:
        """Update default worker settings."""
        self.ensure_initialized()
        try:
            self._local.num_workers = num_workers
            if pin_memory is not None:
                cuda_manager = get_cuda_manager()
                self._local.pin_memory = pin_memory and cuda_manager.is_available()
                
            logger.debug(
                f"Updated DataLoader settings: num_workers={self._local.num_workers}, "
                f"pin_memory={self._local.pin_memory}"
            )
            
        except Exception as e:
            logger.error(f"Error updating worker settings: {str(e)}")
            logger.error(traceback.format_exc())
            raise
__all__ = ['DataLoaderManager']