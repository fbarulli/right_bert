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
    def __init__(self, cuda_manager: CUDAManager, config: Optional[Dict[str, Any]] = None):
        self._cuda_manager = cuda_manager  # Set before super()
        super().__init__(config)
        self._local.num_workers = 0
        self._local.pin_memory = False

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize dataloader manager for this process."""
        try:
            super()._initialize_process_local(config)
            
            # Access config
            effective_config = config if config is not None else self._config
            training_config = effective_config.get('training', {})
            
            # Validate that cuda_manager is initialized, but don't throw exception
            # if it's not - just use safe defaults instead
            if not hasattr(self, '_cuda_manager') or not self._cuda_manager.is_initialized():
                logger.warning("CUDA manager not initialized, defaulting to CPU-only mode")
                self._local.num_workers = training_config.get('num_workers', 0)
                self._local.pin_memory = False
            else:
                # Configure dataloader based on CUDA availability
                cuda_available = self._cuda_manager.is_available()
                
                # Set number of workers
                self._local.num_workers = (
                    training_config.get('num_workers', 0) if cuda_available 
                    else min(training_config.get('num_workers', 0), 1)
                )
                
                # Set pin memory flag
                self._local.pin_memory = cuda_available and self._local.num_workers > 0
            
            logger.info(
                f"DataLoaderManager initialized for process {self._local.pid} "
                f"with {self._local.num_workers} workers and pin_memory={self._local.pin_memory}"
            )
        
        except Exception as e:
            logger.error(f"Failed to initialize DataLoaderManager: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise

    def create_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = None,
        persistent_workers: bool = True,
        config: Optional[Dict[str, Any]] = None
    ) -> torch.utils.data.DataLoader:
        """Create a DataLoader for the given dataset."""
        self.ensure_initialized()

        # Safety check for multiprocessing
        if num_workers > 0:
            # Check if dataset has non-picklable objects
            self._verify_dataset_picklability(dataset)
        
        # Use a simple worker init function that doesn't reference self
        def worker_init_fn(worker_id: int) -> None:
            """Initialize the worker process."""
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            # No manager references here - they'll be initialized in the worker

        # Fix for pickle error - use detached collate function
        actual_collate_fn = collate_fn if collate_fn else self._safe_collate_fn

        # Create the DataLoader
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and self._cuda_manager.is_available(),
            worker_init_fn=worker_init_fn,
            collate_fn=actual_collate_fn,
            persistent_workers=persistent_workers and num_workers > 0
        )

        logger.info(f"Created DataLoader with batch_size={batch_size}, num_workers={num_workers}")
        return loader

    def _safe_collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        """
        A collate function that doesn't reference any manager objects.
        
        Args:
            batch: A list of items to collate
            
        Returns:
            Dict[str, torch.Tensor]: A collated batch
        """
        # Simple implementation without references to self or other managers
        elem = batch[0]
        if isinstance(elem, dict):
            return {
                key: self._safe_collate_fn([d[key] for d in batch])
                if key in elem else None
                for key in elem
            }
        elif isinstance(elem, torch.Tensor):
            return torch.stack(batch, 0)
        else:
            return default_collate(batch)

    def _verify_dataset_picklability(self, dataset: torch.utils.data.Dataset) -> None:
        """
        Verify that a dataset can be pickled for multiprocessing.
        
        Args:
            dataset: The dataset to check
        """
        import pickle
        try:
            # Try to pickle just a tiny bit of the dataset to test
            pickle.dumps(type(dataset))
        except Exception as e:
            logger.warning(f"Dataset type may not be picklable: {e}")
            
        # Check for common problematic attributes
        for attr_name in dir(dataset):
            if attr_name.startswith('_'):
                continue
            try:
                attr = getattr(dataset, attr_name)
                if hasattr(attr, '__dict__'):
                    for sub_attr_name in dir(attr):
                        if isinstance(getattr(attr, sub_attr_name), threading.RLock):
                            logger.warning(f"Dataset contains RLock in {attr_name}.{sub_attr_name}")
                            # Try to make this attribute None or replace it
                            setattr(attr, sub_attr_name, None)
            except Exception:
                pass

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
            # Ensure _local exists before proceeding
            if not hasattr(self, '_local'):
                logger.info("No local resources to clean up in DataLoaderManager")
                return

            # Reset attributes if they exist
            if hasattr(self._local, 'num_workers'):
                self._local.num_workers = 0
            if hasattr(self._local, 'pin_memory'):
                self._local.pin_memory = False

            logger.info(f"Cleaned up DataLoaderManager for process {self._local.pid}")
            super().cleanup()

        except Exception as e:
            logger.error(f"Error cleaning up DataLoaderManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise


__all__ = ['DataLoaderManager']