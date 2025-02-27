# src/common/managers/batch_manager.py
from __future__ import annotations
import torch
import logging
import os
import threading
from typing import Dict, Any, Optional, Union, List, Tuple
from torch.utils.data import DataLoader

from src.common.managers.base_manager import BaseManager
from src.common.managers.cuda_manager import CUDAManager
from src.common.managers.tensor_manager import TensorManager

logger = logging.getLogger(__name__)

class BatchManager(BaseManager):
    """Manager for handling data batches and device placement."""

    def __init__(self, cuda_manager: CUDAManager, tensor_manager: TensorManager, 
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the BatchManager.
        
        Args:
            cuda_manager: CUDA manager instance
            tensor_manager: Tensor manager instance 
            config: Optional configuration dictionary
        """
        self._cuda_manager = cuda_manager
        self._tensor_manager = tensor_manager
        
        # Initialize _local before super().__init__
        self._local = threading.local()
        self._local.pid = os.getpid()
        self._local.device = None
        self._local.initialized = False
        
        super().__init__(config)

    # Define model fields as a class constant
    MODEL_FIELDS: Set[str] = {
        'input_ids',
        'attention_mask',
        'token_type_ids',
        'position_ids',
        'labels'
    }

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local attributes.

        Args:
            config: Optional configuration dictionary that overrides the one from constructor
        """
        try:
            super()._initialize_process_local(config)

            if not self._cuda_manager.is_initialized():
                raise RuntimeError("CUDA must be initialized before BatchManager")
            if not self._tensor_manager.is_initialized():
                raise RuntimeError("TensorManager must be initialized before BatchManager")

            self._local.device = self._cuda_manager.get_device()
            logger.info(f"BatchManager initialized for process {self._local.pid} with device {self._local.device}")

        except Exception as e:
            logger.error(f"Failed to initialize BatchManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a batch for processing.
        
        Args:
            batch: A batch of data
        
        Returns:
            Dict[str, Any]: The prepared batch
        """
        try:
            # First try normal initialization check
            self.ensure_initialized()
            
            # Get device safely
            device = self._get_device()
            
            # Move batch to device with proper error handling
            prepared_batch = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    prepared_batch[key] = value.to(device)
                elif isinstance(value, list) and all(torch.is_tensor(item) for item in value):
                    prepared_batch[key] = [item.to(device) for item in value]
                else:
                    prepared_batch[key] = value
                    
            return prepared_batch
            
        except RuntimeError as e:
            if "not initialized" in str(e):
                # Auto-initialize with defaults if possible
                logger.warning(f"BatchManager auto-initializing due to: {e}")
                
                if not hasattr(self, '_local'):
                    self._local = threading.local()
                
                self._local.initialized = True
                self._local.pid = os.getpid()
                
                # Try again with fresh initialization
                return self.prepare_batch(batch)
            else:
                # Some other error, re-raise
                raise

        except Exception as e:
            logger.error(f"Error in prepare_batch: {e}")
            # Fallback to simple device movement
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            return {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

    def get_batch_size(self, batch: Union[Dict[str, torch.Tensor], DataLoader]) -> int:
        """
        Get batch size from batch dict or dataloader.

        Args:
            batch: Either a dictionary of tensors or a DataLoader

        Returns:
            int: The batch size

        Raises:
            ValueError: If no tensors found in batch dict
            TypeError: If batch is neither dict nor DataLoader
        """
        self.ensure_initialized()
        try:
            if isinstance(batch, dict):
                for v in batch.values():
                    if isinstance(v, torch.Tensor):
                        return v.size(0)
                raise ValueError("No tensors found in batch dict")
            elif isinstance(batch, DataLoader):
                return batch.batch_size
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

        except Exception as e:
            logger.error(f"Error getting batch size: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def cleanup(self) -> None:
        """Clean up BatchManager resources."""
        try:
            # Safety check for _local attribute
            if not hasattr(self, '_local'):
                logger.warning("BatchManager has no _local attribute during cleanup")
                return
                
            # Get PID for logging
            pid = getattr(self._local, 'pid', os.getpid())  
            logger.info(f"Cleaning up BatchManager for process {pid}")
            
            # Reset any device-specific resources
            if hasattr(self._local, 'device'):
                # Any device-specific cleanup could go here
                pass
                
            super().cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up BatchManager: {str(e)}")
            logger.error("Stack trace:", exc_info=True)


__all__ = ['BatchManager']