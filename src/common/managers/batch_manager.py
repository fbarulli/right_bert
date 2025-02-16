# src/common/managers/batch_manager.py
from __future__ import annotations
import torch
import logging
import traceback
from typing import Dict, Any, Optional, Union
from torch.utils.data import DataLoader

from src.common.managers.base_manager import BaseManager

logger = logging.getLogger(__name__)

class BatchManager(BaseManager):
    """Process-local batch manager for device placement and memory management."""

    def __init__(self, cuda_manager, tensor_manager):
        super().__init__()
        self.cuda_manager = cuda_manager
        self.tensor_manager = tensor_manager

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        super()._initialize_process_local(config)
        self.cuda_manager.ensure_initialized()
        self.tensor_manager.ensure_initialized()

        self._local.device = None

    def prepare_batch(
        self,
        batch: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """Move batch tensors to target device."""
        self.ensure_initialized()
        try:
            if device is None:
                device = self.cuda_manager.get_device()

            model_fields = {'input_ids', 'attention_mask', 'token_type_ids', 'position_ids', 'labels'}
            return {
                k: v.to(device=device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k in model_fields
            }

        except Exception as e:
            logger.error(f"Error preparing batch: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def get_batch_size(self, batch: Union[Dict[str, torch.Tensor], DataLoader]) -> int:
        """Get batch size from batch dict or dataloader."""
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

__all__ = ['BatchManager']