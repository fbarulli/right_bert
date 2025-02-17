# src/common/managers/batch_manager.py
import torch
import logging
import traceback
from typing import Dict, Any, Optional, Union
from torch.utils.data import DataLoader

from src.common.managers.base_manager import BaseManager

logger = logging.getLogger(__name__)

class BatchManager(BaseManager):
    """Process-local batch manager for device placement."""

    def __init__(self): # No arguments!
        super().__init__()
        # self.cuda_manager = cuda_manager  # NO!
        # self.tensor_manager = tensor_manager # NO!

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        super()._initialize_process_local(config)
        # No more runtime imports in _initialize_process_local!
        self._local.device = None  # Initialize device

    def prepare_batch(
        self,
        batch: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """Move batch tensors to target device."""
        self.ensure_initialized()
        try:
            if device is None:
                # Get the managers using the getter functions
                device = get_cuda_manager().get_device()

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
                # Get first tensor's batch size
                for v in batch.values():
                    if isinstance(v, torch.Tensor):
                        return v.size(0)
                raise ValueError("No tensors found in batch dict")
            elif isinstance(batch, DataLoader):
                return batch.batch_size  # DataLoader has a batch_size attribute
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

        except Exception as e:
            logger.error(f"Error getting batch size: {str(e)}")
            logger.error(traceback.format_exc())
            raise

__all__ = ['BatchManager']