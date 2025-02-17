# src/common/managers/model_manager.py (CORRECTED)
from __future__ import annotations
import torch
import logging
from typing import Dict, Any, Optional
from transformers import PreTrainedModel, BertConfig
from transformers.utils import logging as transformers_logging
from transformers.utils.hub import HFValidationError
import gc

from src.common.managers.base_manager import BaseManager
# DELAYED IMPORTS
# from src.common.managers import get_cuda_manager

logger = logging.getLogger(__name__)

def get_embedding_model():
    """Get EmbeddingBert model at runtime to avoid circular imports."""
    from src.embedding.models import EmbeddingBert
    return EmbeddingBert

def get_classification_model():
    """Get ClassificationBert model at runtime to avoid circular imports."""
    from src.classification.model import ClassificationBert
    return ClassificationBert

class ModelManager(BaseManager):
    """Manages model creation, loading, and device placement."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ModelManager."""
        super().__init__(config)
        self.process_models = {}
        self.model_refs = weakref.WeakValueDictionary()


    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        super()._initialize_process_local(config)


    def _verify_model_device(self, model: torch.nn.Module, device: torch.device) -> bool:
        """Verify model is on the correct device."""
        try:
            for param in model.parameters():
                if param.device != device:
                    logger.error(f"Parameter {param.shape} on {param.device}, expected {device}")
                    return False

            for buffer in model.buffers():
                if buffer.device != device:
                    logger.error(f"Buffer {buffer.shape} on {buffer.device}, expected {device}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error verifying model device: {str(e)}")
            return False

    def _move_model_to_device(self, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
        """Move model to device with verification."""
        try:
            model = model.to(device)
            if not self._verify_model_device(model, device):
                raise RuntimeError("Failed to move model to correct device")
            return model
        except Exception as e:
            logger.error(f"Error moving model to device: {str(e)}")
            raise

    def _optimize_model(self, model: torch.nn.Module, config: Dict[str, Any]) -> torch.nn.Module:
        """Apply PyTorch optimizations to model."""
        try:
            if config['training']['jit']:
                logger.info("Applying TorchScript optimization")
                with torch.jit.optimized_execution(True):
                    model = torch.jit.script(model)

            if config['training']['compile']:
                logger.info("Applying torch.compile optimization")
                model = torch.compile(
                    model,
                    mode=config['training']['compile_mode'],
                    fullgraph=True,
                    dynamic=False
                )
            return model
        except Exception as e:
            logger.error(f"Error optimizing model: {str(e)}")
            raise

    def get_worker_model(self, worker_id: int, model_name: str, model_type: str, device_id: Optional[int] = None, config: Optional[Dict[str, Any]] = None) -> PreTrainedModel:
        """Create model for worker."""
        current_pid = os.getpid()
        logger.debug(f"get_worker_model called from process {current_pid}")

        from src.common.managers import get_cuda_manager #DELAYED
        cuda_manager = get_cuda_manager()
        if config:
             cuda_manager.setup(config)

        device = cuda_manager.get_device()
        cache_key = f"{worker_id}_{model_name}"

        if cache_key in self.process_models:
            model = self.process_models[cache_key]
            if model is not None and self._verify_model_device(model, device):
                logger.info(f"Using cached model for worker {worker_id}")
                return model
            else:
                logger.warning(f"Removing invalid model from cache for worker {worker_id}")
                del self.process_models[cache_key]
                self.model_refs.pop(cache_key, None)
                gc.collect()
                cuda_manager.cleanup()

        logger.info(f"Creating new model for worker {worker_id} in process {current_pid}")
        cuda_manager.cleanup()


        try:
            logger.info(f"Creating model with clean CUDA state in process {current_pid}")
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("Invalid model name provided")

            transformers_logging.set_verbosity_error()

            try:
                model_config = BertConfig.from_pretrained(
                    model_name,
                    hidden_dropout_prob=config['training']['hidden_dropout_prob'],
                    attention_probs_dropout_prob=config['training']['attention_probs_dropout_prob']
                )

                if model_type == 'embedding':
                    model = get_embedding_model()(
                        config=model_config,
                        tie_weights=True
                    )
                elif model_type == 'classification':
                    model = get_classification_model()(
                        config=model_config,
                        num_labels=config['model']['num_labels']
                    )
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            except (OSError, HFValidationError) as e:
                raise RuntimeError(f"Failed to load model '{model_name}': {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error loading model: {str(e)}")
            finally:
                transformers_logging.set_verbosity_warning()

            if not isinstance(model, PreTrainedModel):
                raise RuntimeError("Model initialization failed")

            model = self._move_model_to_device(model, device)
            model = self._optimize_model(model, config)

            if model is None:
                raise ValueError("Model creation failed")

        except Exception as e:
            logger.error(f"Error creating model in process {current_pid}: {str(e)}")
            raise

        self.process_models[cache_key] = model
        self.model_refs[cache_key] = model
        logger.info(f"Successfully created model for worker {worker_id} on {device} in process {current_pid}")

        return model

    def cleanup_worker(self, worker_id: int):
        """Cleanup worker's model resources."""
        from src.common.managers import get_cuda_manager #DELAYED
        cuda_manager = get_cuda_manager()
        try:
            # Remove models for this worker
            keys_to_remove = [k for k in self.process_models.keys() if k.startswith(f"{worker_id}_")]
            for key in keys_to_remove:
                if key in self.process_models:
                    try:
                        # Move model to CPU before deletion
                        model = self.process_models[key]
                        if model is not None:
                            model.cpu()
                    except Exception as e:
                        logger.warning(f"Error cleaning up model: {str(e)}")
                    finally:
                        del self.process_models[key]
                        self.model_refs.pop(key, None)

            # Force garbage collection
            gc.collect()
            # Clean up CUDA resources
            cuda_manager.cleanup()
            logger.info(f"Cleaned up resources for worker {worker_id} in process {self._local.pid}")

        except Exception as e:
            logger.error(f"Error during worker cleanup: {str(e)}")
            cuda_manager.cleanup() #Cleanup even if exception

__all__ = ['ModelManager']