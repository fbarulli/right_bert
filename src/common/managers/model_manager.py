# src/common/managers/model_manager.py
from __future__ import annotations
import torch
import logging
from typing import Dict, Any, Optional
from transformers import PreTrainedModel, BertConfig
from transformers.utils import logging as transformers_logging
from transformers.utils.hub import HFValidationError
from src.common.managers import get_cuda_manager 

# Get manager instance
cuda_manager = get_cuda_manager()
import os
import gc
import weakref
import threading

from .base_manager import BaseManager

logger = logging.getLogger(__name__)

def get_embedding_model():
    """Get EmbeddingBert model at runtime to avoid circular imports."""
    from src.embedding.models import EmbeddingBert  
    return EmbeddingBert

def get_classification_model():
    """Get ClassificationBert model at runtime to avoid circular imports."""
    from src.classification.models import ClassificationBert  
    return ClassificationBert

class ModelManager(BaseManager):
    """Manages model creation, loading, and device placement."""

    def __init__(self):
        super().__init__()

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes."""
        super()._initialize_process_local(config)
        self._local.process_models = {}
        self._local.model_refs = weakref.WeakValueDictionary()

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

            for key, tensor in model.state_dict().items():
                if tensor.device != device:
                    logger.error(f"State dict tensor {key} on {tensor.device}, expected {device}")
                    return False

            def check_inputs(module, input):
                if isinstance(input, (tuple, list)):
                    for x in input:
                        if isinstance(x, torch.Tensor) and x.device != device:
                            logger.error(f"Input tensor {x.shape} on {x.device}, expected {device}")
                            return False
                elif isinstance(input, torch.Tensor) and input.device != device:
                    logger.error(f"Input tensor {input.shape} on {input.device}, expected {device}")
                    return False
                return True

            handle = model.register_forward_pre_hook(check_inputs)
            handle.remove()
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
            if config['training'].get('jit', False):
                logger.info("Applying TorchScript optimization")
                with torch.jit.optimized_execution(True):
                    model = torch.jit.script(model)

            if config['training'].get('compile', False):
                logger.info("Applying torch.compile optimization")
                model = torch.compile(
                    model,
                    mode=config['training'].get('compile_mode', 'default'),
                    fullgraph=True,
                    dynamic=False
                )
            if config['training'].get('static_graph', False):
                logger.info("Enabling static graph optimization")
                if hasattr(model, '_set_static_graph'):
                    model._set_static_graph()

            return model
        except Exception as e:
            logger.error(f"Error optimizing model: {str(e)}")
            raise

    def get_worker_model(
        self,
        worker_id: int,
        model_name: str,
        model_type: str,
        device_id: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> PreTrainedModel:
        """Create model for worker using process-local resources."""
        current_pid = os.getpid()
        logger.debug(f"get_worker_model called from process {current_pid}")
        if config:
            if not hasattr(cuda_manager._local, 'initialized'):
                cuda_manager.initialize()
            cuda_manager.setup(config)

        device = cuda_manager.get_device()
        cache_key = f"{worker_id}_{model_name}"

        try:
            if cache_key in self._local.process_models:
                model = self._local.process_models[cache_key]
                if model is not None and self._verify_model_device(model, device):
                    logger.info(f"Using cached model for worker {worker_id}")
                    return model
                else:
                    logger.warning(f"Removing invalid model from cache for worker {worker_id}")
                    del self._local.process_models[cache_key]
                    self._local.model_refs.pop(cache_key, None)
                    gc.collect()
                    cuda_manager.cleanup()

            logger.info(f"Creating new model for worker {worker_id} in process {current_pid}")
            cuda_manager.cleanup()
            cuda_manager.verify_cuda_state()

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
                    raise RuntimeError("Model initialization failed to return a valid model instance")

                model = self._move_model_to_device(model, device)
                model = self._optimize_model(model, config)
                cuda_manager.register_model(model)

                if model is None:
                    raise ValueError("Model creation failed")

            except Exception as e:
                logger.error(f"Error creating model in process {current_pid}: {str(e)}")
                raise

            self._local.process_models[cache_key] = model
            self._local.model_refs[cache_key] = model
            logger.info(f"Successfully created model for worker {worker_id} on {device} in process {current_pid}")

            return model

        except Exception as e:
            logger.error(f"Failed to get worker model in process {current_pid}: {str(e)}")
            if cache_key in self._local.process_models:
                try:
                    self._local.process_models[cache_key].cpu()
                except:
                    pass
                del self._local.process_models[cache_key]
                self._local.model_refs.pop(cache_key, None)
            cuda_manager.cleanup()
            gc.collect()
            raise

    def cleanup_worker(self, worker_id: int):
        """Cleanup worker's model resources for current process."""
        try:
            keys_to_remove = [k for k in self._local.process_models.keys() if k.startswith(f"{worker_id}_")]
            for key in keys_to_remove:
                if key in self._local.process_models:
                    try:
                        model = self._local.process_models[key]
                        if model is not None:
                            cuda_manager.unregister_model(model)
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.detach_()
                                    param.grad.zero_()
                            model.cpu()
                    except Exception as e:
                        logger.warning(f"Error cleaning up model: {str(e)}")
                    finally:
                        del self._local.process_models[key]
                        self._local.model_refs.pop(key, None)

            gc.collect()
            cuda_manager.cleanup()
            logger.info(f"Cleaned up resources for worker {worker_id} in process {self._local.pid}")

        except Exception as e:
            logger.error(f"Error during worker cleanup: {str(e)}")
            cuda_manager.cleanup()

__all__ = ['ModelManager']