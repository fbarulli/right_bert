# src/common/managers/model_manager.py
from __future__ import annotations
import os
import torch
import logging
import gc
import weakref
from typing import Dict, Any, Optional
from transformers import PreTrainedModel, BertConfig
from transformers.utils import logging as transformers_logging
from transformers.utils.hub import HFValidationError

from src.common.managers.base_manager import BaseManager
from src.common.managers.cuda_manager import CUDAManager
from src.common.managers.tokenizer_manager import TokenizerManager

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
    """
    Manages model creation, loading, and device placement.
    
    This manager handles:
    - Model initialization and configuration
    - Device placement and verification
    - Model optimization and caching
    - Resource cleanup
    """

    def __init__(
        self,
        cuda_manager: CUDAManager,
        tokenizer_manager: TokenizerManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ModelManager.

        Args:
            cuda_manager: Injected CUDAManager instance
            tokenizer_manager: Injected TokenizerManager instance
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self._cuda_manager = cuda_manager
        self._tokenizer_manager = tokenizer_manager
        self._local.process_models = {}
        self._local.model_refs = weakref.WeakValueDictionary()

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local attributes.

        Args:
            config: Optional configuration dictionary that overrides the one from constructor
        """
        try:
            super()._initialize_process_local(config)

            if not self._cuda_manager.is_initialized():
                raise RuntimeError("CUDAManager must be initialized before ModelManager")
            if not self._tokenizer_manager.is_initialized():
                raise RuntimeError("TokenizerManager must be initialized before ModelManager")

            logger.info(f"ModelManager initialized for process {self._local.pid}")

        except Exception as e:
            logger.error(f"Failed to initialize ModelManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _verify_model_device(self, model: torch.nn.Module, device: torch.device) -> bool:
        """
        Verify model is on the correct device.

        Args:
            model: The model to verify
            device: Expected device

        Returns:
            bool: True if model is on correct device, False otherwise
        """
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
        """
        Move model to device with verification.

        Args:
            model: The model to move
            device: Target device

        Returns:
            torch.nn.Module: Model on target device

        Raises:
            RuntimeError: If model fails to move to device
        """
        try:
            model = model.to(device)
            if not self._verify_model_device(model, device):
                raise RuntimeError("Failed to move model to correct device")
            return model

        except Exception as e:
            logger.error(f"Error moving model to device: {str(e)}")
            raise

    def _optimize_model(self, model: torch.nn.Module, config: Dict[str, Any]) -> torch.nn.Module:
        """
        Apply PyTorch optimizations to model.

        Args:
            model: The model to optimize
            config: Configuration dictionary

        Returns:
            torch.nn.Module: Optimized model
        """
        try:
            training_config = self.get_config_section(config, 'training')

            if training_config.get('jit', False):
                logger.info("Applying TorchScript optimization")
                with torch.jit.optimized_execution(True):
                    model = torch.jit.script(model)

            if training_config.get('compile', False):
                logger.info("Applying torch.compile optimization")
                model = torch.compile(
                    model,
                    mode=training_config.get('compile_mode', 'default'),
                    fullgraph=True,
                    dynamic=False
                )
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
        """
        Create or get cached model for worker.

        Args:
            worker_id: Worker process ID
            model_name: Name/path of the model
            model_type: Type of model ('embedding' or 'classification')
            device_id: Optional device ID
            config: Optional configuration dictionary

        Returns:
            PreTrainedModel: The model instance

        Raises:
            ValueError: If model name or type is invalid
            RuntimeError: If model creation fails
        """
        self.ensure_initialized()

        try:
            # Setup device
            if config:
                self._cuda_manager.setup(config)
            device = self._cuda_manager.get_device()

            # Check cache
            cache_key = f"{worker_id}_{model_name}"
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
                    self._cuda_manager.cleanup()

            # Create new model
            logger.info(f"Creating new model for worker {worker_id} in process {self._local.pid}")
            self._cuda_manager.cleanup()

            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("Invalid model name provided")

            transformers_logging.set_verbosity_error()
            try:
                # Get model configuration
                model_config = BertConfig.from_pretrained(
                    model_name,
                    hidden_dropout_prob=config['training']['hidden_dropout_prob'],
                    attention_probs_dropout_prob=config['training']['attention_probs_dropout_prob']
                )

                # Create model based on type
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

            # Move to device and optimize
            model = self._move_model_to_device(model, device)
            model = self._optimize_model(model, config)

            if model is None:
                raise ValueError("Model creation failed")

            # Cache model
            self._local.process_models[cache_key] = model
            self._local.model_refs[cache_key] = model

            logger.info(
                f"Successfully created model for worker {worker_id} "
                f"on {device} in process {self._local.pid}"
            )
            return model

        except Exception as e:
            logger.error(f"Error creating model in process {self._local.pid}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def cleanup_worker(self, worker_id: int) -> None:
        """
        Cleanup worker's model resources.

        Args:
            worker_id: Worker process ID to cleanup
        """
        self.ensure_initialized()
        try:
            # Remove models for this worker
            keys_to_remove = [
                k for k in self._local.process_models.keys()
                if k.startswith(f"{worker_id}_")
            ]
            for key in keys_to_remove:
                if key in self._local.process_models:
                    try:
                        # Move model to CPU before deletion
                        model = self._local.process_models[key]
                        if model is not None:
                            model.cpu()
                    except Exception as e:
                        logger.warning(f"Error cleaning up model: {str(e)}")
                    finally:
                        del self._local.process_models[key]
                        self._local.model_refs.pop(key, None)

            # Force garbage collection
            gc.collect()
            # Clean up CUDA resources
            self._cuda_manager.cleanup()

            logger.info(f"Cleaned up resources for worker {worker_id} in process {self._local.pid}")

        except Exception as e:
            logger.error(f"Error during worker cleanup: {str(e)}")
            logger.error(traceback.format_exc())
            self._cuda_manager.cleanup()  # Cleanup even if exception
            raise

    def cleanup(self) -> None:
        """Clean up model manager resources."""
        try:
            # Clear model caches
            self._local.process_models.clear()
            self._local.model_refs.clear()

            # Force garbage collection
            gc.collect()

            logger.info(f"Cleaned up ModelManager for process {self._local.pid}")
            super().cleanup()

        except Exception as e:
            logger.error(f"Error cleaning up ModelManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise


__all__ = ['ModelManager']
