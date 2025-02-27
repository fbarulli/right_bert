# src/common/managers/tokenizer_manager.py
from __future__ import annotations
import logging
import os
import weakref
import threading
import traceback
from typing import Dict, Any, Optional
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from transformers.utils import logging as transformers_logging

from src.common.managers.base_manager import BaseManager

logger = logging.getLogger(__name__)

class TokenizerManager(BaseManager):
    """
    Process-local tokenizer manager.

    This manager handles:
    - Tokenizer creation and caching
    - Worker-specific tokenizer management
    - Shared tokenizer access
    - Resource cleanup
    """

    # Class-level shared resources
    _shared_tokenizer: Optional[PreTrainedTokenizerFast] = None
    _tokenizer_lock = threading.Lock()

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TokenizerManager.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self._local.process_tokenizers = {}
        self._local.tokenizer_refs = weakref.WeakValueDictionary()

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local attributes.

        Args:
            config: Optional configuration dictionary that overrides the one from constructor
        """
        try:
            super()._initialize_process_local(config)
            logger.info(f"TokenizerManager initialized for process {self._local.pid}")

        except Exception as e:
            logger.error(f"Failed to initialize TokenizerManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Setup TokenizerManager for the current process.
        
        Args:
            config: Configuration dictionary
        """
        try:
            # Ensure process-local initialization
            if not hasattr(self._local, 'initialized') or not self._local.initialized:
                self._initialize_process_local(config or self._config)
            
            logger.info(f"TokenizerManager setup complete for process {self._local.pid}")
        except Exception as e:
            logger.error(f"TokenizerManager setup failed: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise

    @classmethod
    def set_shared_tokenizer(cls, tokenizer: PreTrainedTokenizerFast) -> None:
        """
        Set the shared tokenizer instance.

        Args:
            tokenizer: Tokenizer instance to share
        """
        with cls._tokenizer_lock:
            cls._shared_tokenizer = tokenizer
            logger.info("Set shared tokenizer")

    @classmethod
    def get_shared_tokenizer(cls) -> Optional[PreTrainedTokenizerFast]:
        """
        Get the shared tokenizer instance.

        Returns:
            Optional[PreTrainedTokenizerFast]: Shared tokenizer if available
        """
        with cls._tokenizer_lock:
            return cls._shared_tokenizer

    def get_worker_tokenizer(
        self,
        worker_id: int,
        model_name: Optional[str] = None,
        model_type: str = 'embedding'
    ) -> PreTrainedTokenizerFast:
        """
        Get or create tokenizer for a worker.

        Args:
            worker_id: Worker process ID
            model_name: Name/path of the model
            model_type: Type of model ('embedding' or 'classification')

        Returns:
            PreTrainedTokenizerFast: The tokenizer instance
        """
        # Check if we need auto-setup
        if not self.is_initialized():
            logger.warning(f"Auto-initializing TokenizerManager in process {os.getpid()}")
            self.setup(self._config)
        
        try:
            # Check shared tokenizer first
            shared_tokenizer = self.get_shared_tokenizer()
            if shared_tokenizer is not None:
                logger.info(f"Using shared tokenizer for worker {worker_id}")
                return shared_tokenizer

            # Handle model_name input
            if model_name is None:
                model_name = self._config.get('model', {}).get('name')
                if not model_name:
                    raise ValueError("No model name provided and none in config")

            # Check cache
            cache_key = f"{worker_id}_{model_name}"
            if hasattr(self._local, 'process_tokenizers') and cache_key in self._local.process_tokenizers:
                tokenizer = self._local.process_tokenizers[cache_key]
                if tokenizer is not None:
                    logger.info(f"Using cached tokenizer for worker {worker_id}")
                    return tokenizer

            # Initialize tokenizers dict if not exists
            if not hasattr(self._local, 'process_tokenizers'):
                self._local.process_tokenizers = {}
                self._local.tokenizer_refs = weakref.WeakValueDictionary()
            
            # Create new tokenizer
            logger.info(f"Creating new tokenizer for worker {worker_id} in process {os.getpid()}")

            transformers_logging.set_verbosity_error()
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                if not isinstance(tokenizer, PreTrainedTokenizerFast):
                    logger.warning(f"Fast tokenizer not available for {model_name}, using slow version")
            except Exception as e:
                logger.error(f"Error loading tokenizer: {e}")
                raise RuntimeError(f"Failed to load tokenizer for model '{model_name}': {str(e)}")
            finally:
                transformers_logging.set_verbosity_warning()

            # Cache tokenizer
            self._local.process_tokenizers[cache_key] = tokenizer
            self._local.tokenizer_refs[cache_key] = tokenizer

            logger.info(f"Successfully created tokenizer for worker {worker_id} in process {os.getpid()}")
            return tokenizer

        except Exception as e:
            logger.error(f"Error getting worker tokenizer: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def cleanup_worker(self, worker_id: int) -> None:
        """
        Clean up worker's tokenizer resources.

        Args:
            worker_id: Worker process ID to cleanup
        """
        self.ensure_initialized()
        try:
            # Remove tokenizers for this worker
            keys_to_remove = [
                k for k in self._local.process_tokenizers.keys()
                if k.startswith(f"{worker_id}_")
            ]
            for key in keys_to_remove:
                if key in self._local.process_tokenizers:
                    del self._local.process_tokenizers[key]
                    self._local.tokenizer_refs.pop(key, None)

            logger.info(
                f"Cleaned up tokenizer resources for worker {worker_id} "
                f"in process {self._local.pid}"
            )

        except Exception as e:
            logger.error(f"Error cleaning up worker {worker_id}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def cleanup(self) -> None:
        """Clean up tokenizer manager resources."""
        try:
            # Clear process-local tokenizers
            self._local.process_tokenizers.clear() #can be dependent on model_manager if tokenizer is tied to model
            self._local.tokenizer_refs.clear()

            # Clear shared tokenizer
            with self._tokenizer_lock:
                self._shared_tokenizer = None

            logger.info(f"Cleaned up TokenizerManager for process {self._local.pid}")
            super().cleanup()

        except Exception as e:
            logger.error(f"Error cleaning up TokenizerManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise


__all__ = ['TokenizerManager']