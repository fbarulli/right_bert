
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
        model_name: str,
        model_type: str = 'embedding',
        config: Optional[Dict[str, Any]] = None
    ) -> PreTrainedTokenizerFast:
        """
        Get or create tokenizer for a worker.

        Args:
            worker_id: Worker process ID
            model_name: Name/path of the model
            model_type: Type of model ('embedding' or 'classification')
            config: Optional configuration dictionary

        Returns:
            PreTrainedTokenizerFast: The tokenizer instance

        Raises:
            ValueError: If model name is invalid
            RuntimeError: If tokenizer creation fails
        """
        self.ensure_initialized()

        try:
            logger.debug(f"get_worker_tokenizer called from process {self._local.pid}")

            # Check shared tokenizer first
            shared_tokenizer = self.get_shared_tokenizer()
            if shared_tokenizer is not None:
                logger.info(f"Using shared tokenizer for worker {worker_id}")
                return shared_tokenizer

            # Check cache
            cache_key = f"{worker_id}_{model_name}"
            if cache_key in self._local.process_tokenizers:
                tokenizer = self._local.process_tokenizers[cache_key]
                if tokenizer is not None:
                    logger.info(f"Using cached tokenizer for worker {worker_id}")
                    return tokenizer
                else:
                    logger.warning(f"Removing invalid tokenizer from cache for worker {worker_id}")
                    del self._local.process_tokenizers[cache_key]
                    self._local.tokenizer_refs.pop(cache_key, None)

            # Create new tokenizer
            logger.info(f"Creating new tokenizer for worker {worker_id} in process {self._local.pid}")

            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("Invalid model name provided")

            transformers_logging.set_verbosity_error()
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                if not isinstance(tokenizer, PreTrainedTokenizerFast):
                    raise RuntimeError(f"Failed to get fast tokenizer for model '{model_name}'")

            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer for model '{model_name}': {str(e)}")
            finally:
                transformers_logging.set_verbosity_warning()

            if tokenizer is None:
                raise ValueError("Tokenizer creation failed")

            # Cache tokenizer
            self._local.process_tokenizers[cache_key] = tokenizer
            self._local.tokenizer_refs[cache_key] = tokenizer

            logger.info(
                f"Successfully created tokenizer for worker {worker_id} "
                f"in process {self._local.pid}"
            )
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