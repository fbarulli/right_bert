# src/common/managers/tokenizer_manager.py
# src/common/managers/tokenizer_manager.py (FINAL CORRECTED)
from __future__ import annotations
import logging
import os
import weakref
from typing import Dict, Any, Optional
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from transformers.utils import logging as transformers_logging
import threading

from .base_manager import BaseManager
# from src.common import get_shared_tokenizer, set_shared_tokenizer  # Import Moved down

logger = logging.getLogger(__name__)

class TokenizerManager(BaseManager):
    """Process-local tokenizer manager."""

    _shared_tokenizer = None  # Class-level shared tokenizer
    _tokenizer_lock = threading.Lock()  # Lock for accessing the shared tokenizer


    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Tokenizer Manager"""
        super().__init__(config) # Initialize base.
        # self.process_tokenizers = {}   # Moved to _initialize_process_local
        # self.tokenizer_refs = weakref.WeakValueDictionary() # Moved

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        super()._initialize_process_local(config)
        logger.info(f"Initializing TokenizerManager for process {os.getpid()}")
        self._local.process_tokenizers = {}
        self._local.tokenizer_refs = weakref.WeakValueDictionary()


    @classmethod
    def set_shared_tokenizer(cls, tokenizer: PreTrainedTokenizerFast):
        """Set the shared tokenizer instance."""
        with cls._tokenizer_lock:
            cls._shared_tokenizer = tokenizer

    @classmethod
    def get_shared_tokenizer(cls) -> Optional[PreTrainedTokenizerFast]:
        """Get the shared tokenizer instance."""
        with cls._tokenizer_lock:
            return cls._shared_tokenizer


    def get_worker_tokenizer(self, worker_id: int, model_name: str, model_type: str = 'embedding', config: Optional[Dict[str, Any]] = None) -> PreTrainedTokenizerFast:

        # self.ensure_initialized() # Removed. Called by get_shared.

        logger.debug(f"get_worker_tokenizer called from process {os.getpid()}")
        from src.common import get_shared_tokenizer, set_shared_tokenizer  # Import Moved here
        # --- Check for shared tokenizer FIRST ---
        shared_tokenizer = get_shared_tokenizer()
        if shared_tokenizer is not None:
            logger.info(f"Using shared tokenizer for worker {worker_id}")
            return shared_tokenizer

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

        logger.info(f"Creating new tokenizer for worker {worker_id} in process {os.getpid()}")

        try:
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

        except Exception as e:
            logger.error(f"Error creating tokenizer in process {os.getpid()}: {str(e)}")
            raise

        self._local.process_tokenizers[cache_key] = tokenizer
        self._local.tokenizer_refs[cache_key] = tokenizer
        logger.info(f"Successfully created tokenizer for worker {worker_id} in process {os.getpid()}")
        return tokenizer

    def cleanup_worker(self, worker_id: int):
        # self.ensure_initialized() # Removed

        try:
            keys_to_remove = [k for k in self._local.process_tokenizers.keys() if k.startswith(f"{worker_id}_")]
            for key in keys_to_remove:
                if key in self._local.process_tokenizers:
                    del self._local.process_tokenizers[key]
                    self._local.tokenizer_refs.pop(key, None)

            logger.info(f"Cleaned up tokenizer resources for worker {worker_id} in process {os.getpid()}")

        except Exception as e:
            logger.error(f"Error during worker cleanup: {str(e)}")
__all__ = ['TokenizerManager']