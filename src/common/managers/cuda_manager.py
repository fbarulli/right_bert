# src/common/managers/cuda_manager.py
# src/common/managers/cuda_manager.py (FINAL CORRECTED)

import os
import torch
import logging
import gc
import weakref
import traceback
from typing import Dict, Any, Optional
from contextlib import contextmanager

from src.common.managers.base_manager import BaseManager

logger = logging.getLogger(__name__)

class CUDAManager(BaseManager):
    """Process-local CUDA manager."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config) # Initialize base
        self.device = None # Initialize device

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes."""
        super()._initialize_process_local(config)
        try:

            logger.info("Initializing CUDAManager for process %s", os.getpid())
            # Initialize CUDA if available
            if torch.cuda.is_available():
                try:
                    # Attempt to initialize CUDA explicitly. This is crucial.
                    torch.cuda.init()  # Explicitly initialize
                    self.device = torch.device("cuda")
                    logger.info(f"CUDA initialized successfully for process {os.getpid()}.  Device: {self.device}")
                except Exception as e:
                    logger.error(f"CUDA initialization failed: {e}")
                    self.device = torch.device("cpu")  # Fallback to CPU
                    logger.warning(f"Falling back to CPU for process {os.getpid()}.")
            else:
                self.device = torch.device("cpu")
                logger.warning(f"CUDA not available, using CPU for process {os.getpid()}.")
        except Exception as e:
            logger.critical(f"Failed to initialize CUDAManager: {e}")
            raise

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def get_device(self) -> torch.device:
        self.ensure_initialized()
        if self.device is None:
            if self.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        return self.device

    def setup(self, config: Dict[str, Any]) -> None:
        self.ensure_initialized()

        if not self.is_available():
            logger.warning("CUDA not available, running on CPU")
            return

        device = self.get_device()
        torch.cuda.set_device(device.index) # type: ignore

        # These calls should happen *after* initialization and setting the device.
        torch.cuda.reset_peak_memory_stats() # type: ignore
        torch.cuda.reset_max_memory_allocated() # type: ignore

        logger.info(f"CUDA setup complete on {device}")

    def log_memory_stats(self) -> None:
        self.ensure_initialized()
        if self.is_available():
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()

            allocated_gb = allocated / 1024**3
            cached_gb = cached / 1024**3

            logger.info(
                f"CUDA Memory: {allocated_gb:.2f}GB allocated, "
                f"{cached_gb:.2f}GB cached"
            )

    def cleanup(self) -> None:
        """Clean up CUDA memory and resources."""
        self.ensure_initialized()
        try:
            if self.is_available():
                # Clear CUDA memory
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats() # type: ignore
                torch.cuda.reset_max_memory_allocated() # type: ignore

        except Exception as e:
            logger.error(f"Error during CUDA cleanup: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    @contextmanager
    def track_memory(self, tag: str = '') -> None:
        """Context manager to track memory usage."""
        self.ensure_initialized()
        if self.is_available():
            torch.cuda.reset_peak_memory_stats()
            try:
                yield
            finally:
                peak = torch.cuda.max_memory_allocated() / 1024**3 # type: ignore
                logger.info(f"Peak memory usage{f' [{tag}]' if tag else ''}: {peak:.2f}GB")

__all__ = ['CUDAManager']