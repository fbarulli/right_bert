# src/common/managers/cuda_manager.py (Corrected import)
from __future__ import annotations
import os
import torch
import logging
import gc
import weakref
import traceback
from typing import Dict, Any, Optional
from contextlib import contextmanager
from .base_manager import BaseManager


logger = logging.getLogger(__name__)

class CUDAManager(BaseManager):
    """Process-local CUDA manager."""
    
    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes."""
        try:
            super()._initialize_process_local(config)
            
            logger.info("Initializing CUDAManager for process %s", os.getpid())
            
            self._local.memory_allocated = 0.0
            self._local.memory_cached = 0.0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_max_memory_allocated()
                
            self._local.device = None
            self._local.models = weakref.WeakSet()
            
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                logger.info(
                    f"Initial CUDA memory state for process {self._local.pid}:\n"
                    f"- Allocated: {allocated:.2f}GB\n"
                    f"- Cached: {cached:.2f}GB"
                )
            
            self._local.settings_initialized = True
            logger.info("CUDA manager initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize CUDA manager: {str(e)}")
            logger.error(traceback.format_exc())
            if hasattr(self._local, 'settings_initialized'):
                delattr(self._local, 'settings_initialized')
            raise
            
    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()
            
    def get_device(self) -> torch.device:
        """Get current device."""
        self.ensure_initialized()
        if self._local.device is None:
            if self.is_available():
                self._local.device = torch.device('cuda:0')
            else:
                self._local.device = torch.device('cpu')
        return self._local.device
            
    def setup(self, config: Dict[str, Any]) -> None:
        """Setup CUDA environment."""
        self.ensure_initialized()
            
        if not self.is_available():
            logger.warning("CUDA not available, running on CPU")
            return
            
        device = self.get_device()
        torch.cuda.set_device(device.index)
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        self._local.memory_allocated = 0.0
        self._local.memory_cached = 0.0
        
        logger.info(f"CUDA setup complete on {device}")
            
    def log_memory_stats(self) -> None:
        """Log CUDA memory statistics."""
        self.ensure_initialized()
        if self.is_available():
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            
            allocated_gb = allocated / 1024**3
            cached_gb = cached / 1024**3
            
            allocated_diff = allocated_gb - self._local.memory_allocated
            cached_diff = cached_gb - self._local.memory_cached
            
            logger.info(
                f"CUDA Memory: {allocated_gb:.2f}GB allocated ({allocated_diff:+.2f}GB), "
                f"{cached_gb:.2f}GB cached ({cached_diff:+.2f}GB)"
            )
            
            self._local.memory_allocated = allocated_gb
            self._local.memory_cached = cached_gb
            
    def is_initialized(self) -> bool:
        """Check if manager is fully initialized."""
        return (
            super().is_initialized() and
            hasattr(self._local, 'settings_initialized') and
            self._local.settings_initialized
        )

    def cleanup(self) -> None:
        """Clean up CUDA memory and resources."""
        self.ensure_initialized()
        try:
            if self.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_max_memory_allocated()
                
                if hasattr(self._local, 'models'):
                    self._local.models.clear()
                
                self._local.memory_allocated = 0.0
                self._local.memory_cached = 0.0
                
                logger.info("CUDA resources cleaned up")
            
            if hasattr(self._local, 'settings_initialized'):
                delattr(self._local, 'settings_initialized')
                
        except Exception as e:
            logger.error(f"Error during CUDA cleanup: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def register_model(self, model: torch.nn.Module) -> None:
        """Register model for CUDA memory tracking."""
        self.ensure_initialized()
        self._local.models.add(model)
        logger.debug(f"Registered model {id(model)} for CUDA tracking")

    def unregister_model(self, model: torch.nn.Module) -> None:
        """Unregister model from CUDA memory tracking."""
        self.ensure_initialized()
        self._local.models.discard(model)
        logger.debug(f"Unregistered model {id(model)} from CUDA tracking")

    def verify_cuda_state(self) -> None:
        """Verify CUDA memory state is clean."""
        self.ensure_initialized()
        if self.is_available():
            self.log_memory_stats()
            allocated = torch.cuda.memory_allocated()
            if allocated > 0:
                logger.warning(
                    f"CUDA memory not clean: {allocated / 1024**3:.2f}GB allocated. "
                    "This may indicate a memory leak from previous operations."
                )

    @contextmanager
    def track_memory(self, tag: str = '') -> None:
        """Context manager to track memory usage."""
        self.ensure_initialized()
        if self.is_available():
            torch.cuda.reset_peak_memory_stats()
            try:
                yield
            finally:
                peak = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"Peak memory usage{f' [{tag}]' if tag else ''}: {peak:.2f}GB")

__all__ = ['CUDAManager']