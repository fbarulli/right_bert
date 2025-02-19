
# src/common/managers/cuda_manager.py
from __future__ import annotations
import os
import torch
import logging
import gc
import traceback
from typing import Dict, Any, Optional
from contextlib import contextmanager

from src.common.managers.base_manager import BaseManager

logger = logging.getLogger(__name__)

class CUDAManager(BaseManager):
    """
    Process-local CUDA manager.

    This manager handles:
    - CUDA initialization and device management
    - Memory tracking and cleanup
    - Device selection and configuration
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CUDAManager.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self._local.device = None

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local attributes.

        Args:
            config: Optional configuration dictionary that overrides the one from constructor
        """
        try:
            super()._initialize_process_local(config)
            logger.info(f"Initializing CUDAManager for process {self._local.pid}")

            if torch.cuda.is_available():
                try:
                    # Explicitly initialize CUDA
                    torch.cuda.init()

                    # Get device configuration
                    effective_config = config if config is not None else self._config
                    device_id = None
                    if effective_config:
                        resources_config = self.get_config_section(effective_config, 'resources')
                        if 'gpu_device_id' in resources_config:
                            device_id = resources_config['gpu_device_id']

                    # Set device
                    if device_id is not None and device_id < torch.cuda.device_count():
                        self._local.device = torch.device(f"cuda:{device_id}")
                    else:
                        self._local.device = torch.device("cuda")

                    # Configure device
                    torch.cuda.set_device(self._local.device)
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_max_memory_allocated()

                    logger.info(
                        f"CUDA initialized successfully for process {self._local.pid} "
                        f"using device: {self._local.device}"
                    )

                except Exception as e:
                    logger.error(f"CUDA initialization failed: {e}")
                    logger.error(traceback.format_exc())
                    self._local.device = torch.device("cpu")
                    logger.warning(f"Falling back to CPU for process {self._local.pid}")
            else:
                self._local.device = torch.device("cpu")
                logger.warning(f"CUDA not available, using CPU for process {self._local.pid}")

        except Exception as e:
            logger.critical(f"Failed to initialize CUDAManager: {e}")
            logger.error(traceback.format_exc())
            raise

    def is_available(self) -> bool:
        """
        Check if CUDA is available.

        Returns:
            bool: True if CUDA is available, False otherwise
        """
        return torch.cuda.is_available()

    def get_device(self) -> torch.device:
        """
        Get the current device.

        Returns:
            torch.device: The current device (CUDA or CPU)
        """
        self.ensure_initialized()
        return self._local.device

    def setup(self, config: Dict[str, Any]) -> None:
        """
        Setup CUDA environment with configuration.

        Args:
            config: Configuration dictionary
        """
        self.ensure_initialized()

        if not self.is_available():
            logger.warning("CUDA not available, running on CPU")
            return

        try:
            # Apply any CUDA-specific configuration
            if 'resources' in config:
                resources_config = config['resources']
                if 'gpu_memory_gb' in resources_config:
                    max_memory = int(resources_config['gpu_memory_gb'] * 1024 * 1024 * 1024)  # Convert GB to bytes
                    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                        total_memory = torch.cuda.get_device_properties(self._local.device).total_memory
                        fraction = min(max_memory / total_memory, 1.0)
                        torch.cuda.set_per_process_memory_fraction(fraction, self._local.device)
                        logger.info(f"Set GPU memory limit to {resources_config['gpu_memory_gb']}GB")

            logger.info(f"CUDA setup complete on {self._local.device}")

        except Exception as e:
            logger.error(f"Error during CUDA setup: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def log_memory_stats(self) -> None:
        """Log current CUDA memory statistics."""
        self.ensure_initialized()
        if self.is_available():
            try:
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                max_allocated = torch.cuda.max_memory_allocated()

                logger.info(
                    f"\nCUDA Memory Stats for process {self._local.pid}:"
                    f"\n- Allocated: {allocated / 1024**3:.2f}GB"
                    f"\n- Reserved: {reserved / 1024**3:.2f}GB"
                    f"\n- Peak: {max_allocated / 1024**3:.2f}GB"
                )
            except Exception as e:
                logger.error(f"Error logging memory stats: {str(e)}")

    def cleanup(self) -> None:
        """Clean up CUDA memory and resources."""
        try:
            if self.is_available():
                # Log final memory stats
                self.log_memory_stats()

                # Clear CUDA memory
                torch.cuda.empty_cache()
                gc.collect()

                # Reset memory stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_max_memory_allocated()

                logger.info(f"Cleaned up CUDA resources for process {self._local.pid}")

            # Reset device
            if hasattr(self._local, 'device'):
                self._local.device = None

            super().cleanup()

        except Exception as e:
            logger.error(f"Error during CUDA cleanup: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    @contextmanager
    def track_memory(self, tag: str = '') -> None:
        """
        Context manager to track memory usage.

        Args:
            tag: Optional tag for memory tracking log messages
        """
        self.ensure_initialized()
        if self.is_available():
            try:
                # Reset peak stats at start
                torch.cuda.reset_peak_memory_stats()
                start_allocated = torch.cuda.memory_allocated()

                yield

                # Log memory usage
                peak = torch.cuda.max_memory_allocated()
                end_allocated = torch.cuda.memory_allocated()
                logger.info(
                    f"Memory tracking{f' [{tag}]' if tag else ''}:"
                    f"\n- Peak usage: {peak / 1024**3:.2f}GB"
                    f"\n- Net change: {(end_allocated - start_allocated) / 1024**3:.2f}GB"
                )
            except Exception as e:
                logger.error(f"Error during memory tracking: {str(e)}")
                raise
        else:
            yield


__all__ = ['CUDAManager']