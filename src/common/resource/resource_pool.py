# src/common/resource/resource_pool.py
from __future__ import annotations
import torch
import logging
import threading
import gc
import time
import traceback
from typing import Dict, Optional

from src.common.managers.cuda_manager import CUDAManager

logger = logging.getLogger(__name__)

class ResourcePool:
    """
    Process-local CUDA resource manager.
    
    This class handles:
    - Memory allocation tracking
    - Memory limit enforcement
    - Resource cleanup
    - Thread-safe operations
    """

    def __init__(
        self,
        cuda_manager: CUDAManager,
        memory_limit_gb: float = 5.5,
        cleanup_interval: float = 0.1
    ):
        """
        Initialize ResourcePool with dependency injection.

        Args:
            cuda_manager: Injected CUDAManager instance
            memory_limit_gb: Maximum CUDA memory allowed per process (in GB)
            cleanup_interval: Minimum time between cleanups (in seconds)
        """
        self._cuda_manager = cuda_manager
        self._device = self._cuda_manager.get_device()

        # Initialize memory limits
        self._memory_limit = int(memory_limit_gb * 1024 * 1024 * 1024)
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = 0

        # Thread synchronization
        self._lock = threading.Lock()

        # Log initialization
        if self._device.type == 'cuda':
            total_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(
                f"Initialized ResourcePool:\n"
                f"- Memory limit per process: {memory_limit_gb:.2f}GB\n"
                f"- Total available memory: {total_memory/1e9:.2f}GB\n"
                f"- Cleanup interval: {cleanup_interval:.2f}s"
            )
        else:
            logger.warning("CUDA not available, using CPU")

    def check_memory(
        self,
        size_bytes: Optional[int] = None
    ) -> bool:
        """
        Check if memory usage is within limits.

        Args:
            size_bytes: Optional size of requested allocation

        Returns:
            bool: True if memory usage is within limits
        """
        if self._device.type == 'cpu':
            return True

        try:
            current_allocated = self._cuda_manager.get_memory_allocated()

            if size_bytes is not None:
                return (current_allocated + size_bytes) <= self._memory_limit

            return current_allocated <= self._memory_limit

        except Exception as e:
            logger.error(f"Error checking memory: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def request_memory(self, size_bytes: int) -> bool:
        """
        Request memory allocation.

        Args:
            size_bytes: Size of requested allocation in bytes

        Returns:
            bool: True if memory can be allocated
        """
        if self._device.type == 'cpu':
            return True

        try:
            with self._lock:
                # Check if memory is available
                if not self.check_memory(size_bytes):
                    # Try cleanup if enough time has passed
                    current_time = time.time()
                    if current_time - self._last_cleanup >= self._cleanup_interval:
                        self.cleanup()
                        self._last_cleanup = current_time

                    # Check again after cleanup
                    if not self.check_memory(size_bytes):
                        logger.warning(
                            f"Memory request for {size_bytes/1e9:.2f}GB exceeds limit:\n"
                            f"- Current allocated: {self._cuda_manager.get_memory_allocated()/1e9:.2f}GB\n"
                            f"- Memory limit: {self._memory_limit/1e9:.2f}GB"
                        )
                        return False

                return True

        except Exception as e:
            logger.error(f"Error requesting memory: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def cleanup(self) -> None:
        """Clean up CUDA memory."""
        if self._device.type == 'cpu':
            return

        try:
            # Collect Python garbage
            gc.collect()

            # Clean up CUDA memory
            self._cuda_manager.cleanup()

            # Log memory state
            allocated = self._cuda_manager.get_memory_allocated()
            reserved = self._cuda_manager.get_memory_reserved()
            logger.debug(
                f"Memory after cleanup:\n"
                f"- Allocated: {allocated/1e9:.2f}GB\n"
                f"- Reserved: {reserved/1e9:.2f}GB"
            )

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            logger.error(traceback.format_exc())

    def get_stats(self) -> Dict[str, int]:
        """
        Get current memory statistics.

        Returns:
            Dict[str, int]: Dictionary containing:
            - allocated: Currently allocated memory in bytes
            - reserved: Currently reserved memory in bytes
            - limit: Memory limit in bytes
        """
        if self._device.type == 'cpu':
            return {
                'allocated': 0,
                'reserved': 0,
                'limit': 0
            }

        try:
            return {
                'allocated': self._cuda_manager.get_memory_allocated(),
                'reserved': self._cuda_manager.get_memory_reserved(),
                'limit': self._memory_limit
            }

        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'allocated': 0,
                'reserved': 0,
                'limit': self._memory_limit
            }


__all__ = ['ResourcePool']
