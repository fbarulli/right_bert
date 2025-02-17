# src/common/process/initialization.py
# src/common/process/initialization.py
from __future__ import annotations
"""Core initialization utilities."""
import os
import logging
import multiprocessing as mp
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

def initialize_process() -> Tuple[int, int]:
    """Initialize process-specific settings.

    This function handles the core initialization sequence for any process.
    It uses lazy imports to avoid circular dependencies.

    Returns:
        Tuple of (current_pid, parent_pid)

    Raises:
        RuntimeError: If initialization fails
    """
    try:
        current_pid = os.getpid()
        parent_pid = os.getppid()

        # Set spawn method for any sub-processes
        if mp.get_start_method(allow_none=True) != 'spawn':
            try:
                mp.set_start_method('spawn', force=True)
                logger.info("Set multiprocessing start method to 'spawn'")
            except RuntimeError as e:
                logger.warning(f"Could not set spawn method: {e}")

        # Import managers at runtime to avoid circular imports
        from src.common.managers import (
            get_cuda_manager,
            get_amp_manager
        )

        # Initialize CUDA first
        cuda_manager = get_cuda_manager()
        cuda_manager.ensure_initialized()
        logger.debug(f"CUDA initialized for process {current_pid}")

        # Initialize AMP right after CUDA
        amp_manager = get_amp_manager()
        amp_manager.ensure_initialized()
        logger.debug(f"AMP initialized for process {current_pid}")

        # Initialize remaining resources
        from src.common.resource.resource_initializer import ResourceInitializer
        ResourceInitializer.initialize_process()
        logger.debug(f"Resources initialized for process {current_pid}")

        return current_pid, parent_pid

    except Exception as e:
        logger.error(f"Failed to initialize process {os.getpid()}: {str(e)}")
        raise

def cleanup_process() -> None:
    """Clean up process-specific resources.

    This function handles the cleanup sequence for any process.
    It uses lazy imports to avoid circular dependencies.
    """
    try:
        from src.common.resource.resource_initializer import ResourceInitializer
        ResourceInitializer.cleanup_process()
        logger.info(f"Process {os.getpid()} resources cleaned up successfully")

    except Exception as e:
        logger.error(f"Error cleaning up process {os.getpid()}: {str(e)}")
        raise

def get_worker_init_fn(num_workers: int) -> Optional[callable]:
    """Get worker initialization function if needed."""
    if num_workers > 0:
        return initialize_worker # We changed the logic here
    return None

def initialize_worker(worker_id: int) -> None:
    """Initialize worker process, same logic as the main process."""
    try:
        logger.info(f"Initializing worker {worker_id} (PID: {os.getpid()})")
        initialize_process()
        logger.info(f"Worker {worker_id} initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize worker {worker_id}: {str(e)}")
        raise

__all__ = [ 'initialize_worker',
            'initialize_process',
            'cleanup_process',
            'get_worker_init_fn']