# src/common/process/process_utils.py
from __future__ import annotations
"""Process utility functions for managing process names, IDs, and priorities."""
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

def set_process_name(name: str) -> None:
    """Set process name."""
    try:
        import setproctitle
        setproctitle.setproctitle(name)
        logger.info(f"Set process name to: {name}")
    except ImportError:
        logger.warning("setproctitle not available, process name not set")

def get_process_name() -> str:
    """Get current process name."""
    try:
        import setproctitle
        return setproctitle.getproctitle()
    except ImportError:
        return f"Process-{os.getpid()}"

def is_main_process() -> bool:
    """Check if this is the main process."""
    return os.getpid() == os.getppid()

def get_process_id() -> int:
    """Get current process ID."""
    return os.getpid()

def get_parent_process_id() -> int:
    """Get parent process ID."""
    return os.getppid()

def set_process_priority(priority: int) -> None:
    """Set process priority (nice value)."""
    try:
        os.nice(priority)
        logger.info(f"Set process priority to: {priority}")
    except OSError as e:
        logger.warning(f"Failed to set process priority: {e}")

__all__ = [
    'set_process_name',
    'get_process_name',
    'is_main_process',
    'get_process_id',
    'get_parent_process_id',
    'set_process_priority'
]