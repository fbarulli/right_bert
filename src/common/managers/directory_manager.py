# src/common/managers/directory_manager.py (CORRECTED)
from __future__ import annotations

import logging
import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Optional

from src.common.managers.base_manager import BaseManager

logger = logging.getLogger(__name__)

class DirectoryManager(BaseManager):
    """Manages directory structure for outputs and caching"""

    def __init__(self, base_dir: Path, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the DirectoryManager.

        Args:
            base_dir: Base directory for outputs and caching.
            config: Configuration dictionary (optional).  Not used by DirectoryManager.
        """
        self.base_dir = base_dir   # Store base_dir FIRST
        super().__init__(config)  # Initialize with config
        if base_dir is None:
            raise ValueError("base_dir cannot be None")


    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes."""
        super()._initialize_process_local(config)

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.base_dir / 'cache'
        self.mmap_dir = self.base_dir / 'mmap'
        self.storage_dir = self.base_dir / 'storage' # Ensure storage_dir is initialized
        self.cache_dir.mkdir(exist_ok=True)
        self.mmap_dir.mkdir(exist_ok=True)
        self.storage_dir.mkdir(exist_ok=True) # Ensure storage_dir exists

        logger.debug(f"Created directory structure at {self.base_dir}")

    def get_db_path(self) -> Path:
        """Get path to optuna database."""
        return self.storage_dir / 'optuna.db' # Use self.storage_dir

    def get_history_path(self) -> Path:
        """Get path to trial history file."""
        return self.base_dir / 'trial_history.json'

    def get_cache_path(self, data_path: Path, prefix: str = '') -> Path:
        """Get cache path for a data file."""
        hasher = hashlib.sha256()
        hasher.update(str(data_path).encode())
        if data_path.exists():
            hasher.update(str(data_path.stat().st_mtime).encode())
        cache_hash = hasher.hexdigest()[:16]

        return self.cache_dir / f"{prefix}_{cache_hash}.pt"

    def get_mmap_path(self, array_name: str, trial_number: Optional[int] = None) -> Path:
        """Get path for memory-mapped array."""
        if trial_number is not None:
            return self.mmap_dir / f"trial_{trial_number}_{array_name}.mmap"
        return self.mmap_dir / f"{array_name}.mmap"

    def cleanup_cache(self, older_than_days: Optional[float] = None) -> None:
        """Clean up old cache files."""
        try:
            if older_than_days is not None:
                import time
                cutoff = time.time() - (older_than_days * 24 * 60 * 60)
                for path in self.cache_dir.glob('*.pt'):
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                        logger.debug(f"Deleted old cache file: {path}")
            else:
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
                logger.debug("Cleared cache directory")
        except Exception as e:
            logger.warning(f"Error cleaning cache: {e}")

    def cleanup_mmap(self, trial_number: Optional[int] = None) -> None:
        """Clean up memory-mapped files."""
        try:
            if trial_number is not None:
                pattern = f"trial_{trial_number}_*.mmap"
            else:
                pattern = "*.mmap"

            for path in self.mmap_dir.glob(pattern):
                try:
                    path.unlink()
                except Exception as e:
                    logger.warning(f"Error deleting mmap file {path}: {e}")

            logger.debug(f"Cleaned up mmap files{f' for trial {trial_number}' if trial_number else ''}")

        except Exception as e:
            logger.warning(f"Error cleaning mmap directory: {e}")


    def cleanup_all(self) -> None:
        """Clean up all temporary files."""
        try:
            self.cleanup_cache()
            self.cleanup_mmap()
            logger.debug("Cleaned up all temporary files")
        except Exception as e:
            logger.warning(f"Error in cleanup: {e}")

__all__ = ['DirectoryManager']