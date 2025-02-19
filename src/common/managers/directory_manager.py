# src/common/managers/directory_manager.py
from __future__ import annotations
import logging
import os
import shutil
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional

from src.common.managers.base_manager import BaseManager

logger = logging.getLogger(__name__)

class DirectoryManager(BaseManager):
    """
    Manages directory structure for outputs and caching.
    
    This manager handles:
    - Directory creation and structure
    - Cache management
    - Memory-mapped file paths
    - Storage paths for databases and history
    """

    def __init__(
        self,
        base_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize DirectoryManager.

        Args:
            base_dir: Base directory for outputs and caching
            config: Optional configuration dictionary

        Raises:
            ValueError: If base_dir is None
        """
        if base_dir is None:
            raise ValueError("base_dir cannot be None")
        
        self.base_dir = Path(base_dir)
        super().__init__(config)
        self._local.initialized_dirs = set()

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local attributes.

        Args:
            config: Optional configuration dictionary that overrides the one from constructor
        """
        try:
            super()._initialize_process_local(config)

            # Create directory structure
            self.base_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize subdirectories
            self._local.cache_dir = self.base_dir / 'cache'
            self._local.mmap_dir = self.base_dir / 'mmap'
            self._local.storage_dir = self.base_dir / 'storage'
            self._local.output_dir = self.base_dir / 'output'

            # Create all directories
            for dir_path in [
                self._local.cache_dir,
                self._local.mmap_dir,
                self._local.storage_dir,
                self._local.output_dir
            ]:
                dir_path.mkdir(exist_ok=True)
                self._local.initialized_dirs.add(dir_path)

            logger.info(
                f"DirectoryManager initialized for process {self._local.pid} "
                f"with base directory: {self.base_dir}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize DirectoryManager: {str(e)}")
            logger.error(f"Base directory: {self.base_dir}")
            raise

    def get_db_path(self) -> Path:
        """
        Get path to optuna database.

        Returns:
            Path: Path to the database file
        """
        self.ensure_initialized()
        return self._local.storage_dir / 'optuna.db'

    def get_history_path(self) -> Path:
        """
        Get path to trial history file.

        Returns:
            Path: Path to the history file
        """
        self.ensure_initialized()
        return self._local.output_dir / 'trial_history.json'

    def get_cache_path(self, data_path: Path, prefix: str = '') -> Path:
        """
        Get cache path for a data file.

        Args:
            data_path: Path to the original data file
            prefix: Optional prefix for the cache file name

        Returns:
            Path: Path to the cache file
        """
        self.ensure_initialized()
        try:
            # Create hash from file path and modification time
            hasher = hashlib.sha256()
            hasher.update(str(data_path).encode())
            if data_path.exists():
                hasher.update(str(data_path.stat().st_mtime).encode())
            cache_hash = hasher.hexdigest()[:16]

            return self._local.cache_dir / f"{prefix}_{cache_hash}.pt"

        except Exception as e:
            logger.error(f"Error getting cache path for {data_path}: {str(e)}")
            raise

    def get_mmap_path(self, array_name: str, trial_number: Optional[int] = None) -> Path:
        """
        Get path for memory-mapped array.

        Args:
            array_name: Name of the array
            trial_number: Optional trial number to include in the path

        Returns:
            Path: Path to the memory-mapped file
        """
        self.ensure_initialized()
        if trial_number is not None:
            return self._local.mmap_dir / f"trial_{trial_number}_{array_name}.mmap"
        return self._local.mmap_dir / f"{array_name}.mmap"

    def cleanup_cache(self, older_than_days: Optional[float] = None) -> None:
        """
        Clean up old cache files.

        Args:
            older_than_days: Optional number of days, files older than this will be deleted
        """
        self.ensure_initialized()
        try:
            if older_than_days is not None:
                cutoff = time.time() - (older_than_days * 24 * 60 * 60)
                deleted_count = 0
                for path in self._local.cache_dir.glob('*.pt'):
                    try:
                        if path.stat().st_mtime < cutoff:
                            path.unlink()
                            deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Error deleting cache file {path}: {str(e)}")
                logger.info(f"Deleted {deleted_count} old cache files")
            else:
                shutil.rmtree(self._local.cache_dir)
                self._local.cache_dir.mkdir(exist_ok=True)
                logger.info("Cleared entire cache directory")

        except Exception as e:
            logger.error(f"Error cleaning cache: {str(e)}")
            raise

    def cleanup_mmap(self, trial_number: Optional[int] = None) -> None:
        """
        Clean up memory-mapped files.

        Args:
            trial_number: Optional trial number to clean up specific trial files
        """
        self.ensure_initialized()
        try:
            pattern = f"trial_{trial_number}_*.mmap" if trial_number is not None else "*.mmap"
            deleted_count = 0

            for path in self._local.mmap_dir.glob(pattern):
                try:
                    path.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Error deleting mmap file {path}: {str(e)}")

            logger.info(
                f"Cleaned up {deleted_count} mmap files"
                f"{f' for trial {trial_number}' if trial_number else ''}"
            )

        except Exception as e:
            logger.error(f"Error cleaning mmap directory: {str(e)}")
            raise

    def cleanup(self) -> None:
        """Clean up all directory manager resources."""
        try:
            # Clean up all temporary files
            self.cleanup_cache()
            self.cleanup_mmap()

            # Clear initialized directories set
            self._local.initialized_dirs.clear()

            logger.info(f"Cleaned up DirectoryManager for process {self._local.pid}")
            super().cleanup()

        except Exception as e:
            logger.error(f"Error cleaning up DirectoryManager: {str(e)}")
            raise


__all__ = ['DirectoryManager']
