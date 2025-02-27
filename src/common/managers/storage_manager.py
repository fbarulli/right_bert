# src/common/managers/storage_manager.py
from __future__ import annotations
import logging
import os
import threading
import shutil
import json
import sqlite3
import pickle  # Add missing import
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.common.managers.base_manager import BaseManager
from src.common.managers.directory_manager import DirectoryManager

logger = logging.getLogger(__name__)

class StorageManager(BaseManager):
    """Manager for handling storage operations like saving and loading files."""

    def __init__(self, directory_manager: DirectoryManager, config: Dict[str, Any]):
        """Initialize the StorageManager.
        
        Args:
            directory_manager: The DirectoryManager to get directory paths
            config: Application configuration
        """
        self._directory_manager = directory_manager
        super().__init__(config)

    def _initialize_process_local(self, config: Dict[str, Any]) -> None:
        """Initialize process-local state for storage management."""
        try:
            super()._initialize_process_local(config)
            
            # Ensure directory manager is initialized
            if not self._directory_manager.is_initialized():
                raise RuntimeError("DirectoryManager must be initialized before StorageManager")
            
            # Set up the base storage directories
            self._local.output_dir = self._directory_manager.get_output_dir()
            self._local.cache_dir = self._directory_manager.get_cache_dir()
            
            # Add storage_dir for Optuna
            self._local.storage_dir = os.path.join(self._local.output_dir, "optuna_storage")
            os.makedirs(self._local.storage_dir, exist_ok=True)
            
            # Create directories if they don't exist
            os.makedirs(self._local.output_dir, exist_ok=True)
            os.makedirs(self._local.cache_dir, exist_ok=True)
            
            logger.info(f"StorageManager initialized for process {self._local.pid}")
            
        except Exception as e:
            logger.error(f"Failed to initialize StorageManager: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise

    def save_json(self, data: Dict[str, Any], filename: str, subdir: Optional[str] = None) -> str:
        """Save data as JSON.
        
        Args:
            data: The data to save
            filename: The name of the file
            subdir: Optional subdirectory under output_dir
            
        Returns:
            str: Path to the saved file
        """
        self.ensure_initialized()
        
        save_path = self._get_save_path(filename, subdir)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved JSON data to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save JSON data to {save_path}: {str(e)}")
            raise
            
    def load_json(self, filename: str, subdir: Optional[str] = None) -> Dict[str, Any]:
        """Load data from JSON.
        
        Args:
            filename: The name of the file
            subdir: Optional subdirectory under output_dir
            
        Returns:
            Dict[str, Any]: The loaded data
        """
        self.ensure_initialized()
        
        load_path = self._get_save_path(filename, subdir)
        
        try:
            with open(load_path, 'r') as f:
                data = json.load(f)
            logger.debug(f"Loaded JSON data from {load_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load JSON data from {load_path}: {str(e)}")
            raise
            
    def save_pickle(self, data: Any, filename: str, subdir: Optional[str] = None) -> str:
        """Save data as pickle.
        
        Args:
            data: The data to save
            filename: The name of the file
            subdir: Optional subdirectory under output_dir
            
        Returns:
            str: Path to the saved file
        """
        self.ensure_initialized()
        
        save_path = self._get_save_path(filename, subdir)
        
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Saved pickle data to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save pickle data to {save_path}: {str(e)}")
            raise
            
    def load_pickle(self, filename: str, subdir: Optional[str] = None) -> Any:
        """Load data from pickle.
        
        Args:
            filename: The name of the file
            subdir: Optional subdirectory under output_dir
            
        Returns:
            Any: The loaded data
        """
        self.ensure_initialized()
        
        load_path = self._get_save_path(filename, subdir)
        
        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Loaded pickle data from {load_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load pickle data from {load_path}: {str(e)}")
            raise
    
    def _get_save_path(self, filename: str, subdir: Optional[str] = None) -> str:
        """Get the full path for saving/loading files.
        
        Args:
            filename: The name of the file
            subdir: Optional subdirectory under output_dir
            
        Returns:
            str: The full path
        """
        if subdir:
            dir_path = os.path.join(self._local.output_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
            return os.path.join(dir_path, filename)
        else:
            return os.path.join(self._local.output_dir, filename)
            
    def get_output_dir(self) -> str:
        """Get the output directory path.
        
        Returns:
            str: The output directory path
        """
        self.ensure_initialized()
        return self._local.output_dir
        
    def get_cache_dir(self) -> str:
        """Get the cache directory path.
        
        Returns:
            str: The cache directory path
        """
        self.ensure_initialized()
        return self._local.cache_dir
        
    @property
    def storage_dir(self) -> str:
        """Get the storage directory path for Optuna studies.
        
        Returns:
            str: The storage directory path
        """
        self.ensure_initialized()
        return self._local.storage_dir
        
    def cleanup(self) -> None:
        """Clean up StorageManager resources."""
        try:
            # Use a safety check for pid attribute
            pid = getattr(self._local, 'pid', os.getpid()) if hasattr(self, '_local') else os.getpid()
            logger.info(f"Cleaned up StorageManager for process {pid}")
            super().cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up StorageManager: {str(e)}")
            raise

__all__ = ['StorageManager']