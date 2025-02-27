from __future__ import annotations

import logging
import os
import tempfile
import threading
import traceback  # Add this missing import
from pathlib import Path
from typing import Dict, Any, Optional

from src.common.managers.base_manager import BaseManager

logger = logging.getLogger(__name__)

class DirectoryManager(BaseManager):
    """Manager for handling directory paths and creation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the DirectoryManager.
        
        Args:
            config: Application configuration
        """
        super().__init__(config)

    def _initialize_process_local(self, config: Dict[str, Any]) -> None:
        """Initialize process-local state for directory management."""
        try:
            logger.info("Starting DirectoryManager initialization")
            logger.info(f"Configuration: {config}")
            # Set process ID first before any other operations
            current_pid = os.getpid()
            if not hasattr(self._local, 'pid'):
                self._local.pid = current_pid
            
            # Call parent initialization - this might clear thread local attributes
            super()._initialize_process_local(config)
            
            # Debug log to track values
            logger.debug(f"DirectoryManager initializing for process {current_pid}, has base_dir: {hasattr(self._local, 'base_dir')}")
            
            # Store config paths section for consistent reference
            paths_config = config.get('paths', {})
            if not paths_config:
                logger.warning("No paths configuration found, using defaults")
                paths_config = {
                    'base_dir': os.getcwd(),
                    'output_dir': 'output',
                    'data_dir': 'data',
                    'cache_dir': '.cache',
                    'model_dir': 'models'
                }
                
            # First set all attributes safely
            self._set_all_directory_attributes(paths_config)
            
            # Then create directories in a separate loop
            self._create_all_directories()
            
            # Set initialized flag at the end
            self._local.initialized = True
            
            logger.info(f"DirectoryManager initialized for process {getattr(self._local, 'pid', current_pid)}")
            logger.info(f"DirectoryManager state: {self._local.__dict__}")
            
        except Exception as e:
            # On error, make sure to set initialized to False
            self._local.initialized = False
            logger.error(f"Failed to initialize DirectoryManager: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise

    def _set_directory_attribute(self, attr_name: str, dir_path: str) -> None:
        """Set a directory attribute safely, resolving relative paths.
        
        Args:
            attr_name: The attribute name to set
            dir_path: The directory path value
        """
        logger.info(f"Setting directory attribute {attr_name} to {dir_path}")
        # Make sure base_dir exists
        if not hasattr(self._local, 'base_dir'):
            self._local.base_dir = os.getcwd()
            logger.warning(f"Base directory not set, using current directory: {self._local.base_dir}")
        
        # Set the attribute, handling relative vs absolute paths
        if not os.path.isabs(dir_path):
            setattr(self._local, attr_name, os.path.join(self._local.base_dir, dir_path))
        else:
            setattr(self._local, attr_name, dir_path)
        
        logger.debug(f"Set {attr_name} to {getattr(self._local, attr_name)}")
        logger.info(f"DirectoryManager state after setting {attr_name}: {self._local.__dict__}")

    def _create_directory_if_set(self, attr_name: str) -> None:
        """Create a directory if the attribute exists."""
        try:
            # Get a snapshot of current attributes to avoid race conditions
            local_attrs = vars(self._local) if hasattr(self._local, '__dict__') else {}
            
            if attr_name in local_attrs:
                dir_path = local_attrs[attr_name]
                logger.debug(f"Creating directory: {dir_path}")
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Successfully created directory: {dir_path}")
            else:
                logger.warning(f"Directory attribute {attr_name} not set, cannot create directory")
                # Emergency recovery - set a default path and create it
                if hasattr(self._local, 'base_dir'):
                    default_dir = attr_name.replace('_dir', '')  # output_dir -> output
                    default_path = os.path.join(self._local.base_dir, default_dir)
                    logger.warning(f"Setting default path for {attr_name}: {default_path}")
                    setattr(self._local, attr_name, default_path)
                    os.makedirs(default_path, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory for {attr_name}: {str(e)}")

    def _set_all_directory_attributes(self, paths_config: Dict[str, str]) -> None:
        """Set all directory attributes in one pass to avoid race conditions."""
        try:
            # Set base directory first - everything depends on this
            base_dir_path = paths_config.get('base_dir', os.getcwd())
            self._local.base_dir = os.path.abspath(str(base_dir_path))
            logger.debug(f"Set base_dir to {self._local.base_dir}")
            
            # Create default directory paths that are guaranteed to exist
            default_dirs = {
                'output_dir': 'output',
                'data_dir': 'data',
                'cache_dir': '.cache',
                'model_dir': 'models'
            }
            
            # For each directory attribute, set it directly without separate method call
            for dir_attr, default_path in default_dirs.items():
                path_value = paths_config.get(dir_attr.replace('_dir', ''), default_path)
                # Handle absolute vs. relative paths
                if not os.path.isabs(path_value):
                    full_path = os.path.join(self._local.base_dir, path_value)
                else:
                    full_path = path_value
                    
                # Set the attribute
                setattr(self._local, dir_attr, full_path)
                logger.debug(f"Set {dir_attr} to {full_path}")
                
            # Set temp dir last since it's special
            self._local.temp_dir = tempfile.mkdtemp()
            logger.debug(f"Set temp_dir to {self._local.temp_dir}")
            
            # Log all attributes to verify
            logger.info(f"Directory attributes set: {self._local.__dict__}")
        except Exception as e:
            logger.error(f"Error in _set_all_directory_attributes: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _create_all_directories(self) -> None:
        """Create all directories at once."""
        try:
            logger.info("Starting directory creation")
            
            # Define all required directory attributes first
            required_dirs = {
                'base_dir': 'base',
                'output_dir': 'output', 
                'data_dir': 'data', 
                'cache_dir': 'cache', 
                'model_dir': 'model'  # Fixed: use 'model' not 'models'
            }
            
            # ALWAYS ensure base_dir is set first
            if not hasattr(self._local, 'base_dir'):
                self._local.base_dir = os.getcwd()
                logger.warning(f"Created missing base_dir: {self._local.base_dir}")
            
            # Now create ALL other directories from the required list
            for attr_name, dir_name in required_dirs.items():
                if attr_name != 'base_dir':  # We already handled base_dir
                    if not hasattr(self._local, attr_name):
                        path = os.path.join(self._local.base_dir, dir_name)
                        setattr(self._local, attr_name, path)
                        logger.warning(f"Created missing {attr_name}: {path}")
            
            # Create special temp_dir if missing
            if not hasattr(self._local, 'temp_dir'):
                self._local.temp_dir = tempfile.mkdtemp()
                logger.warning(f"Created missing temp_dir: {self._local.temp_dir}")
            
            # Verify all required attributes now exist
            for attr_name in required_dirs.keys():
                if not hasattr(self._local, attr_name):
                    logger.critical(f"CRITICAL: Failed to create {attr_name} attribute")
                    # Use emergency recovery if this happens
                    self._emergency_directory_recovery()
                    return
            
            # Now create all physical directories
            for attr_name, _ in required_dirs.items():
                dir_path = getattr(self._local, attr_name)
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    logger.info(f"Created directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Failed to create directory {dir_path}: {e}")
                    # Continue with other directories
        
        except Exception as e:
            logger.error(f"Error in _create_all_directories: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Use emergency recovery
            self._emergency_directory_recovery()

    def _emergency_directory_recovery(self) -> None:
        """Emergency recovery for directory creation failures."""
        try:
            logger.critical("Performing emergency directory recovery")
            base_dir = os.getcwd()
            
            # Set required attributes directly (avoid __dict__ access)
            self._local.base_dir = base_dir
            self._local.output_dir = os.path.join(base_dir, 'output')
            self._local.data_dir = os.path.join(base_dir, 'data')
            self._local.cache_dir = os.path.join(base_dir, 'cache')
            self._local.model_dir = os.path.join(base_dir, 'models')
            self._local.temp_dir = tempfile.mkdtemp()
            
            # Create directories
            for attr_name in ['output_dir', 'data_dir', 'cache_dir', 'model_dir']:
                dir_path = getattr(self._local, attr_name)
                os.makedirs(dir_path, exist_ok=True)
                logger.critical(f"Emergency created directory: {dir_path}")
                
        except Exception as e:
            logger.critical(f"Even emergency recovery failed: {e}")

    def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Set up the directory manager for a new process.
        
        Args:
            config: Configuration dictionary or None to use the stored config
        """
        current_pid = os.getpid()
        if not hasattr(self, '_local') or not hasattr(self._local, 'pid') or self._local.pid != current_pid:
            logger.info(f"Re-initializing DirectoryManager for process {current_pid}")
            self._local = threading.local()
            # Set pid immediately to avoid any race conditions
            self._local.pid = current_pid
            self._local.initialized = False
            
            # Initialize with provided config or stored config
            effective_config = config if config is not None else self._config
            
            # Ensure we have a paths section in the config to avoid KeyError
            if 'paths' not in effective_config:
                logger.warning(f"No 'paths' section found in config, adding default paths")
                effective_config['paths'] = {
                    'base_dir': os.getcwd(),
                    'output_dir': 'output',
                    'data_dir': 'data',
                    'cache_dir': '.cache',
                    'model_dir': 'models'
                }
            
            self._initialize_process_local(effective_config)
            logger.debug(f"DirectoryManager setup completed for process {current_pid}")
        else:
            logger.debug(f"DirectoryManager already set up for process {current_pid}")

    @property
    def base_dir(self) -> Path:
        """Get the base directory as a Path object.
        
        Returns:
            Path: Base directory path
        """
        self.ensure_initialized()
        return Path(self._local.base_dir)
        
    @property
    def output_dir(self) -> Path:
        """Get the output directory as a Path object.
        
        Returns:
            Path: Output directory path
        """
        # Add safety check in case output_dir doesn't exist
        if not hasattr(self, '_local'):
            self._local = threading.local()
            self._local.pid = os.getpid()
        
        if not hasattr(self._local, 'output_dir'):
            logger.warning(f"output_dir not set in DirectoryManager, using default")
            self._local.output_dir = os.path.join(os.getcwd(), 'output')
            os.makedirs(self._local.output_dir, exist_ok=True)
        
        return Path(self._local.output_dir)
        
    @property
    def data_dir(self) -> Path:
        """Get the data directory as a Path object.
        
        Returns:
            Path: Data directory path
        """
        self.ensure_initialized()
        return Path(self._local.data_dir)
        
    @property
    def cache_dir(self) -> Path:
        """Get the cache directory as a Path object.
        
        Returns:
            Path: Cache directory path
        """
        self.ensure_initialized()
        return Path(self._local.cache_dir)
        
    @property
    def model_dir(self) -> Path:
        """Get the model directory as a Path object.
        
        Returns:
            Path: Model directory path
        """
        self.ensure_initialized()
        return Path(self._local.model_dir)
        
    @property
    def temp_dir(self) -> Path:
        """Get the temp directory as a Path object.
        
        Returns:
            Path: Temp directory path
        """
        self.ensure_initialized()
        return Path(self._local.temp_dir)
        
    def create_directory(self, path: str, base_dir: Optional[str] = None) -> str:
        """Create a directory.
        
        Args:
            path: The directory path to create
            base_dir: Optional base directory path
            
        Returns:
            str: The full path to the created directory
        """
        self.ensure_initialized()
        
        if base_dir is None:
            base_dir = self._local.base_dir
            
        full_path = os.path.join(base_dir, path) if not os.path.isabs(path) else path
        os.makedirs(full_path, exist_ok=True)
        
        return full_path
        
    def cleanup(self) -> None:
        """Clean up DirectoryManager resources."""
        try:
            # Clean up temp directory if needed
            import shutil
            if hasattr(self._local, 'temp_dir') and os.path.exists(self._local.temp_dir):
                shutil.rmtree(self._local.temp_dir, ignore_errors=True)
                
            logger.info(f"Cleaned up DirectoryManager for process {self._local.pid}")
            super().cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up DirectoryManager: {str(e)}")
            raise

    # Remove duplicate accessor methods and keep only these ones
    def get_base_dir(self) -> Path:
        """Get the base directory as a Path object."""
        return self.base_dir
        
    def get_output_dir(self) -> Path:
        """Get the output directory as a Path object."""
        return self.output_dir
        
    def get_data_dir(self) -> Path:
        """Get the data directory as a Path object."""
        return self.data_dir
        
    def get_cache_dir(self) -> Path:
        """Get the cache directory as a Path object."""
        return self.cache_dir
        
    def get_model_dir(self) -> Path:
        """Get the model directory as a Path object."""
        return self.model_dir
        
    def get_temp_dir(self) -> Path:
        """Get the temp directory as a Path object."""
        return self.temp_dir

__all__ = ['DirectoryManager']