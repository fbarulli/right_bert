"""Base manager class for process-aware state management."""
import os
import threading
import logging
import traceback
from typing import Dict, Any, Optional, Callable, TypeVar, Generic, Type

logger = logging.getLogger(__name__)

class BaseManager:
    """Base class for all managers in the application."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base manager.
        
        Args:
            config: Optional configuration dictionary
        """
        # Always ensure we have thread-local storage right away
        if not hasattr(self, '_local'):
            self._local = threading.local()
            
        # Initialize essential attributes immediately to avoid AttributeError
        self._local.pid = os.getpid()
        self._local.initialized = False
        
        # Store the config
        self._config = config or {}
        
        try:
            # Initialize process-local state
            self._initialize_process_local(config)
        except Exception as e:
            logger.error(f"Error initializing {self.__class__.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            # Don't propagate the exception, allow object creation even with errors
            # Managers should check initialization status using is_initialized()
            
    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local state.
        
        Args:
            config: Optional configuration dictionary
        """
        self._local.pid = os.getpid()
        self._local.initialized = True
    
    def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Public method to setup or reinitialize the manager.
        
        Args:
            config: Optional configuration dictionary
        """
        try:
            # Reinitialize process-local state
            self._initialize_process_local(config)
        except Exception as e:
            logger.error(f"Error setting up {self.__class__.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def ensure_initialized(self) -> None:
        """
        Ensure that the manager is initialized for the current process.
        
        Raises:
            RuntimeError: If the manager is not initialized for the current process
        """
        # First ensure we have _local
        if not hasattr(self, '_local'):
            class_name = self.__class__.__name__
            error_msg = f"{class_name} has no _local attribute. Create it in __init__."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        # Then check if pid is correct
        if not hasattr(self._local, 'pid') or self._local.pid != os.getpid():
            class_name = self.__class__.__name__
            this_pid = os.getpid()
            local_pid = getattr(self._local, 'pid', 'None')
            error_msg = (
                f"{class_name} initialized in process {local_pid} but "
                f"accessed in process {this_pid}. Call _initialize_process_local first."
            )
            logger.error(error_msg)
            try:
                # Auto-recover by attempting reinitialization
                self._initialize_process_local(self._config)
                logger.warning(f"Auto-reinitialized {class_name} in process {this_pid}")
                return  # If recovery worked, return without raising
            except Exception as e:
                logger.error(f"Auto-recovery failed: {e}")
            # If we get here, recovery failed
            raise RuntimeError(error_msg)
            
        # Finally check if initialized flag is true
        if not hasattr(self._local, 'initialized') or not self._local.initialized:
            class_name = self.__class__.__name__
            error_msg = f"{class_name} not initialized for process {os.getpid()}. Call _initialize_process_local first."
            logger.error(error_msg)
            try:
                # Auto-recover by attempting reinitialization
                self._initialize_process_local(self._config)
                logger.warning(f"Auto-reinitialized {class_name} in process {os.getpid()}")
                return  # If recovery worked, return without raising
            except Exception as e:
                logger.error(f"Auto-recovery failed: {e}")
            # If we get here, recovery failed
            raise RuntimeError(error_msg)
    
    def is_initialized(self) -> bool:
        """
        Check if the manager is initialized for the current process.
        
        Returns:
            bool: True if the manager is initialized, False otherwise
        """
        try:
            if not hasattr(self, '_local'):
                return False
                
            if not hasattr(self._local, 'initialized'):
                return False
                
            if not hasattr(self._local, 'pid') or self._local.pid != os.getpid():
                return False
                
            return self._local.initialized
        except Exception as e:
            logger.error(f"Error checking if {self.__class__.__name__} is initialized: {str(e)}")
            return False
            
    def get_config_section(self, config: Dict[str, Any], section: str) -> Dict[str, Any]:
        """
        Get a section from the configuration with defaults.
        
        Args:
            config: The configuration dictionary
            section: The section name
            
        Returns:
            Dict[str, Any]: The section or an empty dictionary if it doesn't exist
        """
        if config and section in config:
            return config[section]
        return {}
        
    def cleanup(self) -> None:
        """Clean up the manager's resources."""
        try:
            # Mark as not initialized, but keep the pid
            if hasattr(self, '_local'):
                self._local.initialized = False
                
            logger.debug(f"Cleaned up {self.__class__.__name__} for process {os.getpid()}")
        except Exception as e:
            logger.error(f"Error cleaning up {self.__class__.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            
    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during deletion

    def _validate_dependency(self, dependency: Any, name: str) -> None:
        """
        Validate that a dependency manager is properly initialized.
        
        Args:
            dependency: The manager to validate
            name: Name of the manager for error messages
            
        Raises:
            RuntimeError: If the dependency is missing or not initialized
        """
        if dependency is None:
            raise RuntimeError(f"{name} dependency is missing")
            
        # Check if the dependency has an is_initialized method
        if hasattr(dependency, 'is_initialized') and callable(dependency.is_initialized):
            if not dependency.is_initialized():
                # Try to initialize it
                logger.warning(f"{name} is not initialized, attempting to initialize it")
                try:
                    if hasattr(dependency, '_initialize_process_local') and callable(dependency._initialize_process_local):
                        dependency._initialize_process_local(self._config)
                        # Check again
                        if not dependency.is_initialized():
                            raise RuntimeError(f"{name} failed to initialize after attempt")
                    else:
                        raise RuntimeError(f"{name} has no _initialize_process_local method")
                except Exception as e:
                    logger.error(f"Failed to initialize {name}: {e}")
                    raise RuntimeError(f"{name} must be initialized before {self.__class__.__name__}")
        else:
            # If no is_initialized method, check _local.initialized directly
            if not hasattr(dependency, '_local') or not hasattr(dependency._local, 'initialized') or not dependency._local.initialized:
                raise RuntimeError(f"{name} must be initialized before {self.__class__.__name__}")
                
        logger.debug(f"Dependency validation successful for {name}")