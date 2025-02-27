from __future__ import annotations
import threading
import logging
import os  # Ensure this import is present
from typing import Dict, Any, Optional
from abc import ABC

logger = logging.getLogger(__name__)

class BaseManager(ABC):
    """
    Base class for managers with dependency injection support.

    This class provides:
    - Process-local storage management
    - Initialization tracking
    - Configuration handling
    - Resource cleanup
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the manager."""
        logger.debug(f"{self.__class__.__name__}.__init__ called with config keys={config.keys() if config else None}")
        
        self._config = config
        if not hasattr(self, '_local'):
            logger.debug(f"{self.__class__.__name__} creating new _local")
            self._local = threading.local()
        
        self._local.pid = os.getpid()
        self._local.initialized = False
        
        logger.debug(f"{self.__class__.__name__} created for PID {self._local.pid}")
        
        try:
            logger.debug(f"Calling _setup_process_local for {self.__class__.__name__}")
            self._setup_process_local(config)
            logger.debug(f"_setup_process_local completed for {self.__class__.__name__}")
            
            logger.debug(f"Calling _initialize_process_local for {self.__class__.__name__}")
            self._initialize_process_local(config)
            # Only mark as initialized if no exception was raised
            self._local.initialized = True
            logger.debug(f"_initialize_process_local completed for {self.__class__.__name__}")
        except Exception as e:
            self._local.initialized = False
            logger.error(f"Failed to initialize {self.__class__.__name__}: {str(e)}", exc_info=True)
            raise

    def _setup_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Setup process-local resources. Override in subclasses.

        Args:
            config: Optional configuration dictionary
        """
        pass

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local storage.

        Args:
            config: Optional configuration dictionary that overrides the one from constructor
        """
        current_pid = os.getpid()
        logger.debug(f"=== {self.__class__.__name__} Initialization Debug ===")
        logger.debug(f"Process ID: {current_pid}")
        logger.debug(f"Has _local: {hasattr(self, '_local')}")
        logger.debug(f"Config: {config}")
        
        if not hasattr(self._local, 'pid') or self._local.pid != current_pid:
            logger.debug(f"Initializing {self.__class__.__name__} for process {current_pid}")
            self._local.pid = current_pid
            self._local.initialized = True
            logger.info(f"{self.__class__.__name__} initialized for process {current_pid}")

    def is_initialized(self) -> bool:
        """
        Check if the manager is initialized for the current process.

        Returns:
            bool: True if initialized, False otherwise
        """
        has_local = hasattr(self, '_local')
        has_initialized = has_local and hasattr(self._local, 'initialized')
        is_init = has_initialized and self._local.initialized
        has_pid = has_local and hasattr(self._local, 'pid')
        right_pid = has_pid and self._local.pid == os.getpid()
        
        logger.debug(f"{self.__class__.__name__} initialization check: "
                    f"has_local={has_local}, "
                    f"has_initialized={has_initialized}, "
                    f"is_init={is_init}, "
                    f"has_pid={has_pid}, "
                    f"right_pid={right_pid}")
        
        return has_local and has_initialized and is_init and has_pid and right_pid

    def ensure_initialized(self) -> None:
        """
        Ensure the manager is initialized for the current process.

        Raises:
            RuntimeError: If the manager is not initialized
        """
        if not self.is_initialized():
            error_msg = (
                f"{self.__class__.__name__} not initialized for process {os.getpid()}. "
                f"Call _initialize_process_local first."
            )
            raise RuntimeError(error_msg)

    def _validate_dependency(self, manager: BaseManager, name: str) -> None:
        """Validate that a dependent manager is properly initialized.
        
        Args:
            manager: Manager instance to validate
            name: Name of the manager for error messages
            
        Raises:
            RuntimeError: If manager is not initialized
        """
        if not manager.is_initialized():
            raise RuntimeError(f"{name} must be initialized before {self.__class__.__name__}")

    def get_config_section(self, config: Optional[Dict[str, Any]], section: str) -> Dict[str, Any]:
        """
        Get a section from the configuration.

        Args:
            config: Configuration dictionary to use, or None to use the one from constructor
            section: Name of the configuration section to get

        Returns:
            Dict[str, Any]: Configuration section, or empty dict if not found
        """
        effective_config = config if config is not None else self._config
        if effective_config is None:
            return {}
        return effective_config.get(section, {})

    def cleanup(self) -> None:
        """Clean up manager resources.
        
        Override in subclasses for proper cleanup.
        """
        try:
            # Safe way to get process ID
            pid = getattr(self._local, 'pid', os.getpid()) if hasattr(self, '_local') else os.getpid()
            
            if hasattr(self, '_local'):
                # Reset initialization state
                self._local.initialized = False
                
            logger.info(f"Base cleanup for {self.__class__.__name__} in process {pid}")
        except Exception as e:
            logger.debug(f"Error in base manager cleanup: {e}")

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during deletion