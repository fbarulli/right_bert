# src/common/managers/base_manager.py
from __future__ import annotations
import threading
import logging
import os
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
        """
        Initialize the manager.

        Args:
            config: Optional configuration dictionary
        """
        self._config = config
        self._local = threading.local()
        self._initialize_process_local(config)

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local storage.

        Args:
            config: Optional configuration dictionary that overrides the one from constructor
        """
        current_pid = os.getpid()
        if not hasattr(self._local, 'pid') or self._local.pid != current_pid:
            logger.debug(f"Initializing {self.__class__.__name__} for process {current_pid}")
            self._local.pid = current_pid
            self._local.initialized = False
            try:
                self._setup_process_local(config)
                self._local.initialized = True
                logger.info(f"{self.__class__.__name__} initialized for process {current_pid}")
            except Exception as e:
                logger.error(f"Failed to initialize {self.__class__.__name__}: {str(e)}")
                if hasattr(self._local, 'initialized'):
                    delattr(self._local, 'initialized')
                raise

    def _setup_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Setup process-local resources. Override in subclasses.

        Args:
            config: Optional configuration dictionary
        """
        pass

    def is_initialized(self) -> bool:
        """
        Check if the manager is initialized for the current process.

        Returns:
            bool: True if initialized, False otherwise
        """
        return (
            hasattr(self._local, 'initialized') and
            self._local.initialized and
            self._local.pid == os.getpid()
        )

    def ensure_initialized(self) -> None:
        """
        Ensure the manager is initialized for the current process.

        Raises:
            RuntimeError: If the manager is not initialized
        """
        if not self.is_initialized():
            raise RuntimeError(
                f"{self.__class__.__name__} not initialized for process {os.getpid()}. "
                "Call _initialize_process_local first."
            )

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
        """
        Clean up manager resources. Override in subclasses.
        
        This method should:
        1. Release any acquired resources
        2. Reset any process-local state
        3. Log cleanup actions
        """
        try:
            if hasattr(self._local, 'initialized'):
                delattr(self._local, 'initialized')
            logger.debug(f"Cleaned up {self.__class__.__name__} for process {self._local.pid}")
        except Exception as e:
            logger.error(f"Error cleaning up {self.__class__.__name__}: {str(e)}")
            raise

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during deletion
