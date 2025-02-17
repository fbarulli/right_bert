# src/common/managers/base_manager.py
# src/common/managers/base_manager.py (CORRECTED)
import threading
import logging
import os
import weakref
from typing import Dict, Type, ClassVar, Any, Optional

logger = logging.getLogger(__name__)

class BaseManager:
    """Base class for managers with factory-based initialization."""

    def __init__(self, factory: 'ManagerFactory', *args, **kwargs):
        """
        Initialize the manager.

        Args:
            factory: The ManagerFactory instance that created this manager.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.factory = factory
        self._storage = threading.local()
        self._initialize_process_local()

    @property
    def config(self) -> Dict[str, Any]:
        """Get the configuration from the factory."""
        return self.factory.config

    def _initialize_process_local(self) -> None:
        """Initialize process-local storage."""
        current_pid = os.getpid()
        if not hasattr(self._storage, 'pid') or self._storage.pid != current_pid:
            logger.debug(f"Initializing {self.__class__.__name__} for process {current_pid}")
            self._storage.pid = current_pid
            self._storage.initialized = False
            try:
                self._setup_process_local()
                self._storage.initialized = True
                logger.info(f"{self.__class__.__name__} initialized for process {current_pid}")
            except Exception as e:
                logger.error(f"Failed to initialize {self.__class__.__name__}: {str(e)}")
                if hasattr(self._storage, 'initialized'):
                    delattr(self._storage, 'initialized')
                raise

    def _setup_process_local(self) -> None:
        """Setup process-local resources. Override in subclasses."""
        pass

    def is_initialized(self) -> bool:
        """Check if the manager is initialized for the current process."""
        return (
            hasattr(self._storage, 'initialized') and
            self._storage.initialized and
            self._storage.pid == os.getpid()
        )

    def get_config_section(self, section: str) -> Dict[str, Any]:
        """Get a section from the configuration."""
        return self.config.get(section, {})

    def cleanup(self) -> None:
        """Clean up manager resources. Override in subclasses."""
        if hasattr(self._storage, 'initialized'):
            delattr(self._storage, 'initialized')
        logger.debug(f"Cleaned up {self.__class__.__name__}")