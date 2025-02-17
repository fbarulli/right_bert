# src/common/managers/base_manager.py
import threading
import logging
import os
import weakref
from typing import Dict, Type, ClassVar, Any, Optional


logger = logging.getLogger(__name__)

def init_manager_class(cls):
    """Class decorator to initialize class-level variables."""
    cls._instances: ClassVar[Dict[Type['BaseManager'], 'BaseManager']] = weakref.WeakValueDictionary()
    cls._storage_registry: ClassVar[Dict[Type['BaseManager'], threading.local]] = {}
    return cls


@init_manager_class  # Apply the decorator
class BaseManager:
    """Base class for process-local managers with isolated storage."""

    def __new__(cls):
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
            # Note: storage_registry is already initialized by the decorator
        return cls._instances[cls]

    @property
    def _local(self) -> threading.local:
        return self.__class__._storage_registry[self.__class__]

    def ensure_initialized(self, config: Optional[Dict[str, Any]] = None) -> None:
        current_pid = os.getpid()
        if not hasattr(self._local, 'initialized') or self._local.pid != current_pid:
            logger.debug(f"Initializing {self.__class__.__name__} for process {current_pid}")
            self._local.pid = current_pid
            self._local.initialized = False  # Initialize to False
            try:
                self._initialize_process_local(config)
                self._local.initialized = True  # Only set to True on success
                logger.info(f"{self.__class__.__name__} initialized for process {current_pid}")
            except Exception as e:
                logger.error(f"Failed to initialize {self.__class__.__name__}: {str(e)}")
                if hasattr(self._local, 'initialized'):  # Prevent AttributeError
                    delattr(self._local, 'initialized')
                raise

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass  # Implementation in subclasses

    def is_initialized(self) -> bool:
        return (
            hasattr(self._local, 'initialized') and
            self._local.initialized and
            self._local.pid == os.getpid()
        )

    def get_config_section(self, config: Dict[str, Any], section: str) -> Dict[str, Any]:
        if config is None:
            return {}
        return config.get(section, {})

    @classmethod
    def cleanup_all(cls) -> None:
        for manager_cls, storage in cls._storage_registry.items():
            try:
                if hasattr(storage, 'initialized'):
                    delattr(storage, 'initialized')
                logger.debug(f"Cleaned up storage for {manager_cls.__name__}")
            except Exception as e:
                logger.error(f"Error cleaning up {manager_cls.__name__}: {str(e)}")