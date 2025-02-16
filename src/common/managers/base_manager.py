# src/common/managers/base_manager.py
from __future__ import annotations
import threading
import logging
import os
import weakref
from typing import Dict, Type, ClassVar, Any, Optional

logger = logging.getLogger(__name__)

class BaseManager:
    """Base class for process-local managers with isolated storage."""
    _instances: ClassVar[Dict[Type['BaseManager'], 'BaseManager']] = weakref.WeakValueDictionary()
    _storage_registry: ClassVar[Dict[Type['BaseManager'], threading.local]] = {}

    def __new__(cls):
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
            cls._storage_registry[cls] = threading.local()
        return cls._instances[cls]

    @property
    def _local(self) -> threading.local:
        return self.__class__._storage_registry[self.__class__]

    def ensure_initialized(self, config: Optional[Dict[str, Any]] = None) -> None:
        current_pid = os.getpid()
        if not hasattr(self._local, 'initialized') or self._local.pid != current_pid:
            logger.debug(f"Initializing {self.__class__.__name__} for process {current_pid}")
            self._local.pid = current_pid
            self._local.initialized = False
            try:
                self._initialize_process_local(config)
                self._local.initialized = True
                logger.info(f"{self.__class__.__name__} initialized for process {current_pid}")
            except Exception as e:
                logger.error(f"Failed to initialize {self.__class__.__name__}: {str(e)}")
                if hasattr(self._local, 'initialized'):
                    delattr(self._local, 'initialized')
                raise

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes. Override in subclasses."""
        pass

    def is_initialized(self) -> bool:
        return (
            hasattr(self._local, 'initialized') and
            self._local.initialized and
            self._local.pid == os.getpid()
        )

    def get_config_section(self, config: Dict[str, Any], section: str) -> Dict[str, Any]:
        """Get a section from the config with safe defaults."""
        if config is None:
            return {}
        return config.get(section, {})

    @classmethod
    def cleanup_all(cls) -> None:
        """Clean up all manager instances."""
        for manager_cls, storage in cls._storage_registry.items():
            try:
                if hasattr(storage, 'initialized'):
                    delattr(storage, 'initialized')
                logger.debug(f"Cleaned up storage for {manager_cls.__name__}")
            except Exception as e:
                logger.error(f"Error cleaning up {manager_cls.__name__}: {str(e)}")