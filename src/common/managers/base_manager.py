# src/common/managers/base_manager.py (CORRECTED)
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

@init_manager_class
class BaseManager:
    """Base class for process-local managers with isolated storage."""

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
            cls._storage_registry[cls] = threading.local()  # Initialize storage here!
        return cls._instances[cls]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Don't re-initialize if it's already initialized for this process
        if not self.is_initialized():
            self._initialize_process_local(config)  # Call only if not initialized

    @property
    def _local(self) -> threading.local:
      if self.__class__ not in self.__class__._storage_registry:
          self.__class__._storage_registry[self.__class__] = threading.local()  # type: ignore
      return self.__class__._storage_registry[self.__class__]  # type: ignore

    def ensure_initialized(self, config: Optional[Dict[str, Any]] = None) -> None:
        current_pid = os.getpid()
        if not hasattr(self._local, 'initialized') or self._local.pid != current_pid:
            logger.debug(f"Initializing {self.__class__.__name__} for process {current_pid}")
            self._local.pid = current_pid
            self._local.initialized = False
            try:
                # Do NOT call _initialize_process_local here.  It's handled in __init__.
                self._local.initialized = True  # Only set to True on success
                logger.info(f"{self.__class__.__name__} initialized for process {current_pid}")
            except Exception as e:
                logger.error(f"Failed to initialize {self.__class__.__name__}: {str(e)}")
                if hasattr(self._local, 'initialized'):
                    delattr(self._local, 'initialized')
                raise

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local storage with configuration."""
        if not hasattr(self._local, 'config'):
            self._local.config = {}
        
        if config is not None:
            self._local.config.update(config)
            
        self._validate_config(self._local.config)
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration. Override in subclasses for specific validation."""
        if not config:
            raise ValueError("Configuration cannot be empty")

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
