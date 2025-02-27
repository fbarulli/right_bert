from __future__ import annotations
import os
import threading
import logging
import traceback
import multiprocessing as mp
from typing import Dict, Any, Optional, Set

logger = logging.getLogger(__name__)

class BaseProcessManager:
    """Base class for process-aware manager objects.
    
    This class properly handles initialization across process boundaries
    when using multiprocessing, particularly in 'spawn' mode.
    """
    
    # Track initialization status across processes
    _process_registry = {}  # Process ID -> {manager class -> initialization status}
    _registry_lock = threading.RLock()
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the manager.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config or {}
        self._local = threading.local()
        self._local.pid = os.getpid()
        self._local.initialized = False
        
        # Register this process-manager combo
        self._register_process()
        
    def _register_process(self) -> None:
        """Register this manager instance with the current process."""
        pid = os.getpid()
        cls_name = self.__class__.__name__
        
        with self._registry_lock:
            if pid not in self._process_registry:
                self._process_registry[pid] = {}
            self._process_registry[pid][cls_name] = False  # Not initialized yet
    
    def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Setup manager for the current process.
        
        This method should be called in each process that uses the manager.
        
        Args:
            config: Optional configuration to use (falls back to stored config)
        """
        current_pid = os.getpid()
        if not hasattr(self._local, 'pid') or self._local.pid != current_pid:
            logger.debug(f"{self.__class__.__name__}: Process change detected: {getattr(self._local, 'pid', 'Unknown')} -> {current_pid}")
            self._local = threading.local()
            self._local.pid = current_pid
            self._local.initialized = False
            self._register_process()
        
        if not self._local.initialized:
            try:
                logger.debug(f"Initializing {self.__class__.__name__} in process {current_pid}")
                effective_config = config if config is not None else self._config
                self._initialize_process_local(effective_config)
                self._local.initialized = True
                
                # Update registry
                with self._registry_lock:
                    if current_pid in self._process_registry:
                        self._process_registry[current_pid][self.__class__.__name__] = True
                
                logger.info(f"{self.__class__.__name__} initialized for process {current_pid}")
            except Exception as e:
                logger.error(f"Failed to initialize {self.__class__.__name__}: {str(e)}")
                logger.error(traceback.format_exc())
                raise
    
    def _initialize_process_local(self, config: Dict[str, Any]) -> None:
        """Initialize process-local state.
        
        Override this in subclasses to handle process-specific initialization.
        
        Args:
            config: Configuration dictionary
        """
        self._local.config = config
        self._local.pid = os.getpid()
    
    def is_initialized(self) -> bool:
        """Check if the manager is initialized in the current process.
        
        Returns:
            bool: True if initialized
        """
        return hasattr(self._local, 'initialized') and self._local.initialized
    
    def ensure_initialized(self) -> None:
        """Ensure the manager is initialized.
        
        Raises:
            RuntimeError: If not initialized
        """
        if not self.is_initialized():
            current_pid = os.getpid()
            error_msg = (
                f"{self.__class__.__name__} not initialized for process {current_pid}. "
                f"Call setup() first."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def auto_setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Auto-initialize if needed and warn about it.
        
        Args:
            config: Optional configuration dictionary
        """
        if not self.is_initialized():
            logger.warning(f"Auto-initializing {self.__class__.__name__} in process {os.getpid()}")
            self.setup(config)
    
    @classmethod
    def log_process_registry(cls) -> None:
        """Log the current process registry status."""
        with cls._registry_lock:
            process_count = len(cls._process_registry)
            logger.info(f"Process Registry Status: {process_count} processes tracked")
            
            for pid, managers in cls._process_registry.items():
                logger.info(f"  Process {pid}:")
                for manager, status in managers.items():
                    status_str = "Initialized" if status else "Not Initialized"
                    logger.info(f"    - {manager}: {status_str}")
    
    def cleanup(self) -> None:
        """Clean up manager resources.
        
        Override in subclasses for proper cleanup.
        """
        current_pid = os.getpid()
        with self._registry_lock:
            if current_pid in self._process_registry:
                if self.__class__.__name__ in self._process_registry[current_pid]:
                    self._process_registry[current_pid][self.__class__.__name__] = False
        
        self._local.initialized = False
        logger.info(f"Cleaned up {self.__class__.__name__} for process {current_pid}")

__all__ = ['BaseProcessManager']
