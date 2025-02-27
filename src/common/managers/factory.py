from __future__ import annotations
import os
import sys
import logging
import threading
import traceback
from typing import Dict, Any, Optional, Type, Callable

from src.common.managers.base_process_manager import BaseProcessManager
from src.common.managers.process_manager_debug import process_debugger, register_manager

logger = logging.getLogger(__name__)

class ManagerFactory:
    """Factory for creating and managing manager instances across processes."""
    
    def __init__(self):
        """Initialize the manager factory."""
        self._managers = {}
        self._config = {}
        self._manager_classes = {}
        self._dependencies = {}
        self._lock = threading.RLock()
    
    def register_manager_class(
        self, 
        name: str, 
        manager_class: Type[BaseProcessManager],
        dependencies: Optional[list] = None
    ) -> None:
        """Register a manager class with the factory.
        
        Args:
            name: Manager name
            manager_class: Manager class
            dependencies: List of manager names this manager depends on
        """
        with self._lock:
            self._manager_classes[name] = manager_class
            self._dependencies[name] = dependencies or []
            logger.debug(f"Registered manager class: {name}")
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set the configuration for all managers.
        
        Args:
            config: Configuration dictionary
        """
        with self._lock:
            self._config = config.copy()
    
    def get_manager(self, name: str, auto_create: bool = True) -> Optional[BaseProcessManager]:
        """Get a manager instance by name, optionally creating it if it doesn't exist.
        
        Args:
            name: Manager name
            auto_create: Whether to automatically create the manager if it doesn't exist
            
        Returns:
            Optional[BaseProcessManager]: The manager instance or None
        """
        pid = os.getpid()
        manager_key = f"{name}_{pid}"
        
        # Try to get existing manager for this process
        with self._lock:
            if manager_key in self._managers:
                manager = self._managers[manager_key]
                # Ensure the manager is initialized in this process
                if isinstance(manager, BaseProcessManager) and not manager.is_initialized():
                    logger.debug(f"Auto-initializing existing manager {name} in process {pid}")
                    manager.setup(self._config)
                return manager
        
        # Create manager if needed and auto_create is True
        if auto_create:
            return self.create_manager(name)
        
        return None
    
    def create_manager(self, name: str) -> BaseProcessManager:
        """Create a manager instance.
        
        Args:
            name: Manager name
            
        Returns:
            BaseProcessManager: The created manager
            
        Raises:
            ValueError: If the manager class is not registered
            RuntimeError: If a dependency cannot be created
        """
        with self._lock:
            if name not in self._manager_classes:
                raise ValueError(f"Manager class {name} not registered")
            
            pid = os.getpid()
            manager_key = f"{name}_{pid}"
            
            # Check if manager already exists for this process
            if manager_key in self._managers:
                return self._managers[manager_key]
            
            # Create dependencies first
            dependencies = {}
            for dep_name in self._dependencies.get(name, []):
                logger.debug(f"Creating dependency {dep_name} for {name}")
                dep = self.create_manager(dep_name)
                dependencies[dep_name] = dep
            
            # Create the manager
            try:
                logger.debug(f"Creating manager {name} in process {pid}")
                manager_class = self._manager_classes[name]
                
                # Pass config and dependencies to constructor
                manager = manager_class(self._config, **dependencies)
                
                # Initialize the manager
                manager.setup(self._config)
                
                # Store the manager
                self._managers[manager_key] = manager
                
                # Register with debugger
                register_manager(name, manager)
                
                logger.info(f"Created and initialized manager {name} in process {pid}")
                return manager
            except Exception as e:
                logger.error(f"Error creating manager {name}: {str(e)}")
                logger.error(traceback.format_exc())
                raise
    
    def cleanup_manager(self, name: str) -> None:
        """Clean up a manager and its instances.
        
        Args:
            name: Manager name
        """
        with self._lock:
            pid = os.getpid()
            manager_key = f"{name}_{pid}"
            
            if manager_key in self._managers:
                try:
                    logger.debug(f"Cleaning up manager {name} in process {pid}")
                    manager = self._managers[manager_key]
                    manager.cleanup()
                    del self._managers[manager_key]
                except Exception as e:
                    logger.error(f"Error cleaning up manager {name}: {str(e)}")
    
    def cleanup_all(self) -> None:
        """Clean up all manager instances in the current process."""
        pid = os.getpid()
        logger.debug(f"Cleaning up all managers in process {pid}")
        
        # Get all manager keys for this process
        with self._lock:
            manager_keys = [k for k in self._managers if k.endswith(f"_{pid}")]
            
            # Clean up in reverse dependency order
            cleaned = set()
            
            # First pass: clean up managers without dependencies on others
            for key in manager_keys:
                name = key.rsplit('_', 1)[0]
                if not self._dependencies.get(name, []):
                    try:
                        manager = self._managers[key]
                        manager.cleanup()
                        cleaned.add(key)
                        logger.debug(f"Cleaned up independent manager {name}")
                    except Exception as e:
                        logger.error(f"Error cleaning up manager {name}: {str(e)}")
            
            # Second pass: clean up remaining managers
            remaining = [k for k in manager_keys if k not in cleaned]
            for key in remaining:
                if key not in cleaned:
                    try:
                        name = key.rsplit('_', 1)[0]
                        manager = self._managers[key]
                        manager.cleanup()
                        cleaned.add(key)
                        logger.debug(f"Cleaned up dependent manager {name}")
                    except Exception as e:
                        name = key.rsplit('_', 1)[0]
                        logger.error(f"Error cleaning up manager {name}: {str(e)}")
            
            # Remove cleaned managers
            for key in cleaned:
                del self._managers[key]

# Global factory instance
manager_factory = ManagerFactory()

def get_manager(name: str, auto_create: bool = True) -> Optional[BaseProcessManager]:
    """Get a manager by name.
    
    Args:
        name: Manager name
        auto_create: Whether to create the manager if it doesn't exist
        
    Returns:
        Optional[BaseProcessManager]: The manager instance or None
    """
    return manager_factory.get_manager(name, auto_create)

def initialize_factory(config: Dict[str, Any]) -> None:
    """Initialize the manager factory with configuration.
    
    Args:
        config: Configuration dictionary
    """
    manager_factory.set_config(config)
    logger.info(f"Manager factory initialized in process {os.getpid()}")

def cleanup_all_managers() -> None:
    """Clean up all manager instances in the current process."""
    manager_factory.cleanup_all()
    logger.info(f"All managers cleaned up in process {os.getpid()}")

__all__ = [
    'manager_factory',
    'get_manager',
    'initialize_factory',
    'cleanup_all_managers'
]
