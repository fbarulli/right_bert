import logging
import inspect
from typing import Dict, Any
import threading

logger = logging.getLogger(__name__)

class ContainerTracker:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.containers = {}
                cls._instance.active_container = None
            return cls._instance
    
    def register_container(self, name, container):
        """Register a container with the tracker."""
        caller = inspect.getframeinfo(inspect.currentframe().f_back)
        logger.debug(f"Container registered: {name} from {caller.filename}:{caller.lineno}")
        self.containers[name] = container
        
    def set_active_container(self, name):
        """Set which container is currently active."""
        if name not in self.containers:
            logger.error(f"Container {name} not registered")
            return
        
        caller = inspect.getframeinfo(inspect.currentframe().f_back)
        logger.debug(f"Active container set to: {name} from {caller.filename}:{caller.lineno}")
        self.active_container = name
        
    def get_container(self, name=None):
        """Get a container by name or the active container if name is None."""
        if name is None:
            name = self.active_container
            
        if name is None or name not in self.containers:
            caller = inspect.getframeinfo(inspect.currentframe().f_back)
            logger.error(f"Container {name} not found, called from {caller.filename}:{caller.lineno}")
            return None
            
        return self.containers[name]

# Create a singleton instance
container_tracker = ContainerTracker()
