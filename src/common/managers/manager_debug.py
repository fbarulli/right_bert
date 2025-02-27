import os
import logging
import threading
import traceback
from typing import Dict, Any, List, Optional
import multiprocessing as mp

logger = logging.getLogger(__name__)

class ManagerDebugger:
    """Utility class to debug manager initialization and process issues."""
    
    def __init__(self):
        """Initialize the manager debugger."""
        self._managers = {}
        self._process_info = {}
    
    def register_manager(self, name: str, manager: Any) -> None:
        """Register a manager instance for debugging.
        
        Args:
            name: Name of the manager
            manager: The manager instance
        """
        self._managers[name] = manager
        logger.debug(f"Registered manager '{name}' in process {os.getpid()}")
    
    def log_process_info(self) -> None:
        """Log information about the current process."""
        process = mp.current_process()
        thread = threading.current_thread()
        pid = os.getpid()
        ppid = os.getppid()
        
        info = {
            'pid': pid,
            'ppid': ppid,
            'process_name': process.name,
            'thread_name': thread.name,
            'thread_id': thread.ident,
        }
        
        self._process_info[pid] = info
        
        logger.info(f"""
=== Process Information ===
PID:          {pid}
Parent PID:   {ppid}
Process Name: {process.name}
Thread Name:  {thread.name}
Thread ID:    {thread.ident}
=========================
""")
    
    def check_manager_states(self) -> Dict[str, bool]:
        """Check initialization state of all managers.
        
        Returns:
            Dict[str, bool]: Mapping of manager names to initialization state
        """
        states = {}
        for name, manager in self._managers.items():
            try:
                is_init = manager.is_initialized() if hasattr(manager, 'is_initialized') else False
                states[name] = is_init
            except Exception as e:
                logger.error(f"Error checking {name} state: {e}")
                states[name] = False
        
        logger.info(f"Manager States in Process {os.getpid()}:")
        for name, state in states.items():
            logger.info(f"  - {name}: {'Initialized' if state else 'Not Initialized'}")
        
        return states
    
    def log_container_state(self, container) -> None:
        """Log the state of a dependency injection container.
        
        Args:
            container: The container to inspect
        """
        try:
            # Get list of attributes
            attrs = [attr for attr in dir(container) if not attr.startswith('_')]
            
            logger.info(f"""
=== Container State in Process {os.getpid()} ===
""")
            
            for attr in attrs:
                try:
                    value = getattr(container, attr)
                    if callable(value):
                        logger.info(f"  - {attr}: <callable>")
                    else:
                        logger.info(f"  - {attr}: {value}")
                except Exception as e:
                    logger.info(f"  - {attr}: Error: {e}")
            
            logger.info("=============================================")
            
        except Exception as e:
            logger.error(f"Error logging container state: {e}")
            logger.error(traceback.format_exc())

manager_debugger = ManagerDebugger()

def debug_current_process() -> None:
    """Log debug information about the current process."""
    manager_debugger.log_process_info()

def check_managers() -> Dict[str, bool]:
    """Check all registered managers.
    
    Returns:
        Dict[str, bool]: Manager initialization states
    """
    return manager_debugger.check_manager_states()

__all__ = ['manager_debugger', 'debug_current_process', 'check_managers']
