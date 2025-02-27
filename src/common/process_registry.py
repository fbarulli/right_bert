"""Global process registry to track and manage process hierarchies."""
import os
import threading
import logging
import time
import atexit
import signal  # Add signal module for handling termination signals
from typing import Dict, Set, Optional, Any, Callable

logger = logging.getLogger(__name__)

# Global registry
_registry = {
    'processes': {},  # pid -> {'parent': ppid, 'type': proc_type, 'start_time': time}
    'managers': {}    # pid -> {manager_name -> instance}
}
_registry_lock = threading.RLock()

# Original signal handlers
_original_sigterm = None
_original_sigint = None

def register_process(process_type: str = 'worker', parent_pid: Optional[int] = None) -> int:
    """Register the current process in the global registry.
    
    Args:
        process_type: Type of process ('main', 'worker', 'trial', etc.)
        parent_pid: Parent process ID or None to detect automatically
        
    Returns:
        int: Current process ID
    """
    pid = os.getpid()
    ppid = parent_pid or os.getppid()
    
    with _registry_lock:
        if pid not in _registry['processes']:
            _registry['processes'][pid] = {
                'parent': ppid,
                'type': process_type,
                'start_time': time.time()
            }
            logger.debug(f"Registered process {pid} (type: {process_type}, parent: {ppid})")
    
    # Install signal handlers for graceful shutdown, but only in the main thread
    try:
        install_signal_handlers()
    except ValueError:
        # This is expected in non-main threads, just log it
        logger.debug(f"Signal handlers not installed for {process_type} in non-main thread")
            
    return pid

def register_manager(name: str, instance: Any) -> None:
    """Register a manager instance for the current process.
    
    Args:
        name: Manager name
        instance: Manager instance
    """
    pid = os.getpid()
    
    # Register process if not already registered
    if pid not in _registry['processes']:
        register_process()
    
    with _registry_lock:
        if pid not in _registry['managers']:
            _registry['managers'][pid] = {}
        _registry['managers'][pid][name] = instance
        logger.debug(f"Registered manager '{name}' for process {pid}")

def get_process_info(pid: Optional[int] = None) -> Dict:
    """Get information about a process.
    
    Args:
        pid: Process ID or None for current process
        
    Returns:
        Dict: Process information or empty dict if not found
    """
    pid = pid or os.getpid()
    with _registry_lock:
        return _registry['processes'].get(pid, {})

def get_child_processes(pid: Optional[int] = None) -> Set[int]:
    """Get child processes for the specified process.
    
    Args:
        pid: Parent process ID or None for current process
        
    Returns:
        Set[int]: Set of child process IDs
    """
    pid = pid or os.getpid()
    with _registry_lock:
        return {
            child_pid for child_pid, info in _registry['processes'].items()
            if info['parent'] == pid
        }

def cleanup_process() -> None:
    """Clean up the current process in the registry."""
    pid = os.getpid()
    
    with _registry_lock:
        if pid in _registry['managers']:
            # Clean up managers in reverse dependency order
            manager_order = [
                "DataManager", "ModelManager", "BatchManager",  # High-level
                "TokenizerManager", "TensorManager",            # Mid-level
                "CUDAManager", "AMPManager"                     # Core
            ]
            
            # Get managers for this process
            managers = _registry['managers'][pid]
            
            # Clean up in order
            for name in manager_order:
                if name in managers:
                    try:
                        logger.debug(f"Cleaning up manager '{name}' for process {pid}")
                        managers[name].cleanup()
                    except Exception as e:
                        logger.error(f"Error cleaning up manager '{name}': {str(e)}")
            
            # Clean up any remaining managers
            for name, manager in list(managers.items()):
                if name not in manager_order:
                    try:
                        logger.debug(f"Cleaning up remaining manager '{name}' for process {pid}")
                        manager.cleanup()
                    except Exception as e:
                        logger.error(f"Error cleaning up manager '{name}': {str(e)}")
                        
            del _registry['managers'][pid]
        
        if pid in _registry['processes']:
            logger.debug(f"Unregistering process {pid}")
            del _registry['processes'][pid]

def signal_handler(signum: int, frame) -> None:
    """Handle termination signals gracefully.
    
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    pid = os.getpid()
    logger.info(f"Process {pid} received signal {signum}, performing graceful shutdown...")
    
    try:
        # Clean up process resources
        cleanup_process()
        
        # Call any previously registered signal handlers
        if signum == signal.SIGTERM and _original_sigterm is not None:
            _original_sigterm(signum, frame)
        elif signum == signal.SIGINT and _original_sigint is not None:
            _original_sigint(signum, frame)
        else:
            # Default behavior for termination signals
            logger.info(f"Process {pid} exiting due to signal {signum}")
            os._exit(128 + signum)
    except Exception as e:
        logger.error(f"Error during signal handling: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        os._exit(1)

def install_signal_handlers() -> None:
    """Install signal handlers for graceful process termination."""
    global _original_sigterm, _original_sigint
    
    # Only install signal handlers in the main thread
    import threading
    if threading.current_thread() != threading.main_thread():
        # Signal handlers can only be installed in the main thread
        logger.debug(f"Skipping signal handler installation in non-main thread")
        return
    
    # Save original handlers
    _original_sigterm = signal.getsignal(signal.SIGTERM)
    _original_sigint = signal.getsignal(signal.SIGINT)
    
    # Install new handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.debug(f"Signal handlers installed for process {os.getpid()} in main thread")

# Register cleanup handler for normal termination
atexit.register(cleanup_process)

# Initialize main process
register_process(process_type='main', parent_pid=0)
