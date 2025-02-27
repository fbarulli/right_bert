"""Utilities for safe process and signal handling."""
import os
import sys
import signal
import logging
import atexit
import time
from typing import Dict, List, Set, Optional, Callable, Any

logger = logging.getLogger(__name__)

# Track child processes and registered cleanup handlers
_child_processes: Set[int] = set()
_cleanup_handlers: List[Callable[[], None]] = []
_shutdown_in_progress = False

def register_child_process(pid: int) -> None:
    """
    Register a child process for tracking.
    
    Args:
        pid: Process ID of child
    """
    _child_processes.add(pid)
    logger.debug(f"Registered child process {pid}, total tracked: {len(_child_processes)}")
    
def unregister_child_process(pid: int) -> None:
    """
    Unregister a child process.
    
    Args:
        pid: Process ID of child
    """
    if pid in _child_processes:
        _child_processes.remove(pid)
        logger.debug(f"Unregistered child process {pid}, remaining: {len(_child_processes)}")
        
def register_cleanup_handler(handler: Callable[[], None]) -> None:
    """
    Register a cleanup handler to be called on exit.
    
    Args:
        handler: Cleanup function
    """
    _cleanup_handlers.append(handler)
    logger.debug(f"Registered cleanup handler {handler.__name__}, total handlers: {len(_cleanup_handlers)}")
    
def cleanup_all() -> None:
    """Execute all registered cleanup handlers."""
    global _shutdown_in_progress
    
    if _shutdown_in_progress:
        return
        
    _shutdown_in_progress = True
    logger.info("Running cleanup handlers")
    
    # Execute cleanup handlers in reverse order
    for handler in reversed(_cleanup_handlers):
        try:
            handler()
        except Exception as e:
            logger.error(f"Error in cleanup handler {handler.__name__}: {e}")
            
    # Terminate any remaining child processes
    for pid in list(_child_processes):
        try:
            logger.warning(f"Terminating child process {pid}")
            os.kill(pid, signal.SIGTERM)
            # Give process a chance to terminate
            time.sleep(0.1)
        except ProcessLookupError:
            # Process already terminated
            pass
        except Exception as e:
            logger.error(f"Error terminating process {pid}: {e}")
            
    _shutdown_in_progress = False
    
def init_signal_handlers() -> None:
    """Initialize signal handlers for graceful shutdown."""
    # Register the cleanup function to run at exit
    atexit.register(cleanup_all)
    
    # Set up signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        cleanup_all()
        sys.exit(0)
        
    # Register for common termination signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register SIGCHLD handler to clean up child processes
    if hasattr(signal, 'SIGCHLD'):  # Not available on Windows
        def handle_sigchld(sig, frame):
            # Reap zombie processes
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid > 0:
                    unregister_child_process(pid)
                    logger.debug(f"Child process {pid} exited with status {status}")
            except ChildProcessError:
                pass
                
        signal.signal(signal.SIGCHLD, handle_sigchld)
        
    logger.info("Signal handlers initialized")
