"""Training utilities with robust fallback mechanisms."""
import os
import logging
import tempfile
from pathlib import Path
import threading
from typing import Dict, Any, Optional, Tuple
import traceback

logger = logging.getLogger(__name__)

def setup_metrics_directory(config: Dict[str, Any]) -> str:
    """
    Set up a metrics directory with maximum robustness.
    
    Args:
        config: The configuration dictionary
        
    Returns:
        str: Path to the metrics directory
    """
    logger.info("Setting up metrics directory with fallback mechanisms")
    metrics_dir = None
    
    # Try 1: Use DirectoryManager
    try:
        from src.common.managers import get_directory_manager
        directory_manager = get_directory_manager()
        if directory_manager.is_initialized():
            try:
                output_dir = directory_manager.output_dir
                metrics_dir = str(output_dir / "metrics")
                os.makedirs(metrics_dir, exist_ok=True)
                logger.info(f"Created metrics directory using DirectoryManager: {metrics_dir}")
                return metrics_dir
            except Exception as e:
                logger.error(f"Error accessing DirectoryManager.output_dir: {e}")
        else:
            logger.warning("DirectoryManager not initialized")
    except Exception as e:
        logger.error(f"Error using DirectoryManager: {e}")
    
    # Try 2: Use PersistentDirectoryManager
    try:
        from src.common.managers.persistent_directory_manager import PersistentDirectoryManager
        directories = PersistentDirectoryManager.get_directories(config)
        metrics_dir = str(directories['output_dir'] / "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        logger.warning(f"Created metrics directory using PersistentDirectoryManager: {metrics_dir}")
        return metrics_dir
    except Exception as e:
        logger.error(f"PersistentDirectoryManager failed: {e}")
    
    # Try 3: Use config directly
    try:
        output_dir = config.get('output', {}).get('dir', None)
        if output_dir:
            metrics_dir = os.path.join(output_dir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            logger.warning(f"Created metrics directory from config: {metrics_dir}")
            return metrics_dir
    except Exception as e:
        logger.error(f"Failed to create metrics directory from config: {e}")
    
    # Try 4: Ultimate fallback - use current working directory
    metrics_dir = os.path.join(os.getcwd(), "output", "metrics")
    try:
        os.makedirs(metrics_dir, exist_ok=True)
        logger.critical(f"Using last-resort metrics directory: {metrics_dir}")
    except Exception as e:
        # Final desperation attempt - use temp directory
        metrics_dir = os.path.join(tempfile.mkdtemp(), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        logger.critical(f"Using emergency temp metrics directory: {metrics_dir}")
    
    return metrics_dir
