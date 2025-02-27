"""Persistent directory handling that works even when managers fail."""
import os
from pathlib import Path
import tempfile
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PersistentDirectoryManager:
    """
    Static directory utility that works without initialization.
    Use this only as a last resort when regular managers fail.
    """
    
    @staticmethod
    def get_directories(config: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
        """
        Get standard directories without needing initialization.
        
        Args:
            config: Optional config to read paths from
            
        Returns:
            Dictionary of Path objects for standard directories
        """
        # Default to current directory
        base_dir = Path(os.getcwd())
        
        # Try to use config if provided
        if config and 'paths' in config:
            paths = config['paths']
            if 'base_dir' in paths:
                base_dir = Path(paths['base_dir'])
                
        # Create standard directories
        directories = {
            'base_dir': base_dir,
            'output_dir': base_dir / 'output',
            'data_dir': base_dir / 'data',
            'cache_dir': base_dir / 'cache',
            'model_dir': base_dir / 'model',
            'temp_dir': Path(tempfile.mkdtemp())
        }
        
        # Create directories
        for name, path in directories.items():
            try:
                if name != 'temp_dir':  # temp_dir already exists
                    os.makedirs(path, exist_ok=True)
                logger.info(f"PersistentDirectoryManager created {name}: {path}")
            except Exception as e:
                logger.error(f"Failed to create directory {path}: {e}")
                
        return directories
