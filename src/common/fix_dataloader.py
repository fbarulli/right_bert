"""Utility for fixing DataLoader pickle issues."""
import logging
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def fix_dataloader_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix DataLoader configuration by forcing single-process mode.
    
    Args:
        config: Config dictionary
        
    Returns:
        Dict[str, Any]: Modified config with safe num_workers
    """
    if 'training' in config:
        if 'num_workers' in config['training']:
            original = config['training']['num_workers']
            if original > 0:
                logger.warning(f"Setting num_workers=0 to avoid pickle errors (original: {original})")
                config['training']['num_workers'] = 0
                
    return config

def create_safe_dataloader(dataset, batch_size, **kwargs):
    """
    Create a DataLoader that's safe from pickle errors.
    
    Args:
        dataset: The dataset to load
        batch_size: Batch size
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader: A safe DataLoader instance
    """
    # Force single-process mode
    kwargs['num_workers'] = 0
    
    return DataLoader(dataset=dataset, batch_size=batch_size, **kwargs)
