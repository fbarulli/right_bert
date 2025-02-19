# src/common/managers/__init__.py
from __future__ import annotations
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dependency_injector.containers import Container

logger = logging.getLogger(__name__)

def cleanup_managers(container: Optional[Container] = None):
    """
    Clean up all manager instances.
    
    Args:
        container: Optional dependency injection container. If provided,
                  will use it to get and cleanup managers in the correct order.
    """
    if container is not None:
        try:
            # Clean up in reverse dependency order
            
            # Complex managers first
            if hasattr(container, 'resource_manager'):
                container.resource_manager().cleanup()
            if hasattr(container, 'worker_manager'):
                container.worker_manager().cleanup()
            if hasattr(container, 'data_manager'):
                container.data_manager().cleanup()
            if hasattr(container, 'model_manager'):
                container.model_manager().cleanup()
            if hasattr(container, 'metrics_manager'):
                container.metrics_manager().cleanup()
            if hasattr(container, 'batch_manager'):
                container.batch_manager().cleanup()
            
            # Secondary managers
            if hasattr(container, 'amp_manager'):
                container.amp_manager().cleanup()
            if hasattr(container, 'tensor_manager'):
                container.tensor_manager().cleanup()
            if hasattr(container, 'dataloader_manager'):
                container.dataloader_manager().cleanup()
            if hasattr(container, 'storage_manager'):
                container.storage_manager().cleanup()
            
            # Primary managers last
            if hasattr(container, 'tokenizer_manager'):
                container.tokenizer_manager().cleanup()
            if hasattr(container, 'directory_manager'):
                container.directory_manager().cleanup()
            if hasattr(container, 'cuda_manager'):
                container.cuda_manager().cleanup()
            
            logger.info("Successfully cleaned up all managers using DI container")
        except Exception as e:
            logger.error(f"Error during manager cleanup: {str(e)}")
            raise
    else:
        logger.warning("No DI container provided for cleanup - some resources may not be properly released")

__all__ = ['cleanup_managers']
