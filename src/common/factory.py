import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

def get_factory():
    """Get the dependency injection container."""
    logger.debug("=== Factory Creation Debug ===")
    if not hasattr(get_factory, '_factory'):
        logger.debug("Creating new factory instance")
        from src.common.containers import Container
        container = Container()
        container.config.from_yaml('config/embedding_config.yaml')
        logger.debug(f"Loaded config: {container.config()}")
        
        # Register with container tracker
        from src.common.container_debug import container_tracker
        container_tracker.register_container("factory", container)
        container_tracker.set_active_container("factory")
        
        get_factory._factory = container
    return get_factory._factory
