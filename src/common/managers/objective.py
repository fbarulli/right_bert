"""Objective function utilities for optimization."""
import os
import logging
import traceback
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def ensure_trial_setup(config: Dict[str, Any]) -> None:
    """Ensure proper setup for a trial running in a child process."""
    try:
        # Import process initialization helper
        from src.common.managers.process_init import ensure_process_initialized
        
        # Initialize all managers for this process
        ensure_process_initialized(config)
        
        logger.info(f"Trial setup complete for process {os.getpid()}")
        
    except Exception as e:
        logger.error(f"Error in trial setup: {str(e)}")
        logger.error(traceback.format_exc())
        raise
