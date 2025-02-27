"""Emergency initialization for managers when normal initialization fails."""
import os
import logging
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def force_initialize_data_manager(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Force-initialize the data manager when normal initialization fails.
    
    Args:
        config: Optional configuration
    """
    try:
        from src.common.managers import get_data_manager, get_tokenizer_manager, get_dataloader_manager
        
        # Get managers
        data_manager = get_data_manager()
        tokenizer_manager = get_tokenizer_manager()
        dataloader_manager = get_dataloader_manager()
        
        # Force initialize tokenizer manager first (it's needed by data manager)
        if not tokenizer_manager.is_initialized():
            logger.warning(f"Force-initializing TokenizerManager for process {os.getpid()}")
            tokenizer_manager._local = threading.local()
            tokenizer_manager._local.pid = os.getpid()
            tokenizer_manager._local.initialized = True
            tokenizer_manager._local.tokenizers = {}
            tokenizer_manager._local.shared_tokenizer = None
            try:
                tokenizer_manager._initialize_process_local(config)
            except Exception as e:
                logger.error(f"Error during TokenizerManager forced initialization: {e}")
                
        # Force initialize dataloader manager
        if not dataloader_manager.is_initialized():
            logger.warning(f"Force-initializing DataLoaderManager for process {os.getpid()}")
            dataloader_manager._local = threading.local()
            dataloader_manager._local.pid = os.getpid()
            dataloader_manager._local.initialized = True
            try:
                dataloader_manager._initialize_process_local(config)
            except Exception as e:
                logger.error(f"Error during DataLoaderManager forced initialization: {e}")
        
        # Force initialize data manager
        logger.warning(f"Force-initializing DataManager for process {os.getpid()}")
        data_manager._local = threading.local()
        data_manager._local.pid = os.getpid()
        data_manager._local.initialized = True
        data_manager._local.datasets = {}
        try:
            data_manager._initialize_process_local(config)
            logger.info("DataManager forced initialization successful")
        except Exception as e:
            logger.error(f"Error during DataManager forced initialization: {e}")
            # Even if initialization fails, keep the initialized flag true
            data_manager._local.initialized = True
            
    except Exception as e:
        logger.error(f"Fatal error during emergency DataManager initialization: {e}")
        # Log full traceback
        import traceback
        logger.error(traceback.format_exc())
