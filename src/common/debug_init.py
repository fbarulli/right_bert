import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

def debug_initialization(config: Dict[str, Any]) -> None:
    """Debug the initialization process of all managers."""
    try:
        logger.debug("=== Debugging Manager Initialization ===")
        logger.debug(f"Process ID: {os.getpid()}")
        
        # Import managers
        from src.common.managers import (
            get_factory,
            get_cuda_manager,
            get_directory_manager,
            get_tokenizer_manager,
            get_model_manager,
            get_storage_manager,
            get_parameter_manager,
            get_optuna_manager
        )
        
        # Get each manager and log their initialization status
        factory = get_factory()
        logger.debug(f"Factory initialized: {factory.__class__.__name__}")
        
        # Test core managers
        cuda_manager = get_cuda_manager()
        logger.debug(f"CUDA Manager initialized: {cuda_manager.is_initialized()}")
        
        directory_manager = get_directory_manager() 
        logger.debug(f"Directory Manager initialized: {directory_manager.is_initialized()}")
        logger.debug(f"Directory Manager base_dir: {directory_manager.base_dir}")
        
        tokenizer_manager = get_tokenizer_manager()
        logger.debug(f"Tokenizer Manager initialized: {tokenizer_manager.is_initialized()}")
        
        # Test dependent managers
        model_manager = get_model_manager()
        logger.debug(f"Model Manager initialized: {model_manager.is_initialized()}")
        
        storage_manager = get_storage_manager()
        logger.debug(f"Storage Manager initialized: {storage_manager.is_initialized()}")
        logger.debug(f"Storage path: {storage_manager.storage_dir}")
        
        parameter_manager = get_parameter_manager()
        logger.debug(f"Parameter Manager initialized: {parameter_manager.is_initialized()}")
        
        # Test high-level managers
        try:
            optuna_manager = get_optuna_manager()
            logger.debug(f"Optuna Manager initialized: {optuna_manager.is_initialized()}")
            logger.debug(f"Optuna study name: {optuna_manager.study_name}")
        except Exception as e:
            logger.error(f"Error initializing Optuna Manager: {str(e)}")
        
        logger.debug("=== Debug Initialization Complete ===")
        
    except Exception as e:
        logger.error(f"Debug initialization failed: {str(e)}")
        logger.exception("Exception details:")
