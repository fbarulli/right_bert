# src/common/resource/resource_initializer.py
# src/common/resource/resource_initializer.py (CORRECTED)
from __future__ import annotations
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ResourceInitializer:
    """Initializes and cleans up process-local resources."""

    _config: Optional[Dict[str, Any]] = None  # Class-level config

    @classmethod
    def initialize_process(cls, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local resources."""
        if config is not None:
            cls._config = config # Store the config

        if cls._config is None:
            raise RuntimeError("ResourceInitializer config not set.")

        # Initialize core managers (order matters)
        from src.common.managers import (
            get_cuda_manager,
            get_amp_manager,
            get_data_manager,
            get_dataloader_manager,
            get_tensor_manager,
            get_tokenizer_manager,
            get_model_manager,
            get_metrics_manager,
            get_parameter_manager,
            get_storage_manager,
            get_directory_manager,
            get_worker_manager,
            get_wandb_manager,
            get_optuna_manager
        )

        # Initialize CUDA *first*
        cuda_manager = get_cuda_manager()
        cuda_manager.ensure_initialized(cls._config)

        # Now initialize other managers
        amp_manager = get_amp_manager()
        amp_manager.ensure_initialized(cls._config)
        data_manager = get_data_manager()
        data_manager.ensure_initialized(cls._config)
        dataloader_manager = get_dataloader_manager()
        dataloader_manager.ensure_initialized(cls._config)
        tensor_manager = get_tensor_manager()
        tensor_manager.ensure_initialized(cls._config)
        tokenizer_manager = get_tokenizer_manager()
        tokenizer_manager.ensure_initialized(cls._config)
        model_manager = get_model_manager()
        model_manager.ensure_initialized(cls._config)
        metrics_manager = get_metrics_manager()
        metrics_manager.ensure_initialized(cls._config)
        parameter_manager = get_parameter_manager()
        parameter_manager.ensure_initialized(cls._config)
        directory_manager = get_directory_manager()
        directory_manager.ensure_initialized(cls._config)
        storage_manager = get_storage_manager()
        storage_manager.ensure_initialized(cls._config)
        wandb_manager = get_wandb_manager()
        wandb_manager.ensure_initialized(cls._config)
        optuna_manager = get_optuna_manager()
        optuna_manager.ensure_initialized(cls._config)
        worker_manager = get_worker_manager()
        worker_manager.ensure_initialized(cls._config)


        logger.info("Process resources initialized.")


    @classmethod
    def cleanup_process(cls) -> None:
        """Clean up process-local resources."""
        from src.common.managers import (
            get_cuda_manager,
            get_amp_manager,
            get_data_manager,
            get_dataloader_manager,
            get_tensor_manager,
            get_tokenizer_manager,
            get_model_manager,
            get_metrics_manager,
            get_parameter_manager,
            get_storage_manager,
            get_directory_manager,
            get_wandb_manager,
            get_optuna_manager,
            get_worker_manager
        )
        # Clean up managers (reverse order of initialization)
        try:
            worker_manager = get_worker_manager()
            worker_manager.cleanup_workers()
        except Exception as e:
            logger.warning(f"Error during worker manager cleanup: {e}")
        try:
            optuna_manager = get_optuna_manager()
            optuna_manager.cleanup()
        except Exception as e:
            logger.warning(f"Error during optuna manager cleanup: {e}")
        try:
            wandb_manager = get_wandb_manager()
            wandb_manager.finish()
        except Exception as e:
            logger.warning(f"Error during wandb manager cleanup")
        try:
            storage_manager = get_storage_manager()
            storage_manager.cleanup_all()
        except Exception as e:
            logger.warning(f"Error during storage manager cleanup: {e}")

        try:
            directory_manager = get_directory_manager()
            directory_manager.cleanup_all()
        except Exception as e:
             logger.warning(f"Error during directory manager cleanup: {e}")
        #No cleanup needed for parameter manager

        try:
            metrics_manager = get_metrics_manager()
        except Exception as e:
             logger.warning(f"Error during metrics manager cleanup: {e}")
        try:
            model_manager = get_model_manager()
            model_manager.cleanup_all()
        except Exception as e:
             logger.warning(f"Error during model manager cleanup: {e}")
        try:
            tokenizer_manager = get_tokenizer_manager()
            tokenizer_manager.cleanup_all() # Cleanup tokenizer cache
        except Exception as e:
             logger.warning(f"Error during tokenizer manager cleanup: {e}")
        try:
            tensor_manager = get_tensor_manager()
            tensor_manager.clear_memory()
        except Exception as e:
            logger.warning(f"Error during tensor manager cleanup: {e}")
        try:
            dataloader_manager = get_dataloader_manager()
        except Exception as e:
             logger.warning(f"Error during dataloader manager cleanup: {e}")
        try:
            data_manager = get_data_manager()
        except Exception as e:
             logger.warning(f"Error during data manager cleanup: {e}")

        try:
            amp_manager = get_amp_manager()
        except Exception as e:
            logger.warning(f"Error during AMP manager cleanup: {e}")

        try:
            cuda_manager = get_cuda_manager()
            cuda_manager.cleanup()

        except Exception as e:
            logger.warning(f"Error during CUDA manager cleanup: {e}")

        logger.info("Process resources cleaned up.")