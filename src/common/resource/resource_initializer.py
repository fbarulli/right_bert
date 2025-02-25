# src/common/resource/resource_initializer.py
from __future__ import annotations
import logging
import traceback
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class ResourceInitializer:
    """
    Initializes and cleans up process-local resources.

    This class handles:
    - Manager initialization order
    - Configuration management
    - Resource cleanup
    - Error handling
    """

    _config: Optional[Dict[str, Any]] = None  # Class-level config

    def __init__(
        self,
        cuda_manager=None,  # Type hints removed, will be imported lazily
        amp_manager=None,
        data_manager=None,
        dataloader_manager=None,
        tensor_manager=None,
        tokenizer_manager=None,
        model_manager=None,
        metrics_manager=None,
        parameter_manager=None,
        storage_manager=None,
        directory_manager=None,
        worker_manager=None,
        wandb_manager=None,
        optuna_manager=None
    ):
        """
        Initialize ResourceInitializer with dependency injection.

        Args:
            cuda_manager: Injected CUDAManager instance
            amp_manager: Injected AMPManager instance
            data_manager: Injected DataManager instance
            dataloader_manager: Injected DataLoaderManager instance
            tensor_manager: Injected TensorManager instance
            tokenizer_manager: Injected TokenizerManager instance
            model_manager: Injected ModelManager instance
            metrics_manager: Injected MetricsManager instance
            parameter_manager: Injected ParameterManager instance
            storage_manager: Injected StorageManager instance
            directory_manager: Injected DirectoryManager instance
            worker_manager: Injected WorkerManager instance
            wandb_manager: Optional injected WandbManager instance
            optuna_manager: Optional injected OptunaManager instance
        """
        # Store injected managers (or None if not provided)
        self._cuda_manager = cuda_manager
        self._amp_manager = amp_manager
        self._data_manager = data_manager
        self._dataloader_manager = dataloader_manager
        self._tensor_manager = tensor_manager
        self._tokenizer_manager = tokenizer_manager
        self._model_manager = model_manager
        self._metrics_manager = metrics_manager
        self._parameter_manager = parameter_manager
        self._storage_manager = storage_manager
        self._directory_manager = directory_manager
        self._worker_manager = worker_manager
        self._wandb_manager = wandb_manager
        self._optuna_manager = optuna_manager

        # Store initialization order (will populate lazily in initialize_process)
        self._initialization_order = []

    def initialize_process(self, config: Dict[str, Any]) -> None:
        """
        Initialize process-local resources.

        Args:
            config: Configuration dictionary

        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Lazy imports to avoid circular dependency
            from src.common.managers.cuda_manager import CUDAManager
            from src.common.managers.amp_manager import AMPManager
            from src.common.managers.data_manager import DataManager
            from src.common.managers.dataloader_manager import DataLoaderManager
            from src.common.managers.tensor_manager import TensorManager
            from src.common.managers.tokenizer_manager import TokenizerManager
            from src.common.managers.model_manager import ModelManager
            from src.common.managers.metrics_manager import MetricsManager
            from src.common.managers.parameter_manager import ParameterManager
            from src.common.managers.storage_manager import StorageManager
            from src.common.managers.directory_manager import DirectoryManager
            from src.common.managers.worker_manager import WorkerManager
            from src.common.managers.wandb_manager import WandbManager
            from src.common.managers.optuna_manager import OptunaManager

            # Store config at class level for worker access
            ResourceInitializer._config = config

            # Populate managers if not injected
            self._cuda_manager = self._cuda_manager or CUDAManager()
            self._amp_manager = self._amp_manager or AMPManager()
            self._data_manager = self._data_manager or DataManager()
            self._dataloader_manager = self._dataloader_manager or DataLoaderManager()
            self._tensor_manager = self._tensor_manager or TensorManager()
            self._tokenizer_manager = self._tokenizer_manager or TokenizerManager()
            self._model_manager = self._model_manager or ModelManager()
            self._metrics_manager = self._metrics_manager or MetricsManager()
            self._parameter_manager = self._parameter_manager or ParameterManager()
            self._storage_manager = self._storage_manager or StorageManager()
            self._directory_manager = self._directory_manager or DirectoryManager()
            self._worker_manager = self._worker_manager or WorkerManager()
            self._wandb_manager = self._wandb_manager or (WandbManager() if config.get('wandb') else None)
            self._optuna_manager = self._optuna_manager or (OptunaManager() if config.get('optuna') else None)

            # Define initialization order
            self._initialization_order = [
                ('CUDA', self._cuda_manager),
                ('AMP', self._amp_manager),
                ('Data', self._data_manager),
                ('DataLoader', self._dataloader_manager),
                ('Tensor', self._tensor_manager),
                ('Tokenizer', self._tokenizer_manager),
                ('Model', self._model_manager),
                ('Metrics', self._metrics_manager),
                ('Parameter', self._parameter_manager),
                ('Directory', self._directory_manager),
                ('Storage', self._storage_manager)
            ]
            if self._wandb_manager:
                self._initialization_order.append(('WandB', self._wandb_manager))
            if self._optuna_manager:
                self._initialization_order.append(('Optuna', self._optuna_manager))
            self._initialization_order.append(('Worker', self._worker_manager))

            # Initialize managers in order
            for name, manager in self._initialization_order:
                try:
                    logger.debug(f"Initializing {name} manager")
                    manager.ensure_initialized(config)
                except Exception as e:
                    logger.error(f"Failed to initialize {name} manager: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise RuntimeError(f"Failed to initialize {name} manager") from e

            logger.info(
                "Process resources initialized successfully:\n" +
                "\n".join(f"- {name}" for name, _ in self._initialization_order)
            )

        except Exception as e:
            logger.error("Failed to initialize process resources")
            logger.error(traceback.format_exc())
            self.cleanup_process()  # Clean up any initialized resources
            raise

    def cleanup_process(self) -> None:
        """Clean up process-local resources."""
        cleanup_errors: List[str] = []

        # Lazy imports for cleanup (if needed beyond initialization)
        from src.common.managers.storage_manager import StorageManager
        from src.common.managers.directory_manager import DirectoryManager
        from src.common.managers.model_manager import ModelManager
        from src.common.managers.tokenizer_manager import TokenizerManager
        from src.common.managers.tensor_manager import TensorManager
        from src.common.managers.worker_manager import WorkerManager
        from src.common.managers.wandb_manager import WandbManager

        # Clean up in reverse order
        for name, manager in reversed(self._initialization_order):
            try:
                logger.debug(f"Cleaning up {name} manager")
                if hasattr(manager, 'cleanup'):
                    manager.cleanup()
                elif name == 'Worker':
                    self._worker_manager.cleanup_workers()
                elif name == 'WandB' and self._wandb_manager:
                    self._wandb_manager.finish()
                elif name == 'Storage':
                    self._storage_manager.cleanup()
                elif name == 'Directory':
                    self._directory_manager.cleanup()
                elif name == 'Model':
                    self._model_manager.cleanup()
                elif name == 'Tokenizer':
                    self._tokenizer_manager.cleanup()
                elif name == 'Tensor':
                    self._tensor_manager.cleanup()

            except Exception as e:
                error_msg = f"Error during {name} manager cleanup: {str(e)}"
                logger.warning(error_msg)
                logger.warning(traceback.format_exc())
                cleanup_errors.append(error_msg)

        if cleanup_errors:
            logger.warning(
                "Process cleanup completed with errors:\n" +
                "\n".join(f"- {error}" for error in cleanup_errors)
            )
        else:
            logger.info("Process resources cleaned up successfully")

__all__ = ['ResourceInitializer']