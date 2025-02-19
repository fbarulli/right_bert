# src/common/study/objective_factory.py
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import optuna
from torch.utils.data import DataLoader, Dataset

from src.common.managers.parameter_manager import ParameterManager
from src.common.managers.data_manager import DataManager
from src.common.managers.tokenizer_manager import TokenizerManager
from src.common.managers.directory_manager import DirectoryManager
from src.common.managers.optuna_manager import OptunaManager
from src.common.managers.amp_manager import AMPManager
from src.common.managers.cuda_manager import CUDAManager
from src.common.managers.tensor_manager import TensorManager
from src.common.managers.batch_manager import BatchManager
from src.common.managers.metrics_manager import MetricsManager
from src.common.managers.dataloader_manager import DataLoaderManager
from src.common.managers.storage_manager import StorageManager
from src.common.managers.resource_manager import ProcessResourceManager
from src.common.managers.worker_manager import WorkerManager
from src.common.managers.wandb_manager import WandbManager

logger = logging.getLogger(__name__)

class ObjectiveFactory:
    """
    Factory for creating objective functions for Optuna studies.
    
    This factory handles:
    - Trial configuration and setup
    - Model creation and training
    - Resource management
    - Metric tracking
    """

    def __init__(
        self,
        parameter_manager: ParameterManager,
        data_manager: DataManager,
        tokenizer_manager: TokenizerManager,
        directory_manager: DirectoryManager,
        optuna_manager: OptunaManager,
        amp_manager: AMPManager,
        cuda_manager: CUDAManager,
        tensor_manager: TensorManager,
        batch_manager: BatchManager,
        metrics_manager: MetricsManager,
        dataloader_manager: DataLoaderManager,
        storage_manager: StorageManager,
        resource_manager: ProcessResourceManager,
        worker_manager: WorkerManager,
        wandb_manager: Optional[WandbManager],
        config: Dict[str, Any],
        output_path: Path
    ):
        """
        Initialize ObjectiveFactory with dependency injection.

        Args:
            parameter_manager: Injected ParameterManager instance
            data_manager: Injected DataManager instance
            tokenizer_manager: Injected TokenizerManager instance
            directory_manager: Injected DirectoryManager instance
            optuna_manager: Injected OptunaManager instance
            amp_manager: Injected AMPManager instance
            cuda_manager: Injected CUDAManager instance
            tensor_manager: Injected TensorManager instance
            batch_manager: Injected BatchManager instance
            metrics_manager: Injected MetricsManager instance
            dataloader_manager: Injected DataLoaderManager instance
            storage_manager: Injected StorageManager instance
            resource_manager: Injected ProcessResourceManager instance
            worker_manager: Injected WorkerManager instance
            wandb_manager: Optional injected WandbManager instance
            config: Configuration dictionary
            output_path: Output directory path
        """
        # Store injected managers
        self._parameter_manager = parameter_manager
        self._data_manager = data_manager
        self._tokenizer_manager = tokenizer_manager
        self._directory_manager = directory_manager
        self._optuna_manager = optuna_manager
        self._amp_manager = amp_manager
        self._cuda_manager = cuda_manager
        self._tensor_manager = tensor_manager
        self._batch_manager = batch_manager
        self._metrics_manager = metrics_manager
        self._dataloader_manager = dataloader_manager
        self._storage_manager = storage_manager
        self._resource_manager = resource_manager
        self._worker_manager = worker_manager
        self._wandb_manager = wandb_manager

        # Store configuration
        self.config = config
        self.output_path = output_path

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function to be optimized by Optuna.

        Args:
            trial: Optuna trial instance

        Returns:
            float: Objective value to minimize

        Raises:
            ValueError: If unknown training stage
            Exception: If objective function fails
        """
        try:
            # Create trial directory
            trial_output_dir = self._directory_manager.base_dir / "trials" / f"trial_{trial.number}"
            metrics_dir = trial_output_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)

            # Initialize wandb if needed
            if self.config["training"]["num_trials"] > 1 and self._wandb_manager:
                self._wandb_manager.init_trial(trial.number)

            # Run appropriate objective based on stage
            if self.config['model']['stage'] == 'embedding':
                return self._embedding_objective(
                    trial,
                    metrics_dir
                )
            elif self.config['model']['stage'] == 'classification':
                return self._classification_objective(
                    trial,
                    metrics_dir
                )
            else:
                raise ValueError(
                    f"Unknown training stage: {self.config['model']['stage']}"
                )

        except Exception as e:
            logger.error(f"Objective function failed: {str(e)}")
            logger.error(traceback.format_exc())
            trial.set_user_attr("fail_reason", str(e))
            raise

    def _embedding_objective(
        self,
        trial: optuna.Trial,
        metrics_dir: Path
    ) -> float:
        """
        Objective function for embedding model optimization.

        Args:
            trial: Optuna trial instance
            metrics_dir: Directory for saving metrics

        Returns:
            float: Best validation loss
        """
        try:
            # Get trial configuration
            config = self._parameter_manager.get_trial_config(trial)

            # Initialize tokenizer
            tokenizer = self._tokenizer_manager.get_worker_tokenizer(
                trial.number,
                config['model']['name']
            )
            self._tokenizer_manager.set_shared_tokenizer(tokenizer)

            # Create data loaders
            train_loader, val_loader, train_dataset, val_dataset = (
                self._data_manager.create_dataloaders(config)
            )

            # Create model
            from src.embedding.model import embedding_model_factory
            model = embedding_model_factory(config, trial=trial)

            # Create trainer
            from src.embedding.embedding_trainer import EmbeddingTrainer
            trainer = EmbeddingTrainer(
                cuda_manager=self._cuda_manager,
                batch_manager=self._batch_manager,
                amp_manager=self._amp_manager,
                tokenizer_manager=self._tokenizer_manager,
                metrics_manager=self._metrics_manager,
                storage_manager=self._storage_manager,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                metrics_dir=metrics_dir,
                is_trial=True,
                trial=trial,
                wandb_manager=(
                    self._wandb_manager
                    if self.config["training"]["num_trials"] > 1
                    else None
                ),
                job_id=trial.number,
                train_dataset=train_dataset,
                val_dataset=val_dataset
            )

            # Train model
            trainer.train(config['training']['num_epochs'])
            return trainer.best_val_loss

        except Exception as e:
            logger.error(f"Error in embedding objective: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _classification_objective(
        self,
        trial: optuna.Trial,
        metrics_dir: Path
    ) -> float:
        """
        Objective function for classification model optimization.

        Args:
            trial: Optuna trial instance
            metrics_dir: Directory for saving metrics

        Returns:
            float: Best validation loss
        """
        try:
            # Get trial configuration
            config = self._parameter_manager.get_trial_config(trial)
            config['model']['num_labels'] = self.config['model']['num_labels']

            # Initialize tokenizer
            tokenizer = self._tokenizer_manager.get_worker_tokenizer(
                trial.number,
                self.config['model']['name'],
                model_type='classification'
            )
            self._tokenizer_manager.set_shared_tokenizer(tokenizer)

            # Create data loaders
            train_loader, val_loader, train_dataset, val_dataset = (
                self._data_manager.create_dataloaders(config)
            )

            # Create model
            from src.classification.model import classification_model_factory
            model = classification_model_factory(config, trial=trial)

            # Create trainer
            from src.classification.classification_trainer import ClassificationTrainer
            trainer = ClassificationTrainer(
                cuda_manager=self._cuda_manager,
                batch_manager=self._batch_manager,
                amp_manager=self._amp_manager,
                tokenizer_manager=self._tokenizer_manager,
                metrics_manager=self._metrics_manager,
                storage_manager=self._storage_manager,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                metrics_dir=metrics_dir,
                is_trial=True,
                trial=trial,
                wandb_manager=(
                    self._wandb_manager
                    if self.config["training"]["num_trials"] > 1
                    else None
                ),
                job_id=trial.number,
                train_dataset=train_dataset,
                val_dataset=val_dataset
            )

            # Train model
            trainer.train(config['training']['num_epochs'])
            return trainer.best_val_loss

        except Exception as e:
            logger.error(f"Error in classification objective: {str(e)}")
            logger.error(traceback.format_exc())
            raise


__all__ = ['ObjectiveFactory']
