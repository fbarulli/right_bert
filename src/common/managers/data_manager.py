
# src/common/managers/data_manager.py
from __future__ import annotations
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import logging
import os
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from transformers import PreTrainedTokenizerFast
from torch.utils.data.dataloader import default_collate
import threading

from src.common.managers.base_manager import BaseManager
from src.common.managers.tokenizer_manager import TokenizerManager
from src.common.managers.dataloader_manager import DataLoaderManager
from src.embedding.dataset import EmbeddingDataset

logger = logging.getLogger(__name__)

class DataManager(BaseManager):
    """
    Manages data resources including datasets and dataloaders.

    This manager handles:
    - Dataset creation and caching
    - Dataloader configuration
    - Train/validation data splitting
    """

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        dataloader_manager: DataLoaderManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize DataManager.

        Args:
            tokenizer_manager: Injected TokenizerManager instance
            dataloader_manager: Injected DataLoaderManager instance
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self._tokenizer_manager = tokenizer_manager
        self._dataloader_manager = dataloader_manager
        self._shared_datasets = {}
    self._lock = threading.Lock()

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local attributes.

        Args:
            config: Optional configuration dictionary that overrides the one from constructor
        """
        try:
            super()._initialize_process_local(config)

            if not self._tokenizer_manager.is_initialized():
                raise RuntimeError("TokenizerManager must be initialized before DataManager")
            if not self._dataloader_manager.is_initialized():
                raise RuntimeError("DataLoaderManager must be initialized before DataManager")

            logger.info(f"DataManager initialized for process {self._local.pid}")

        except Exception as e:
            logger.error(f"Failed to initialize DataManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_tokenizer(self, config: Dict[str, Any]) -> PreTrainedTokenizerFast:
        """
        Get the tokenizer for the current process.

        Args:
            config: Configuration dictionary containing model settings

        Returns:
            PreTrainedTokenizerFast: The tokenizer instance
        """
        self.ensure_initialized()
        return self._tokenizer_manager.get_worker_tokenizer(
            worker_id=self._local.pid,
            model_name=config['model']['name'],
            model_type=config['model']['stage']
        )

    def create_dataset(
        self,
        config: Dict[str, Any],
        split: str = 'train'
    ) -> Dataset:
        """
        Create a dataset instance.

        Args:
            config: Configuration dictionary
            split: Data split ('train' or 'val')

        Returns:
            Dataset: The created dataset

        Raises:
            ValueError: If split is invalid or model stage is unsupported
        """
        self.ensure_initialized()

        if split not in ['train', 'val']:
            raise ValueError(f"Invalid split: {split}")

        try:
            tokenizer = self.get_tokenizer(config)
            data_config = self.get_config_section(config, 'data')

            if config['model']['stage'] == 'embedding':
                dataset = EmbeddingDataset(
                    data_path=Path(data_config['csv_path']),
                    tokenizer=tokenizer,
                    split=split,
                    train_ratio=float(data_config['train_ratio']),
                    max_length=data_config['max_length'],
                    mask_prob=data_config['embedding_mask_probability'],
                    max_predictions=data_config['max_predictions'],
                    max_span_length=data_config['max_span_length']
                )
            elif config['model']['stage'] == 'classification':
                from src.classification.dataset import ClassificationDataset  # Conditional import
                dataset = ClassificationDataset(
                    data_path=Path(data_config['csv_path']),
                    tokenizer=tokenizer,
                    split=split,
                    train_ratio=float(data_config['train_ratio']),
                    max_length=data_config['max_length']
                )
            else:
                raise ValueError(f"Unsupported model stage: {config['model']['stage']}")

            logger.debug(f"Created {split} dataset with {len(dataset)} examples")
            return dataset

        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_dataloader(
        self,
        config: Dict[str, Any],
        dataset: Optional[Dataset] = None,
        split: str = 'train'
    ) -> DataLoader:
        """
        Create a dataloader for a dataset.

        Args:
            config: Configuration dictionary
            dataset: Optional dataset instance (will be created if not provided)
            split: Data split ('train' or 'val')

        Returns:
            DataLoader: The created dataloader
        """
        self.ensure_initialized()

        try:
            if dataset is None:
                dataset = self.create_dataset(config, split)

            training_config = self.get_config_section(config, 'training')
            batch_size = training_config['batch_size']
            num_workers = training_config['num_workers']

            loader = self._dataloader_manager.create_dataloader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=default_collate,
                persistent_workers=True,
                config=config
            )

            logger.debug(f"Created {split} dataloader with batch size {loader.batch_size}")
            return loader

        except Exception as e:
            logger.error(f"Error creating dataloader: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_dataloaders(
        self,
        config: Dict[str, Any]
    ) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:
        """
        Create train and validation dataloaders and datasets.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple containing:
            - Train dataloader
            - Validation dataloader
            - Train dataset
            - Validation dataset
        """
        self.ensure_initialized()

        try:
            # Create datasets
            train_dataset = self.create_dataset(config, split='train')
            val_dataset = self.create_dataset(config, split='val')

            # Create dataloaders
            train_loader = self.create_dataloader(config, dataset=train_dataset, split='train')
            val_loader = self.create_dataloader(config, dataset=val_dataset, split='val')

            return train_loader, val_loader, train_dataset, val_dataset

        except Exception as e:
            logger.error(f"Error creating dataloaders: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def cleanup(self) -> None:
        """Clean up data manager resources."""
        try:
            with self._lock:
                self._shared_datasets.clear()
            logger.info(f"Cleaned up DataManager for process {self._local.pid}")
            super().cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up DataManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise
