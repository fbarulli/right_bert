# src/common/managers/data_manager.py
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import logging
import os
import traceback
from pathlib import Path
from filelock import FileLock
from typing import Dict, Any, Tuple, Optional
from transformers import PreTrainedTokenizerFast
from torch.utils.data.dataloader import default_collate
import threading

from src.common.managers.base_manager import BaseManager
from src.common.managers import (
    get_dataloader_manager,
    get_tokenizer_manager
)
from src.common.resource.resource_initializer import ResourceInitializer

from src.embedding.dataset import EmbeddingDataset
from src.data.csv_dataset import CSVDataset

logger = logging.getLogger(__name__)

class DataManager(BaseManager):
    """Manages data resources."""

    _shared_datasets = {}
    _lock = threading.Lock()

    def __init__(self):
        super().__init__()
        self.dataloader_manager = None  # Initialize
        self.tokenizer_manager = None

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        super()._initialize_process_local(config)
        self.dataloader_manager = get_dataloader_manager() # Get instances
        self.tokenizer_manager = get_tokenizer_manager()
        self._local.resources = None
        logger.info(f"DataManager initialized for process {self._local.pid}")

    def get_tokenizer(self, config: Dict[str, Any]) -> 'PreTrainedTokenizerFast':
        """Gets the tokenizer (already initialized)."""
        try:
            worker_id = os.getpid()
            # Use the shared tokenizer
            tokenizer = self.tokenizer_manager.get_worker_tokenizer(
                worker_id=worker_id,
                model_name=config['model']['name'],
                model_type=config['model']['stage']
            )
            logger.debug(f"Got tokenizer for worker {worker_id}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to get tokenizer: {str(e)}")
            raise

    def create_dataset(
        self,
        config: Dict[str, Any],
        split: str = 'train'
    ) -> Dataset:
        """Creates a single dataset instance (EmbeddingDataset or ClassificationDataset)."""

        try:
            if split not in ['train', 'val']:
                raise ValueError(f"Invalid split: {split}")

            tokenizer = self.get_tokenizer(config)

            if config['model']['stage'] == 'embedding':
                # Embedding-specific parameters
                mask_prob = config['data']['embedding_mask_probability']
                max_predictions = config['data']['max_predictions']
                max_span_length = config['data']['max_span_length']

                dataset = EmbeddingDataset(
                    data_path=Path(config['data']['csv_path']),
                    tokenizer=tokenizer,
                    split=split,
                    train_ratio=float(config['data']['train_ratio']),
                    max_length=config['data']['max_length'],
                    mask_prob=mask_prob,
                    max_predictions=max_predictions,
                    max_span_length=max_span_length
                )
            elif config['model']['stage'] == 'classification':
                # Use ClassificationDataset, no masking needed.
                from src.classification.dataset import ClassificationDataset  # Local import
                dataset = ClassificationDataset(
                    data_path=Path(config['data']['csv_path']),
                    tokenizer=tokenizer,
                    split=split,
                    train_ratio=float(config['data']['train_ratio']),
                    max_length=config['data']['max_length']
                )

            else:
                raise ValueError(f"Unsupported model stage for dataset creation: {config['model']['stage']}")

            logger.debug(f"Created {split} dataset with {len(dataset)} examples")
            return dataset

        except Exception as e:
            logger.error(f"Failed to create {split} dataset: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_dataloader(
        self,
        config: Dict[str, Any],
        dataset: Optional[Dataset] = None,
        split: str = 'train'
    ) -> DataLoader:
        """Creates a dataloader for a dataset."""
        try:
            if dataset is None:
                dataset = self.create_dataset(config, split)

            batch_size = config['training']['batch_size']
            num_workers = config['training']['num_workers']

            loader = self.dataloader_manager.create_dataloader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=default_collate,
                persistent_workers=True,
                config=config  # Pass config to dataloader_manager
            )

            logger.debug(f"Created {split} dataloader with batch size {loader.batch_size}")
            return loader

        except Exception as e:
            logger.error(f"Failed to create {split} dataloader: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _create_datasets(self, config: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
        """Creates train and validation datasets."""
        try:
            train_dataset = self.create_dataset(config, split='train')
            val_dataset = self.create_dataset(config, split='val')
            return train_dataset, val_dataset
        except Exception as e:
            logger.error(f"Error creating datasets: {str(e)}")
            raise

    def _create_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: Dict[str, Any]
    ) -> Tuple[DataLoader, DataLoader]:
        """Creates train and validation dataloaders."""
        try:
            train_loader = self.create_dataloader(
                config,
                dataset=train_dataset,
                split='train'
            )
            val_loader = self.create_dataloader(
                config,
                dataset=val_dataset,
                split='val'
            )
            return train_loader, val_loader
        except Exception as e:
            logger.error(f"Error creating dataloaders: {str(e)}")
            raise

    def init_shared_resources(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initializes shared data resources (datasets)."""
        with self._lock:
            if not self._shared_datasets:
                logger.info("Initializing shared datasets")
                try:
                    # Create the datasets (train and val) *once*
                    train_dataset, val_dataset = self._create_datasets(config)
                    self._shared_datasets['train'] = train_dataset
                    self._shared_datasets['val'] = val_dataset
                    logger.info("Successfully created shared datasets")
                except Exception as e:
                    logger.error(f"Failed to create shared datasets: {str(e)}")
                    raise

        return self._shared_datasets

    def init_process_resources(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initializes process-local resources using shared datasets."""
        self.ensure_initialized(config)

        try:
            # Ensure all process resources are initialized first (especially CUDA)
            from src.common.resource.resource_initializer import ResourceInitializer
            ResourceInitializer.initialize_process(config)

            # Get the *shared* datasets (created by init_shared_resources)
            shared = self.init_shared_resources(config)
            tokenizer = self.get_tokenizer(config)

            # Create dataloaders using the shared datasets
            train_loader, val_loader = self._create_dataloaders(
                shared['train'],
                shared['val'],
                config
            )

            self._local.resources = {
                'tokenizer': tokenizer,  # Include the tokenizer
                'train_dataset': shared['train'],  # Use the *shared* dataset
                'val_dataset': shared['val'],      # Use the *shared* dataset
                'train_loader': train_loader,
                'val_loader': val_loader
            }

            self._validate_resources(self._local.resources)
            logger.info(f"Successfully initialized process {self._local.pid} resources")

        except Exception as e:
            logger.error(f"Error initializing process {self._local.pid} resources: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        return self._local.resources

    def create_dataloaders(
        self,
        config: Dict[str, Any]
    ) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:
        """
        Creates train and validation dataloaders, pulling ALL parameters from config.

        This method no longer takes individual arguments for data_path, tokenizer, etc.
        Instead, it reads *everything* from the provided `config` dictionary.
        """
        try:

            if config['model']['stage'] == 'embedding':
                DatasetClass = EmbeddingDataset
            elif config['model']['stage'] == 'classification':
                from src.classification.dataset import ClassificationDataset
                DatasetClass = ClassificationDataset
            else:
                raise ValueError(f"Unsupported model stage: {config['model']['stage']}")

            tokenizer = self.get_tokenizer(config) # Get tokenizer

            train_dataset = DatasetClass(
                data_path=Path(config['data']['csv_path']),
                tokenizer=tokenizer,
                split='train',
                train_ratio=float(config['data']['train_ratio']),
                max_length=config['data']['max_length'],
                # Only for embedding
                **(dict(mask_prob=config['data']['embedding_mask_probability'],
                max_predictions=config['data']['max_predictions'],
                max_span_length=config['data']['max_span_length']) if config['model']['stage'] == 'embedding' else {})
            )
            val_dataset = DatasetClass(
                data_path=Path(config['data']['csv_path']),
                tokenizer=tokenizer,
                split='val',
                train_ratio=float(config['data']['train_ratio']),
                max_length=config['data']['max_length'],
                **(dict(mask_prob=config['data']['embedding_mask_probability'],
                max_predictions=config['data']['max_predictions'],
                max_span_length=config['data']['max_span_length']) if config['model']['stage'] == 'embedding' else {})
            )

            train_loader = self.dataloader_manager.create_dataloader(
                dataset=train_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=True,
                num_workers=config['training']['num_workers'],
                config=config # Pass config
            )

            val_loader = self.dataloader_manager.create_dataloader(
                dataset=val_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                num_workers=config['training']['num_workers'],
                config=config # Pass config
            )

            return train_loader, val_loader, train_dataset, val_dataset

        except Exception as e:
            logger.error(f"Failed to create dataloaders: {str(e)}")
            logger.error(traceback.format_exc())
            raise


    def _validate_resources(self, resources: Dict[str, Any]) -> None:
        required = {
            'tokenizer': PreTrainedTokenizerFast,
            'train_dataset': Dataset,  # Use the base class
            'val_dataset': Dataset,      # Use the base class
            'train_loader': DataLoader,
            'val_loader': DataLoader
        }

        for name, expected_type in required.items():
            if name not in resources:
                raise ValueError(f"Missing required resource: {name}")
            if not isinstance(resources[name], expected_type):
                raise TypeError(f"Resource {name} has wrong type: {type(resources[name])}, expected {expected_type}")

__all__ = ['DataManager']