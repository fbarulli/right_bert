# src/common/managers/data_manager.py
# src/common/managers/data_manager.py (ENSURE CLASS NAME IS DataManager)
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

class DataManager(BaseManager):  # Correct class name
    """Manages data resources."""

    _shared_datasets = {}  # Class-level shared datasets
    _lock = threading.Lock()  # Class-level lock

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DataManager. No dependencies in constructor."""
        super().__init__(config)  # Initialize base with config

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        super()._initialize_process_local(config)
        logger.info(f"DataManager initialized for process {os.getpid()}")


    def get_tokenizer(self, config: Dict[str, Any]) -> 'PreTrainedTokenizerFast':
        """Gets the tokenizer."""
        tokenizer_manager = get_tokenizer_manager()  # Get tokenizer manager *here*
        return tokenizer_manager.get_worker_tokenizer(
            worker_id=os.getpid(),
            model_name=config['model']['name'],
            model_type=config['model']['stage']
        )

    def create_dataset(self, config: Dict[str, Any], split: str = 'train') -> Dataset:
        """Creates a dataset instance (EmbeddingDataset or ClassificationDataset)."""

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

    def create_dataloader(self, config: Dict[str, Any], dataset: Optional[Dataset] = None, split: str = 'train') -> DataLoader:
        """Creates a dataloader for a dataset."""
        if dataset is None:
            dataset = self.create_dataset(config, split)

        batch_size = config['training']['batch_size']
        num_workers = config['training']['num_workers']
        dataloader_manager = get_dataloader_manager() # Get it here

        loader = dataloader_manager.create_dataloader(
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

    def _create_datasets(self, config: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
        """Creates train and validation datasets."""
        train_dataset = self.create_dataset(config, split='train')
        val_dataset = self.create_dataset(config, split='val')
        return train_dataset, val_dataset

    def _create_dataloaders(self, train_dataset: Dataset, val_dataset: Dataset, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
        """Creates train and validation dataloaders."""
        train_loader = self.create_dataloader(config, dataset=train_dataset, split='train')
        val_loader = self.create_dataloader(config, dataset=val_dataset, split='val')
        return train_loader, val_loader

    def create_dataloaders(self, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:
        """
        Creates train and validation dataloaders, pulling ALL parameters from config.

        This method no longer takes individual arguments for data_path, tokenizer, etc.
        Instead, it reads *everything* from the provided `config` dictionary.
        """
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
        dataloader_manager = get_dataloader_manager() # Get it here
        train_loader = dataloader_manager.create_dataloader(
            dataset=train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            config=config # Pass config
        )

        val_loader = dataloader_manager.create_dataloader(
            dataset=val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            config=config # Pass config
        )

        return train_loader, val_loader, train_dataset, val_dataset


__all__ = ['DataManager']