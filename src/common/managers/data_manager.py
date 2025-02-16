# src/common/managers/data_manager.py (Refactored)
from __future__ import annotations
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
from src.common.managers import (  # Corrected import
    get_dataloader_manager,
    get_tokenizer_manager
)
from src.common.resource.resource_initializer import ResourceInitializer

from src.embedding.dataset import EmbeddingDataset  # Corrected import
from src.data.csv_dataset import CSVDataset  # Corrected import

logger = logging.getLogger(__name__)

class DataManager(BaseManager):
    """Manages data resources with proper process and thread synchronization."""
    
    _shared_datasets = {}
    _lock = threading.Lock()

    def __init__(self):
        super().__init__()
    
    def get_tokenizer(self, config: Dict[str, Any]) -> 'PreTrainedTokenizerFast':
        """Get tokenizer for the current process."""
        try:
            worker_id = os.getpid()
            tokenizer = tokenizer_manager.get_worker_tokenizer(
                worker_id=worker_id,
                model_name=config['model']['name'],
                model_type=config['model']
            )
            logger.debug(f"Created tokenizer for worker {worker_id}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to get tokenizer: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes."""
        try:
            super()._initialize_process_local(config)
            self._local.resources = None
            logger.info(f"DataManager initialized for process {self._local.pid}")
        except Exception as e:
            logger.error(f"Failed to initialize DataManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_dataset(
        self,
        config: Dict[str, Any],
        split: str = 'train'
    ) -> Dataset:
        """Create a single dataset instance."""
        try:
            if split not in ['train', 'val']:
                raise ValueError(f"Invalid split: {split}")
                
            tokenizer = self.get_tokenizer(config)
            
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
        """Create a dataloader for a dataset."""
        try:
            if dataset is None:
                dataset = self.create_dataset(config, split)

            batch_size = config['training']['batch_size']
            num_workers = config['training']['num_workers']

            loader = dataloader_manager.create_dataloader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=default_collate,
                persistent_workers=True
            )

            logger.debug(
                f"Created {split} dataloader with batch size {loader.batch_size}"
            )
            return loader

        except Exception as e:
            logger.error(f"Failed to create {split} dataloader: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _create_datasets(
        self,
        config: Dict[str, Any]
    ) -> Tuple[Dataset, Dataset]:
        """Create train and validation datasets."""
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
        """Create train and validation dataloaders."""
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
        """Initialize shared data resources."""
        with self._lock:
            if not self._shared_datasets:
                logger.info("Initializing shared datasets")
                try:
                    train_dataset, val_dataset = self._create_datasets(config)
                    self._shared_datasets['train'] = train_dataset
                    self._shared_datasets['val'] = val_dataset
                    logger.info("Successfully created shared datasets")
                except Exception as e:
                    logger.error(f"Failed to create shared datasets: {str(e)}")
                    raise

        return self._shared_datasets

    def init_process_resources(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Initialize process-local resources using shared datasets."""
        self.ensure_initialized(config)
        
        try:
            ResourceInitializer.initialize_process(config)
            
            shared = self.init_shared_resources(config)
            tokenizer = self.get_tokenizer(config)

            train_loader, val_loader = self._create_dataloaders(
                shared['train'],
                shared['val'],
                config
            )

            self._local.resources = {
                'tokenizer': tokenizer,
                'train_dataset': shared['train'],
                'val_dataset': shared['val'],
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
        data_path: Path,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int,
        batch_size: int,
        train_ratio: float = 0.9,
        is_embedding: bool = True,
        mask_prob: float = 0.15,
        max_predictions: int = 20,
        max_span_length: int = 1,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:
        """Create train and validation dataloaders."""
        try:
            DatasetClass = EmbeddingDataset if is_embedding else CSVDataset
            
            dataset_kwargs = {
                'tokenizer': tokenizer,
                'max_length': max_length,
                'train_ratio': train_ratio
            }
            if is_embedding:
                dataset_kwargs.update({
                    'mask_prob': mask_prob,
                    'max_predictions': max_predictions,
                    'max_span_length': max_span_length
                })
            
            train_dataset = DatasetClass(
                data_path=data_path,
                split='train',
                **dataset_kwargs
            )
            val_dataset = DatasetClass(
                data_path=data_path,
                split='val',
                **dataset_kwargs
            )
            
            train_loader = dataloader_manager.create_dataloader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
            
            val_loader = dataloader_manager.create_dataloader(
                dataset=val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
            
            return train_loader, val_loader, train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"Failed to create dataloaders: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _validate_resources(self, resources: Dict[str, Any]) -> None:
        """Validate that all required resources exist and are of correct type."""
        required = {
            'tokenizer': PreTrainedTokenizerFast,
            'train_dataset': EmbeddingDataset,
            'val_dataset': EmbeddingDataset,
            'train_loader': DataLoader,
            'val_loader': DataLoader
        }

        for name, expected_type in required.items():
            if name not in resources:
                raise ValueError(f"Missing required resource: {name}")
            if not isinstance(resources[name], expected_type):
                raise TypeError(f"Resource {name} has wrong type: {type(resources[name])}, expected {expected_type}")

__all__ = ['DataManager']