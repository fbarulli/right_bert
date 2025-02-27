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
from src.common.fix_dataloader import fix_dataloader_config

logger = logging.getLogger(__name__)

class DataManager(BaseManager):
    """Manager for handling datasets."""
    
    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        dataloader_manager: DataLoaderManager,
        config: Optional[Dict[str, Any]] = None
    ):
        self._tokenizer_manager = tokenizer_manager  # Set before super()
        self._dataloader_manager = dataloader_manager  # Set before super()
        super().__init__(config)
        self._shared_datasets = {}
        self._lock = threading.Lock()  # Re-added here

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes."""
        try:
            super()._initialize_process_local(config)
            
            # Check dependencies
            if not self._tokenizer_manager.is_initialized():
                logger.warning("TokenizerManager not initialized, attempting to initialize it")
                self._tokenizer_manager._initialize_process_local(config)
                
            if not self._dataloader_manager.is_initialized():
                logger.warning("DataLoaderManager not initialized, attempting to initialize it")
                self._dataloader_manager._initialize_process_local(config)
            
            # Ensure local attributes exist
            if not hasattr(self._local, 'datasets'):
                self._local.datasets = {}
                
            if not hasattr(self._local, 'tokenizer'):
                # Attempt to get shared tokenizer 
                try:
                    self._local.tokenizer = self._tokenizer_manager.get_shared_tokenizer()
                    if self._local.tokenizer is None:
                        # If no shared tokenizer, get a worker tokenizer
                        self._local.tokenizer = self._tokenizer_manager.get_worker_tokenizer(0)
                except Exception as e:
                    logger.error(f"Failed to get tokenizer: {e}")
                    self._local.tokenizer = None
                    
            logger.info(f"DataManager initialized for process {self._local.pid}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DataManager: {str(e)}")
            logger.error(traceback.format_exc())
            # Don't propagate the exception to allow process to continue
            # but mark as not initialized
            self._local.initialized = False

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

    def create_dataset(self, config: Dict[str, Any], split: str = 'train') -> torch.utils.data.Dataset:
        """Create a dataset based on the configuration.
        
        Args:
            config: The configuration dictionary
            split: The dataset split ('train' or 'val')
            
        Returns:
            torch.utils.data.Dataset: The created dataset
        """
        self.ensure_initialized()
        
        try:
            tokenizer = self._tokenizer_manager.get_worker_tokenizer(os.getpid())
            logger.info(f"Creating {split} dataset for worker {os.getpid()}")
            
            # Get data configuration
            data_config = config.get('data', {})
            model_name = config['model'].get('name', '')
            dataset_type = data_config.get('dataset_type', 'embedding')
            
            if dataset_type == 'embedding':
                from src.embedding.dataset import EmbeddingDataset
                
                # Use the from_config method instead of direct instantiation
                logger.debug(f"Creating EmbeddingDataset for split: {split}")
                
                # Check if the from_config class method exists
                if hasattr(EmbeddingDataset, 'from_config'):
                    dataset = EmbeddingDataset.from_config(
                        config=config,
                        tokenizer=tokenizer,
                        split=split
                    )
                else:
                    # Fallback to older direct initialization if needed
                    # Inspect the actual __init__ signature to determine parameters
                    import inspect
                    init_signature = inspect.signature(EmbeddingDataset.__init__)
                    
                    if 'config' in init_signature.parameters:
                        # Create config for constructor
                        from src.embedding.dataset import EmbeddingDatasetConfig
                        from pathlib import Path
                        
                        dataset_config = EmbeddingDatasetConfig(
                            data_path=Path(data_config.get('csv_path', 'data/embedding.csv')),
                            max_length=data_config.get('max_length', 128),
                            split=split,
                            train_ratio=data_config.get('train_ratio', 0.8),
                            mask_prob=data_config.get('embedding_mask_probability', 0.15),
                            max_predictions=data_config.get('max_predictions', 20),
                            max_span_length=data_config.get('max_span_length', 3),
                            log_level=config['training'].get('log_level', 'log'),
                            cache_size=config['training'].get('cache_size', 1000),
                            tensor_pool_size=config['training'].get('tensor_pool_size', 1000),
                            gc_threshold=config['training'].get('gc_threshold', 0.8)
                        )
                        
                        dataset = EmbeddingDataset(tokenizer=tokenizer, config=dataset_config)
                    else:
                        # Assume direct initialization with named parameters
                        dataset = EmbeddingDataset(
                            tokenizer=tokenizer,
                            split=split,
                            data_path=data_config.get('path', 'data/embedding'),
                            max_length=data_config.get('max_length', 128)
                        )
                
                return dataset
                
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
                
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

    def create_dataloaders(self, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:
        """
        Create train and validation data loaders.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Tuple[DataLoader, DataLoader, Dataset, Dataset]: Train loader, validation loader, 
                                                             train dataset, validation dataset
        """
        # First, ensure we are initialized with robust error handling
        try:
            self.ensure_initialized()
        except RuntimeError as e:
            logger.warning(f"Auto-initializing DataManager due to: {e}")
            try:
                self._initialize_process_local(config)
                if not self.is_initialized():
                    logger.error("DataManager initialization failed during auto-recovery")
                    self._local.initialized = True  # Force it to be initialized to continue
            except Exception as init_error:
                logger.error(f"Error during DataManager auto-initialization: {init_error}")
                logger.error(traceback.format_exc())
                # Set local initialized flag anyway as a last resort
                self._local = threading.local()
                self._local.pid = os.getpid()
                self._local.initialized = True
                self._local.datasets = {}
                
        # Continue with dataloader creation
        # Apply fixes to config to avoid pickle errors
        config = fix_dataloader_config(config)
        
        try:
            # First get the datasets
            train_dataset = self.create_dataset(config, "train")
            val_dataset = self.create_dataset(config, "val")
            
            # Get batch size
            batch_size = config['training'].get('batch_size', 16)
            
            # Determine number of workers (now 0 after fix_dataloader_config)
            num_workers = config['training'].get('num_workers', 0)
            
            # Get the dataloader manager from the factory
            from src.common.managers import get_dataloader_manager
            dataloader_manager = get_dataloader_manager()
            
            # Create the dataloaders
            train_loader = dataloader_manager.create_dataloader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=config['training'].get('pin_memory', False),
                persistent_workers=False  # Set to False for safety
            )
            
            val_loader = dataloader_manager.create_dataloader(
                dataset=val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=config['training'].get('pin_memory', False),
                persistent_workers=False  # Set to False for safety
            )
            
            logger.info(
                f"DataLoaders created with:\n"
                f" - Batch size: {batch_size}\n"
                f" - Workers: {num_workers}\n"
                f" - Train samples: {len(train_dataset)}\n"
                f" - Val samples: {len(val_dataset)}"
            )
            
            return train_loader, val_loader, train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"Failed to create dataloaders: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def cleanup(self) -> None:
            try:
                if not hasattr(self, '_local'):
                    logger.info("No local resources to clean up in DataManager")
                    return
                if hasattr(self, '_lock'):
                    with self._lock:  # Safely use lock if it exists
                        self._shared_datasets.clear()
                logger.info(f"Cleaned up DataManager for process {self._local.pid}")
                super().cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up DataManager: {str(e)}")
                logger.error(traceback.format_exc())
                raise
