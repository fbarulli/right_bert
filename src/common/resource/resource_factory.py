# src/common/resource/resource_factory.py
from __future__ import annotations
import logging
import traceback
from typing import Dict, Any, Union, Optional
from pathlib import Path

from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader

from src.common.managers.dataloader_manager import DataLoaderManager
from src.data.csv_dataset import csv_dataset_factory
from src.embedding.dataset import EmbeddingDataset
from src.classification.dataset import ClassificationDataset

logger = logging.getLogger(__name__)

class ResourceFactory:
    """
    Factory for creating resources like datasets and dataloaders.
    
    This factory handles:
    - Dataset creation for different stages
    - DataLoader configuration
    - Resource validation
    - Error handling
    """

    def __init__(
        self,
        dataloader_manager: DataLoaderManager
    ):
        """
        Initialize ResourceFactory with dependency injection.

        Args:
            dataloader_manager: Injected DataLoaderManager instance
        """
        self._dataloader_manager = dataloader_manager

    def create_resource(
        self,
        resource_type: str,
        config: Dict[str, Any],
        **kwargs: Any
    ) -> Union[Dataset, DataLoader]:
        """
        Create resources like datasets and dataloaders.

        Args:
            resource_type: Type of resource ('dataset' or 'dataloader')
            config: Configuration dictionary
            **kwargs: Additional keyword arguments for the specific resource

        Returns:
            Union[Dataset, DataLoader]: The created resource

        Raises:
            ValueError: If resource type is unknown or required args missing
            Exception: If resource creation fails
        """
        try:
            logger.info(f"Creating resource of type: {resource_type}")

            if resource_type == 'dataset':
                return self._create_dataset(config, **kwargs)
            elif resource_type == 'dataloader':
                return self._create_dataloader(config, **kwargs)
            else:
                raise ValueError(f"Unknown resource type: {resource_type}")

        except Exception as e:
            logger.error(f"Error creating resource of type {resource_type}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _create_dataset(
        self,
        config: Dict[str, Any],
        **kwargs: Any
    ) -> Dataset:
        """
        Create dataset based on configuration.

        Args:
            config: Configuration dictionary
            **kwargs: Additional keyword arguments including:
                - tokenizer: PreTrainedTokenizerFast instance
                - split: Data split ('train' or 'val')

        Returns:
            Dataset: Created dataset

        Raises:
            ValueError: If required configuration or arguments missing
        """
        try:
            # Validate configuration
            if 'data_path' not in config['data'] or not config['data']['data_path']:
                raise ValueError("data_path must be specified in config for dataset creation")
            if 'tokenizer' not in kwargs:
                raise ValueError("tokenizer must be provided for dataset creation")

            # Get dataset parameters
            stage = config['model']['stage']
            data_path = Path(config['data']['csv_path'])
            tokenizer = kwargs['tokenizer']
            split = kwargs.get('split', 'train')
            train_ratio = float(config['data']['train_ratio'])
            max_length = config['data']['max_length']

            # Create dataset based on stage
            if stage == 'embedding':
                return EmbeddingDataset(
                    data_path=data_path,
                    tokenizer=tokenizer,
                    split=split,
                    train_ratio=train_ratio,
                    max_length=max_length,
                    mask_prob=config['data']['embedding_mask_probability'],
                    max_predictions=config['data']['max_predictions'],
                    max_span_length=config['data']['max_span_length']
                )
            elif stage == 'classification':
                return ClassificationDataset(
                    data_path=data_path,
                    tokenizer=tokenizer,
                    split=split,
                    train_ratio=train_ratio,
                    max_length=max_length,
                    num_labels=config['model']['num_labels']
                )
            else:
                raise ValueError(f"Unknown stage for dataset creation: {stage}")

        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _create_dataloader(
        self,
        config: Dict[str, Any],
        **kwargs: Any
    ) -> DataLoader:
        """
        Create dataloader based on configuration.

        Args:
            config: Configuration dictionary
            **kwargs: Additional keyword arguments including:
                - dataset: Dataset instance
                - split: Data split ('train' or 'val')

        Returns:
            DataLoader: Created dataloader

        Raises:
            ValueError: If required arguments missing
        """
        try:
            # Validate arguments
            if 'dataset' not in kwargs:
                raise ValueError("dataset must be provided for dataloader creation")

            # Get dataloader parameters
            dataset = kwargs['dataset']
            batch_size = config['training']['batch_size']
            num_workers = config['training']['num_workers']
            split = kwargs.get('split', 'train')

            # Create dataloader
            return self._dataloader_manager.create_dataloader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=split == 'train',
                num_workers=num_workers,
                pin_memory=True,
                config=config
            )

        except Exception as e:
            logger.error(f"Error creating dataloader: {str(e)}")
            logger.error(traceback.format_exc())
            raise


__all__ = ['ResourceFactory']
