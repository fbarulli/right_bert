# src/common/resource/resource_factory.py
# src/common/resource/resource_factory.py
from __future__ import annotations
import logging
from typing import Dict, Any, Union
from pathlib import Path

from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader

from src.data.csv_dataset import csv_dataset_factory
from src.embedding.dataset import EmbeddingDataset
from src.common.managers import get_dataloader_manager

logger = logging.getLogger(__name__)

def create_resource(
    resource_type: str,
    config: Dict[str, Any],
    **kwargs: Any
) -> Union[Dataset, DataLoader]:
    """
    Factory function to create resources like datasets and dataloaders.

    Args:
        resource_type: Type of resource ('dataset' or 'dataloader').
        config: Configuration dictionary.
        **kwargs: Additional keyword arguments for the specific resource.

    Returns:
        The created resource.

    Raises:
        ValueError: If the resource type is unknown.
    """
    try:
        logger.info(f"Creating resource of type: {resource_type}")
        if resource_type == 'dataset':
            # Ensure required arguments are present
            if not 'data_path' in config['data'] or not config['data']['data_path']:
                raise ValueError("data_path must be specified in config for dataset creation.")
            if 'tokenizer' not in kwargs:
                raise ValueError("tokenizer must be provided for dataset creation.")

            stage = config['model']['stage']
            if stage == 'embedding':
                return EmbeddingDataset(
                    data_path=Path(config['data']['csv_path']),
                    tokenizer=kwargs['tokenizer'],
                    split=kwargs.get('split', 'train'),
                    train_ratio=float(config['data']['train_ratio']),
                    max_length=config['data']['max_length'],
                    mask_prob=config['data']['embedding_mask_probability'],
                    max_predictions=config['data']['max_predictions'],
                    max_span_length=config['data']['max_span_length']
                )
            else:
                raise ValueError(f"Unknown stage for dataset creation: {stage}")

        elif resource_type == 'dataloader':
            if 'dataset' not in kwargs:
                raise ValueError("dataset must be provided for dataloader creation.")
            batch_size = config['training']['batch_size']
            num_workers = config['training']['num_workers']
            return get_dataloader_manager().create_dataloader(
                dataset=kwargs['dataset'],
                batch_size=batch_size,
                shuffle=kwargs.get('split', 'train') == 'train',
                num_workers=num_workers,
                pin_memory=True,
                config=config
            )
        else:
            raise ValueError(f"Unknown resource type: {resource_type}")

    except Exception as e:
        logger.error(f"Error creating resource of type {resource_type}: {str(e)}", exc_info=True)
        raise

__all__ = ['create_resource']