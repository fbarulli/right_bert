"""
Embedding module for training and using text embeddings.
"""
import os
import sys
import logging

# Ensure the parent directory is in the path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

logger = logging.getLogger(__name__)
logger.debug(f"Initializing embedding package in process {os.getpid()}")

# Import required modules
# Make sure modules are imported in dependency order
try:
    # First import utilities
    from src.embedding.utils import (
        LogConfig,
        MemoryTracker,
        TensorPool,
        CachingDict
    )
    
    # Then import other modules that depend on utils
    from src.embedding.masking import SpanMaskingModule, MaskingConfig
    from src.embedding.csv_dataset import CSVDataset
    
    # Finally import high-level modules
    from src.embedding.dataset import EmbeddingDataset, EmbeddingDatasetConfig
    
    __all__ = [
        'LogConfig',
        'MemoryTracker',
        'TensorPool',
        'CachingDict',
        'SpanMaskingModule',
        'MaskingConfig',
        'CSVDataset',
        'EmbeddingDataset',
        'EmbeddingDatasetConfig',
        
        # Models
        'EmbeddingBert',
        'EmbeddingModelConfig',
        'embedding_model_factory',
        
        # Losses
        'InfoNCELoss',
        'InfoNCEConfig',
        'info_nce_loss_factory',
        
        # Masking
        'MaskingModule',
        'WholeWordMaskingModule',
        'create_attention_mask',
        
        # Training
        'EmbeddingTrainer',
        'EmbeddingTrainerConfig',
        'train_embeddings',
    ]
    
except ImportError as e:
    logger.error(f"Error importing embedding modules: {str(e)}")
    logger.debug(f"Current sys.path: {sys.path}")
    raise