"""
Embedding module for learning token embeddings through masked language modeling.

This module provides:
- Dataset handling for embedding tasks
- Masking strategies
- Loss functions
- Model definitions
- Training utilities
"""

from src.embedding.model import (
    EmbeddingBert,
    EmbeddingModelConfig,
    embedding_model_factory,
)

from src.embedding.losses import (
    InfoNCELoss,
    InfoNCEConfig,
    info_nce_loss_factory,
)

from src.embedding.masking import (
    MaskingModule,
    MaskingConfig,
    WholeWordMaskingModule,
    SpanMaskingModule,
    create_attention_mask,
)

from src.embedding.dataset import (
    EmbeddingDataset,
    EmbeddingDatasetConfig,
)

from src.embedding.embedding_trainer import (
    EmbeddingTrainer,
    EmbeddingTrainerConfig,
)

from src.embedding.embedding_training import (
    train_embeddings,
)

__all__ = [
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
    'MaskingConfig',
    'WholeWordMaskingModule',
    'SpanMaskingModule',
    'create_attention_mask',
    
    # Dataset
    'EmbeddingDataset',
    'EmbeddingDatasetConfig',
    
    # Training
    'EmbeddingTrainer',
    'EmbeddingTrainerConfig',
    'train_embeddings',
]