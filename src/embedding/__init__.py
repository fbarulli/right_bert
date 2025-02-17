# src/embedding/__init__.py
#src/embedding/__init__.py (CORRECTED)
"""Embedding module for BERT model training."""
from src.embedding.model import EmbeddingBert
from src.embedding.embedding_trainer import EmbeddingTrainer
from src.embedding.embedding_training import train_embeddings, validate_embeddings
from src.embedding.dataset import EmbeddingDataset
from src.embedding.masking import (
    MaskingModule,
    WholeWordMaskingModule,
    SpanMaskingModule
)
from src.embedding.losses import InfoNCELoss, info_nce_loss_factory

__all__ = [
    'EmbeddingBert',
    'EmbeddingTrainer',
    'train_embeddings',
    'validate_embeddings',
    'EmbeddingDataset',
    'MaskingModule',
    'WholeWordMaskingModule',
    'SpanMaskingModule',
    'InfoNCELoss',
    'info_nce_loss_factory',
    'validate_embeddings'
]