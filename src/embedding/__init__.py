#src/embedding/__init__.py (CORRECTED)
"""Embedding module for BERT model training."""
from src.embedding.model import EmbeddingBert  # Corrected absolute import
from src.embedding.embedding_trainer import EmbeddingTrainer  # Corrected absolute import
from src.embedding.embedding_training import train_embeddings, validate_embeddings  # Corrected absolute import
from src.embedding.dataset import EmbeddingDataset  # Corrected absolute import
from src.embedding.masking import (  # Corrected absolute import
    MaskingModule,
    WholeWordMaskingModule,
    SpanMaskingModule
)
from src.embedding.losses import InfoNCELoss, info_nce_loss_factory  # Corrected absolute import

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