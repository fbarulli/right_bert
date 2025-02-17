# __init__.py
# src/classification/__init__.py (CORRECTED)
from src.classification.model import ClassificationBert
from src.classification.classification_training import train_final_model
from src.classification.dataset import ClassificationDataset
from src.classification.losses import FocalLoss

__all__ = [
    'ClassificationBert',
    'run_classification_optimization',
    'train_final_model',
    'ClassificationDataset',
    'FocalLoss'
]