# dataset.py
# src/classification/dataset.py
from __future__ import annotations

import logging
from pathlib import Path
from transformers import PreTrainedTokenizerFast
# Ensure correct relative import based on your project structure
from src.data.csv_dataset import CSVDataset  # Corrected import

logger = logging.getLogger(__name__)

class ClassificationDataset(CSVDataset):
    """Dataset for text classification tasks."""

    def __init__(
        self,
        data_path: Path,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
        split: str = 'train',
        train_ratio: float = 0.9
    ):
        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            split=split,
            train_ratio=train_ratio
        )

__all__ = ['ClassificationDataset']