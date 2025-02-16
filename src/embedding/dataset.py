# src/embedding/dataset.py
from __future__ import annotations
import torch
import random
import logging
import os
from pathlib import Path
from typing import Tuple, Optional, List, Set, Dict
from transformers import PreTrainedTokenizerFast

from src.common.managers import (  # Absolute import
    get_tensor_manager,
    get_tokenizer_manager
)
from src.data.csv_dataset import CSVDataset  # Absolute import
from src.embedding.masking import SpanMaskingModule  # Absolute import

logger = logging.getLogger(__name__)

class EmbeddingDataset(CSVDataset):
    """Dataset for learning embeddings through masked token prediction."""

    def __init__(
        self,
        data_path: Path,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 100,
        split: str = 'train',
        train_ratio: float = 0.9,
        mask_prob: float = 0.15,
        max_predictions: int = 20,
        max_span_length: int = 1
    ):
        """
        Initialize the EmbeddingDataset.

        Args:
            data_path (Path): Path to the CSV data file.
            tokenizer (PreTrainedTokenizerFast): The Hugging Face tokenizer.
            max_length (int): The maximum sequence length.
            split (str): 'train' or 'val'.
            train_ratio (float): The ratio of data to use for training.
            mask_prob (float): The probability of masking a token.
            max_predictions (int): The maximum number of predictions to make.
            max_span_length (int): The maximum length of a masked span.

        """
        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            split=split,
            train_ratio=train_ratio
        )
        self.mask_prob = mask_prob
        self.max_predictions = max_predictions
        self.max_span_length = max_span_length
        # Initialize masking module
        self.masking_module = SpanMaskingModule(
            tokenizer=self.tokenizer,
            mask_prob=self.mask_prob,
            max_span_length=self.max_span_length,
            max_predictions=self.max_predictions,
            worker_id=os.getpid()
        )

    def _mask_tokens(self, item: Dict[str, torch.Tensor], idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply masking using the SpanMaskingModule.

        Args:
            item: Dictionary containing input tensors
            idx: Index into the dataset

        Returns:
            Tuple of (masked inputs, labels) both of shape [seq_len]
        """
        if item['input_ids'].dim() != 1:
            raise ValueError(f"Expected 1D input tensor, got shape: {item['input_ids'].shape}")

        # Apply masking using the module (returns CPU tensors)
        masked_inputs, labels = self.masking_module(item)

        return masked_inputs, labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset with masking applied.

        Args:
            idx: Index into the dataset split

        Returns:
            Dictionary containing masked input_ids, attention_mask, and labels
        """
        # Get base item from parent class
        item = super().__getitem__(idx)

        # Apply masking and log details

        logger.debug(
            f"Applying masking for index {idx}\n"
            f"- Input length: {len(item['input_ids'])}\n"
            f"- Target mask prob: {self.mask_prob:.2%}\n"
            f"- Max span length: {self.max_span_length}"
        )

        input_ids, embedding_labels = self._mask_tokens(item, idx)

        # Log masking results
        mask = embedding_labels != -100
        mask_ratio = mask.sum().item() / len(embedding_labels)
        '''
        logger.info(
            f"Masking results for index {idx}:\n"
            f"- Mask ratio achieved: {mask_ratio:.2%}\n"
            f"- Total tokens: {len(embedding_labels)}\n"
            f"- Masked tokens: {mask.sum().item()}\n"
            f"- Sequence length: {len(item['input_ids'])}"
        )'''

        item['input_ids'] = input_ids
        item['labels'] = embedding_labels

        return item

__all__ = ['EmbeddingDataset']