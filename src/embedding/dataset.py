"""
Dataset implementation for embedding learning with masking functionality.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple
import torch  # Add this missing import
from torch import Tensor
import logging
import os

from src.embedding.csv_dataset import CSVDataset
from src.embedding.masking import SpanMaskingModule, MaskingConfig
from src.common.logging_utils import log_function

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingDatasetConfig:
    """Configuration for embedding dataset."""
    data_path: Path
    max_length: int
    split: str
    train_ratio: float
    mask_prob: float
    max_predictions: int
    max_span_length: int
    log_level: str = 'log'
    cache_size: int = 1000
    tensor_pool_size: int = 1000
    gc_threshold: float = 0.8

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
        if self.max_length < 1:
            raise ValueError(f"Max length must be positive, got {self.max_length}")
        if not 0 < self.train_ratio < 1:
            raise ValueError(f"Train ratio must be between 0 and 1, got {self.train_ratio}")
        if self.split not in ['train', 'val']:
            raise ValueError(f"Split must be 'train' or 'val', got {self.split}")
        if not 0 < self.mask_prob < 1:
            raise ValueError(f"Mask probability must be between 0 and 1, got {self.mask_prob}")
        if self.max_predictions < 1:
            raise ValueError(f"Max predictions must be positive, got {self.max_predictions}")
        if self.max_span_length < 1:
            raise ValueError(f"Max span length must be positive, got {self.max_span_length}")
        if self.log_level not in ['debug', 'log', 'none']:
            raise ValueError(f"Invalid log level: {self.log_level}")

class EmbeddingDataset(CSVDataset):
    """Dataset for learning embeddings through masked token prediction."""

    def __init__(
        self,
        tokenizer,  # Type hint removed due to lazy import
        config: EmbeddingDatasetConfig
    ) -> None:
        """
        Initialize the EmbeddingDataset.

        Args:
            tokenizer: The Hugging Face tokenizer
            config: Dataset configuration
        """
        from transformers import PreTrainedTokenizerFast
        
        # Store config as instance attribute
        self.config = config
        
        # Store tokenizer and create tokenize method
        self.tokenizer = tokenizer
        self.tokenize = lambda text: tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt"
        )
        
        # Try both import methods to ensure one succeeds
        try:
            # First try direct import
            from src.embedding.utils import LogConfig, TensorPool, MemoryTracker, CachingDict
        except ImportError:
            # If that fails, try relative import
            from .utils import LogConfig, TensorPool, MemoryTracker, CachingDict

        # Validate tokenizer type at runtime
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise TypeError("Tokenizer must be a PreTrainedTokenizerFast instance")

        # Initialize parent CSVDataset
        super().__init__(
            data_path=config.data_path,
            tokenizer=tokenizer,
            max_length=config.max_length,
            split=config.split,
            train_ratio=config.train_ratio
        )

        self.log_config = LogConfig(level=config.log_level)

        # Initialize memory management
        self.memory_tracker = MemoryTracker(
            gc_threshold=config.gc_threshold,
            log_config=self.log_config
        )
        self.tensor_pool = TensorPool(max_size=config.tensor_pool_size)
        self.cache = CachingDict(
            maxsize=config.cache_size,
            memory_tracker=self.memory_tracker
        )

        # Create masking module configuration
        masking_config = MaskingConfig(
            mask_prob=config.mask_prob,
            max_predictions=config.max_predictions,
            max_span_length=config.max_span_length,
            worker_id=os.getpid(),
            log_level=config.log_level
        )

        # Initialize masking module
        self.masking_module = SpanMaskingModule(
            tokenizer=tokenizer,
            config=masking_config
        )

        logger.info(
            f"Initialized EmbeddingDataset with:\n"
            f"- Data path: {config.data_path}\n"
            f"- Split: {config.split}\n"
            f"- Max length: {config.max_length}\n"
            f"- Mask probability: {config.mask_prob:.2%}\n"
            f"- Max predictions: {config.max_predictions}\n"
            f"- Max span length: {config.max_span_length}\n"
            f"- Cache size: {config.cache_size}\n"
            f"- Tensor pool size: {config.tensor_pool_size}\n"
            f"- Log level: {config.log_level}"
        )

    @log_function()
    def _mask_tokens(
        self,
        item: Dict[str, Tensor],
        idx: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply masking using the SpanMaskingModule.
        
        Args:
            item: Input item dictionary
            idx: Index of the item
            
        Returns:
            Tuple of (masked_inputs, labels)
            
        Raises:
            ValueError: If input tensor is not 1D
        """
        try:
            # Check cache first
            cache_key = f"masked_{idx}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            input_ids = item['input_ids']
            if input_ids.dim() != 1:
                raise ValueError(f"Expected 1D input tensor, got shape: {input_ids.shape}")

            # Get tensors from pool
            masked_inputs = self.tensor_pool.get(
                input_ids.shape,
                input_ids.dtype,
                input_ids.device
            )
            labels = self.tensor_pool.get(
                input_ids.shape,
                input_ids.dtype,
                input_ids.device
            )

            # Apply masking
            temp_masked, temp_labels = self.masking_module(item)
            masked_inputs.copy_(temp_masked)
            labels.copy_(temp_labels)

            # Cache results
            self.cache.set(cache_key, (masked_inputs, labels))

            if hasattr(self.log_config, 'level') and self.log_config.level == 'debug':
                logger.debug(
                    f"Masking details for item {idx}:\n"
                    f"- Original shape: {input_ids.shape}\n"
                    f"- Masked shape: {masked_inputs.shape}\n"
                    f"- Labels shape: {labels.shape}\n"
                    f"- Num masked tokens: {(labels != -100).sum().item()}"
                )

            return masked_inputs, labels

        finally:
            self.memory_tracker.update()

    @log_function()
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Get a single item from the dataset with masking applied.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            Dictionary containing the processed item with masking applied
        """
        try:
            # Check cache first
            cache_key = f"item_{idx}"
            cached_item = self.cache.get(cache_key)
            if cached_item is not None:
                return cached_item

            # Get base item
            item = super().__getitem__(idx)
            input_ids, embedding_labels = self._mask_tokens(item, idx)

            # Update item with masked versions
            item['input_ids'] = input_ids
            item['labels'] = embedding_labels

            # Cache the processed item
            self.cache.set(cache_key, item)

            if hasattr(self.log_config, 'level') and self.log_config.level == 'debug':
                logger.debug(
                    f"Processed item {idx}:\n"
                    f"- Input shape: {input_ids.shape}\n"
                    f"- Labels shape: {embedding_labels.shape}\n"
                    f"- Input sample: {input_ids[:5].tolist()}\n"
                    f"- Labels sample: {embedding_labels[:5].tolist()}\n"
                    f"- Memory stats: {self.memory_tracker.get_stats()}\n"
                    f"- Cache stats: {self.cache.get_stats()}"
                )

            return item

        finally:
            self.memory_tracker.update()

    def __del__(self):
        """Clean up resources when the dataset is deleted."""
        try:
            # Add safety check for tensor_pool attribute
            if hasattr(self, 'tensor_pool'):
                self.tensor_pool.clear()
        except Exception as e:
            # Quietly handle deletion errors to prevent console spam
            pass

    @classmethod
    @log_function()
    def from_config(
        cls,
        config: Dict[str, Any],
        tokenizer,
        split: str
    ) -> 'EmbeddingDataset':
        """
        Create dataset from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            tokenizer: The tokenizer to use
            split: Which split to load ('train' or 'val')
            
        Returns:
            Initialized dataset
        """
        from pathlib import Path

        dataset_config = EmbeddingDatasetConfig(
            data_path=Path(config['data']['csv_path']),
            max_length=config['data']['max_length'],
            split=split,
            train_ratio=config['data']['train_ratio'],
            mask_prob=config['data']['embedding_mask_probability'],
            max_predictions=config['data']['max_predictions'],
            max_span_length=config['data']['max_span_length'],
            log_level=config['training'].get('log_level', 'log'),
            cache_size=config['training'].get('cache_size', 1000),
            tensor_pool_size=config['training'].get('tensor_pool_size', 1000),
            gc_threshold=config['training'].get('gc_threshold', 0.8)
        )
        
        return cls(tokenizer=tokenizer, config=dataset_config)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example with masking."""
        text = self.df.iloc[idx]["text"]
        
        # Tokenize
        try:
            encoding = self.tokenize(text)
            
            # Add word_ids if not present
            if 'word_ids' not in encoding:
                # Generate simple word_ids (one word ID per token)
                input_ids = encoding["input_ids"]
                
                # Create word boundaries based on spaces in original text
                words = text.split()
                word_to_tokens = {}
                current_word_idx = 0
                current_token_idx = 1  # Start after CLS token
                
                # Simple heuristic for word_ids: 
                # Each token gets assigned to a word index based on order
                word_ids = torch.zeros_like(input_ids)
                for i in range(1, input_ids.size(1) - 1):  # Skip CLS and SEP
                    word_ids[0, i] = min(current_word_idx, len(words) - 1)
                    if i % 1.5 == 0:  # Crude approximation - advance word every ~1.5 tokens
                        current_word_idx += 1
                        
                encoding["word_ids"] = word_ids
                
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "word_ids": encoding.get("word_ids", torch.zeros_like(encoding["input_ids"])).squeeze(0)
            }
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            # Return empty tensors as fallback
            return {
                "input_ids": torch.zeros(self.config.max_length, dtype=torch.long),
                "attention_mask": torch.ones(self.config.max_length, dtype=torch.long),
                "word_ids": torch.zeros(self.config.max_length, dtype=torch.long)
            }

__all__ = ['EmbeddingDataset', 'EmbeddingDatasetConfig']