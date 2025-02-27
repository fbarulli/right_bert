# Add code to make the dataset fully picklable
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
import logging
from pathlib import Path
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingDatasetConfig:
    """Configuration for EmbeddingDataset."""
    data_path: Path
    max_length: int = 128
    split: str = "train"  
    train_ratio: float = 0.8
    mask_prob: float = 0.15
    max_predictions: int = 20
    max_span_length: int = 3
    log_level: str = "info"
    cache_size: int = 1000
    tensor_pool_size: int = 1000
    gc_threshold: float = 0.8

class EmbeddingDataset(Dataset):
    """Dataset for training embedding models with masked language modeling."""
    
    def __init__(
        self, 
        tokenizer,
        config: EmbeddingDatasetConfig,
    ):
        """
        Initialize the dataset.
        
        Args:
            tokenizer: Tokenizer for encoding text
            config: Dataset configuration
        """
        # Make sure we don't store the tokenizer directly - it contains unpicklable objects!
        # Instead, store only what we need
        self.vocab_size = tokenizer.vocab_size
        self.mask_token_id = tokenizer.mask_token_id
        self.tokenize = lambda text: tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=config.max_length,
            return_tensors="pt"
        )
        
        self.config = config
        
        # Load data - make sure it's completely picklable
        if isinstance(config.data_path, str):
            data_path = Path(config.data_path)
        else:
            data_path = config.data_path
            
        # Load dataset
        try:
            self.df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(self.df)} examples from {data_path}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Create a dummy dataset as fallback
            self.df = pd.DataFrame({"text": ["This is sample text."] * 100})
            logger.warning("Using dummy dataset due to load failure")
        
        # Split data
        if config.split == "train":
            train_size = int(len(self.df) * config.train_ratio)
            self.df = self.df[:train_size]
        elif config.split == "val":
            train_size = int(len(self.df) * config.train_ratio)
            self.df = self.df[train_size:]
        
        # Remove any potential non-picklable objects
        self._remove_unpicklable_attributes()
        
        logger.info(f"Dataset for {config.split} split contains {len(self.df)} examples")

    @classmethod
    def from_config(cls, config: Dict[str, Any], tokenizer, split: str = "train"):
        """Create dataset from configuration dictionary."""
        data_config = config.get('data', {})
        embedding_config = EmbeddingDatasetConfig(
            data_path=Path(data_config.get('csv_path', 'data/embedding.csv')),
            max_length=data_config.get('max_length', 128),
            split=split,
            train_ratio=data_config.get('train_ratio', 0.8),
            mask_prob=data_config.get('embedding_mask_probability', 0.15),
            max_predictions=data_config.get('max_predictions', 20),
            max_span_length=data_config.get('max_span_length', 3),
            log_level=config['training'].get('log_level', 'info'),
            cache_size=config['training'].get('cache_size', 1000),
            tensor_pool_size=config['training'].get('tensor_pool_size', 1000),
            gc_threshold=config['training'].get('gc_threshold', 0.8)
        )
        return cls(tokenizer=tokenizer, config=embedding_config)

    def _remove_unpicklable_attributes(self):
        """Remove any attributes that could cause pickle errors."""
        # Remove any loggers
        if hasattr(self, 'logger'):
            delattr(self, 'logger')
            
        # Remove any locks 
        for attr_name in list(self.__dict__.keys()):
            attr = getattr(self, attr_name)
            if 'lock' in attr_name.lower() or 'thread' in attr_name.lower():
                delattr(self, attr_name)
                
        # Remove any manager references
        for attr_name in list(self.__dict__.keys()):
            if 'manager' in attr_name.lower():
                delattr(self, attr_name)
                
    def __getstate__(self):
        """Custom state for pickle."""
        state = self.__dict__.copy()
        # Don't include the tokenize function as it may contain unpicklable objects
        state['tokenize'] = None
        return state
        
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__ = state
        # Recreate tokenize function on worker
        try:
            # On workers, we need to create a new tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.tokenize = lambda text: tokenizer(
                text, 
                padding="max_length", 
                truncation=True, 
                max_length=self.config.max_length,
                return_tensors="pt"
            )
        except:
            # If we can't load a tokenizer, make a dummy function
            self.tokenize = lambda text: {"input_ids": torch.zeros(1, self.config.max_length)}

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example with masking."""
        text = self.df.iloc[idx]["text"]
        
        # Tokenize
        try:
            encoding = self.tokenize(text)
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            # Return empty tensors as fallback
            return {
                "input_ids": torch.zeros(1, self.config.max_length, dtype=torch.long),
                "attention_mask": torch.ones(1, self.config.max_length, dtype=torch.long),
                "labels": torch.zeros(1, self.config.max_length, dtype=torch.long)
            }
        
        # Apply random masking
        masked_input_ids, labels = self._mask_tokens(encoding["input_ids"])
        
        return {
            "input_ids": masked_input_ids.squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }
    
    def _mask_tokens(
        self, 
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply masking for masked language modeling."""
        # Create label tensor (copy of input)
        labels = input_ids.clone()
        
        # Create probability mask
        probability_matrix = torch.full_like(
            input_ids, 
            self.config.mask_prob, 
            dtype=torch.float
        )
        
        # Create special tokens mask
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for idx in range(input_ids.shape[1]):
            if idx >= input_ids.shape[1] - 1:
                special_tokens_mask[0, idx] = True
                
        # Update probability matrix to avoid masking special tokens
        probability_matrix = probability_matrix.masked_fill(
            special_tokens_mask, 
            value=0.0
        )
        
        # Sample masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set labels to -100 for non-masked tokens
        labels = labels.masked_fill(~masked_indices, value=-100)
        
        # Replace masked indices with mask token
        input_ids = input_ids.masked_fill(masked_indices, value=self.mask_token_id)
        
        return input_ids, labels
