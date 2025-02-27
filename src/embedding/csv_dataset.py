"""
Base CSV dataset implementation for loading and preprocessing text data.
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, List, Union, Tuple
from pathlib import Path
import logging
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

class CSVDataset(Dataset):
    """Dataset for loading data from CSV files."""

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 128,
        split: str = 'train',
        train_ratio: float = 0.9,
        text_column: str = 'text',
        id_column: Optional[str] = None,
        filter_column: Optional[str] = None,
        filter_value: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize dataset from CSV file.
        
        Args:
            data_path: Path to CSV file or directory containing CSVs
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            split: Which split to use ('train' or 'val')
            train_ratio: Fraction of data to use for training
            text_column: Column name for text data
            id_column: Column name for IDs (optional)
            filter_column: Column to filter on (optional)
            filter_value: Value to filter for (optional)
            **kwargs: Additional arguments
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.train_ratio = train_ratio
        self.text_column = text_column
        self.id_column = id_column
        
        # Load dataset
        self.df = self._load_data()
        
        # Apply filter if specified
        if filter_column and filter_value:
            self.df = self.df[self.df[filter_column] == filter_value]
        
        # Create train/val split
        self._create_split()
        
        # Cache loaded examples
        self.examples = {}
        
        logger.info(f"Initialized CSVDataset with {len(self)} examples for split '{split}'")
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file(s).
        
        Returns:
            pd.DataFrame: Loaded data
        """
        if self.data_path.is_dir():
            # Load multiple CSV files
            dfs = []
            for file in self.data_path.glob("*.csv"):
                try:
                    df = pd.read_csv(file)
                    if self.text_column not in df.columns:
                        logger.warning(f"CSV file {file} does not contain column '{self.text_column}'. Skipping.")
                        continue
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Error loading {file}: {str(e)}")
            
            if not dfs:
                raise ValueError(f"No valid CSV files found in {self.data_path}")
            
            return pd.concat(dfs, ignore_index=True)
        else:
            # Load single CSV file
            if not self.data_path.exists():
                raise FileNotFoundError(f"CSV file not found: {self.data_path}")
            
            df = pd.read_csv(self.data_path)
            if self.text_column not in df.columns:
                raise ValueError(f"CSV file does not contain column '{self.text_column}'")
            
            return df
    
    def _create_split(self) -> None:
        """Create train/val split indices."""
        n = len(self.df)
        indices = torch.randperm(n).tolist()
        split_idx = int(n * self.train_ratio)
        
        if self.split == 'train':
            self.indices = indices[:split_idx]
        elif self.split == 'val':
            self.indices = indices[split_idx:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
    
    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize text sample.
        
        Args:
            text: Input text
            
        Returns:
            Dict[str, torch.Tensor]: Tokenized inputs
        """
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        return {k: v.squeeze(0) for k, v in encoded.items()}
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Index
            
        Returns:
            Dict[str, torch.Tensor]: Tokenized inputs
        """
        if idx in self.examples:
            return self.examples[idx]
        
        # Get original index
        orig_idx = self.indices[idx]
        row = self.df.iloc[orig_idx]
        
        # Get text
        text = row[self.text_column]
        
        # Tokenize
        encoded = self._tokenize(text)
        
        # Add ID if available
        if self.id_column:
            encoded['id'] = row[self.id_column]
        
        # Cache result
        self.examples[idx] = encoded
        
        return encoded
