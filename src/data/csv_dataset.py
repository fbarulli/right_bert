
# csv_dataset.py
# src/data/csv_dataset.py
from __future__ import annotations
import csv
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from concurrent.futures import ThreadPoolExecutor
import gc
from dataclasses import dataclass

from src.common.utils import get_tensor_manager # Import from utils

logger = logging.getLogger(__name__)

@dataclass
class TokenizedBatch:
    """
    Represents a batch of tokenized data.

    Attributes:
        input_ids: The input token IDs.
        attention_mask: The attention mask.
        special_tokens_mask: Mask indicating special tokens.
        word_ids: Array mapping tokens to their original word indices.
        labels: The labels (if available). Defaults to None.
        index:  The original index of the example. Defaults to None.
    """
    input_ids: np.ndarray
    attention_mask: np.ndarray
    special_tokens_mask: np.ndarray
    word_ids: np.ndarray
    labels: Optional[np.ndarray]
    index: Optional[np.ndarray] = None


def create_memmap_array(filename: str | Path, shape: tuple, dtype: type) -> np.ndarray:
    """
    Creates a memory-mapped array.
    """
    return np.memmap(filename, dtype=dtype, mode='w+', shape=shape)

def load_memmap_array(filename: str | Path, shape: tuple, dtype: type) -> np.ndarray:
    """
    Loads a memory-mapped array.
    """
    return np.memmap(filename, dtype=dtype, mode='r+', shape=shape)


def process_batch_parallel(tokenizer: PreTrainedTokenizerFast, texts: List[str], max_length: int) -> Dict[str, np.ndarray]:
    """
    Process a batch of texts in parallel.
    """
    logger.debug(f"process_batch_parallel called with {len(texts)=} texts")

    def _tokenize_text(text: str) -> List[str]:
        logger.debug(f"Tokenizing text: {text[:50]}...")
        return text.split()

    with ThreadPoolExecutor(max_workers=4) as executor:
        all_words = list(executor.map(_tokenize_text, texts))
        logger.debug(f"Tokenized {len(all_words)} texts into words.")

    encoding = tokenizer(
        all_words,
        is_split_into_words=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np',
        return_special_tokens_mask=True,
        return_offsets_mapping=False
    )
    logger.debug("Completed tokenization with padding and truncation.")


    word_ids = np.array([word_id if word_id is not None else -1 for word_id in encoding.word_ids()], dtype=np.int64)
    logger.debug("Extracted word IDs.")

    result = {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'special_tokens_mask': encoding['special_tokens_mask'],
        'word_ids': word_ids
    }
    logger.debug(f"process_batch_parallel returning: { {k: v.shape for k, v in result.items()} }")
    return result


class CSVDataset(Dataset):
    """
    A PyTorch Dataset for loading data from a CSV file.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int,
        split: str = 'train',
        train_ratio: float = 0.9,
        cache_dir: Union[str, Path] = None

    ):
        logger.debug(f"Initializing CSVDataset with: data_path={data_path}, max_length={max_length}, split={split}, train_ratio={train_ratio}")
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.train_ratio = train_ratio

        if cache_dir is None:
            self.cache_dir = Path(os.getcwd()) / '.cache'
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        logger.debug(f"Using cache directory: {self.cache_dir}")


        data_hash = hash(str(self.data_path.absolute()) + str(max_length))
        self.cache_prefix = self.cache_dir / f"dataset_{data_hash}"
        logger.debug(f"Cache prefix: {self.cache_prefix}")

        input_ids_path = self.cache_prefix.parent / f"{self.cache_prefix.name}_input_ids.npy"
        if not input_ids_path.exists():
            logger.info("Creating memory-mapped arrays for dataset...")
            self._create_memmap_arrays()
        else:
            logger.info("Loading existing memory-mapped arrays...")
            try:
                self._load_memmap_arrays()
            except Exception as e:
                logger.error(f"Error loading memory-mapped arrays: {e}", exc_info=True)
                logger.info("Cleaning up cache and recreating arrays...")
                self._cleanup_cache()
                self._create_memmap_arrays()

        self.split_idx = int(self.total_rows * train_ratio)
        if split == 'train':
            self.start_idx = 0
            self.end_idx = self.split_idx
        else:
            self.start_idx = self.split_idx
            self.end_idx = self.total_rows
        logger.debug(f"Dataset split: {self.split}, start_idx: {self.start_idx}, end_idx: {self.end_idx}")


    def _cleanup_cache(self):
        """Clean up cache files."""
        logger.debug(f"Cleaning up cache files for prefix: {self.cache_prefix}")
        for file in [
            f"{self.cache_prefix}_input_ids.npy",
            f"{self.cache_prefix}_attention_mask.npy",
            f"{self.cache_prefix}_labels.npy",
            f"{self.cache_prefix}_shapes.npy",
            f"{self.cache_prefix}_special_tokens_mask.npy",
            f"{self.cache_prefix}_word_ids.npy"
        ]:
            try:
                Path(file).unlink(missing_ok=True)
                logger.debug(f"Deleted file: {file}")
            except Exception as e:
                logger.warning(f"Failed to delete {file}: {e}", exc_info=True)

    def _create_memmap_arrays(self):
        """Create memory-mapped arrays from scratch."""
        logger.info("Creating memory-mapped arrays...")
        try:
            texts: List[str] = []
            labels: List[int] = []
            with open(self.data_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'rating' in row:
                        try:
                            rating = int(row['rating'])
                            if not 1 <= rating <= 5:
                                logger.error(f"Rating must be between 1-5, got: {rating}")
                                continue
                            label = rating - 1
                            texts.append(row['text'])
                            labels.append(label)
                        except ValueError:
                            logger.error(f"Rating must be an integer, got: {row['rating']}", exc_info=True)
                            continue
                    else:
                        texts.append(row['text'])
                        labels.append(-1)
            logger.debug(f"Read {len(texts)} texts and {len(labels)} labels from CSV.")

            logger.info("Tokenizing texts...")
            batch_size = 1000
            self.total_rows = len(texts)

            input_shape = (self.total_rows, self.max_length)
            labels_shape = (self.total_rows,)

            shapes = {
                'input_ids': input_shape,
                'attention_mask': input_shape,
                'labels': labels_shape,
                'special_tokens_mask': input_shape,
                'word_ids': input_shape

            }
            np.save(f"{self.cache_prefix}_shapes.npy", shapes)
            logger.debug(f"Saved shapes to {self.cache_prefix}_shapes.npy")

            self.input_ids = create_memmap_array(f"{self.cache_prefix}_input_ids.npy", input_shape, dtype=np.int64)
            self.attention_mask = create_memmap_array(f"{self.cache_prefix}_attention_mask.npy", input_shape, dtype=np.int64)
            self.special_tokens_mask = create_memmap_array(f"{self.cache_prefix}_special_tokens_mask.npy", input_shape, dtype=np.int64)
            self.word_ids = create_memmap_array(f"{self.cache_prefix}_word_ids.npy", input_shape, dtype=np.int64)
            self.labels = create_memmap_array(f"{self.cache_prefix}_labels.npy", labels_shape, dtype=np.int64)
            logger.debug(f"Created memory-mapped arrays with prefix: {self.cache_prefix}")

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                logger.debug(f"Processing batch {i // batch_size + 1}/{len(texts) // batch_size + 1}")
                encodings = process_batch_parallel(self.tokenizer, batch_texts, self.max_length)

                self.input_ids[i:i + len(batch_texts)] = encodings['input_ids']
                self.attention_mask[i:i + len(batch_texts)] = encodings['attention_mask']
                self.special_tokens_mask[i:i + len(batch_texts)] = encodings['special_tokens_mask']
                self.word_ids[i:i + len(batch_texts)] = encodings['word_ids']

                self.labels[i:i + len(batch_texts)] = labels[i:i + len(batch_texts)]
                logger.debug(f"Stored batch {i // batch_size + 1} in memory-mapped arrays.")

                gc.collect()
                logger.debug("Garbage collection completed.")

            logger.info(f"Created memory-mapped dataset with {self.total_rows} examples")

        except Exception as e:
            logger.error(f"Error creating memory-mapped arrays: {e}", exc_info=True)
            self._cleanup_cache()
            raise

    def _load_memmap_arrays(self):
        """Load existing memory-mapped arrays."""
        logger.info(f"Loading memory-mapped arrays from {self.cache_prefix}...")
        try:
            shapes = np.load(f"{self.cache_prefix}_shapes.npy", allow_pickle=True).item()
            logger.debug(f"Loaded shapes: {shapes}")

            self.input_ids = load_memmap_array(f"{self.cache_prefix}_input_ids.npy", shapes['input_ids'], dtype=np.int64)
            self.attention_mask = load_memmap_array(f"{self.cache_prefix}_attention_mask.npy", shapes['attention_mask'], dtype=np.int64)
            self.special_tokens_mask = load_memmap_array(f"{self.cache_prefix}_special_tokens_mask.npy", shapes['special_tokens_mask'], dtype=np.int64)
            self.word_ids = load_memmap_array(f"{self.cache_prefix}_word_ids.npy", shapes['word_ids'], dtype=np.int64)
            self.labels = load_memmap_array(f"{self.cache_prefix}_labels.npy", shapes['labels'], dtype=np.int64)
            self.total_rows = shapes['input_ids'][0]
            logger.info(f"Loaded memory-mapped dataset with {self.total_rows} examples")

        except Exception as e:
            logger.error(f"Error loading memory-mapped arrays: {e}", exc_info=True)
            raise

    def __len__(self) -> int:
        len_ = self.end_idx - self.start_idx
        logger.debug(f"__len__ called, returning {len_}")
        return len_

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        logger.debug(f"__getitem__ called with index {idx}")
        idx = idx + self.start_idx
        logger.debug(f"Adjusted index to {idx} (absolute)")
        tensor_manager = get_tensor_manager()

        item = {
            'input_ids': tensor_manager.create_cpu_tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': tensor_manager.create_cpu_tensor(self.attention_mask[idx], dtype=torch.long),
            'special_tokens_mask': tensor_manager.create_cpu_tensor(self.special_tokens_mask[idx], dtype=torch.long),
            'word_ids': tensor_manager.create_cpu_tensor(self.word_ids[idx], dtype=torch.long),
            'index': torch.tensor(idx, dtype=torch.long),
        }
        if self.labels[idx] != -1:
            item['labels'] = tensor_manager.create_cpu_tensor(self.labels[idx], dtype=torch.long)
        logger.debug(f"__getitem__ returning: { {k: v.shape for k, v in item.items()} }")
        return item

def csv_dataset_factory(
    data_path: Union[str, Path],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int,
    split: str = 'train',
    train_ratio: float = 0.9,
    cache_dir: Union[str, Path] = None
) -> CSVDataset:
    """
    Factory function for creating a CSVDataset instance.
    """
    logger.info(f"Creating CSVDataset with factory: data_path={data_path}, max_length={max_length}, split={split}, train_ratio={train_ratio}, cache_dir={cache_dir}")
    try:
        dataset = CSVDataset(data_path, tokenizer, max_length, split, train_ratio, cache_dir)
        logger.info("CSVDataset created successfully.")
        return dataset
    except Exception as e:
        logger.error(f"Error creating CSVDataset: {e}", exc_info=True)
        raise

__all__ = ['CSVDataset', 'TokenizedBatch', 'process_batch_parallel', 'csv_dataset_factory']