# src/embedding/masking.py
# src/embedding/masking.py
from __future__ import annotations
import torch
import random
import logging
import os
from typing import Tuple, Optional, List, Set, Dict
from transformers import PreTrainedTokenizerFast

# Removed top-level import of get_tensor_manager

logger = logging.getLogger(__name__)

class MaskingModule:
    """Base class for masking strategies used in embedding learning."""

    MIN_MASK_PROB = 0.1
    MAX_MASK_PROB = 0.3

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        mask_prob: float = 0.15,
        max_predictions: int = 20,
        worker_id: Optional[int] = None
    ):
        """
        Initialize masking module.
        """
        from src.common.utils import get_tensor_manager # Local import
        if not self.MIN_MASK_PROB <= mask_prob <= self.MAX_MASK_PROB:
            logger.warning(
                f"Mask probability {mask_prob} outside recommended range "
                f"[{self.MIN_MASK_PROB}, {self.MAX_MASK_PROB}]"
            )
        self.mask_prob = max(self.MIN_MASK_PROB, min(self.MAX_MASK_PROB, mask_prob))

        self.tokenizer = tokenizer
        self.base_max_predictions = max_predictions
        self.tensor_manager = get_tensor_manager() # Get tensor_manager here locally
        self.max_predictions = max_predictions

        self.special_token_ids = set([
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
            tokenizer.unk_token_id
        ])

        self.valid_vocab_ids = [
            i for i in range(tokenizer.vocab_size)
            if i not in self.special_token_ids
        ]

    def _get_word_boundaries(
        self,
        input_ids: torch.Tensor,
        word_ids: List[Optional[int]]
    ) -> List[Tuple[int, int]]:
        """
        Get word boundaries respecting word pieces.
        """
        word_boundaries = []
        start_idx = None

        for i, word_id in enumerate(word_ids):
            if word_id is None:
                if start_idx is not None:
                    word_boundaries.append((start_idx, i))
                    start_idx = None
                continue

            if start_idx is None:
                start_idx = i
            elif word_id != word_ids[i - 1]:
                word_boundaries.append((start_idx, i))
                start_idx = i

        if start_idx is not None:
            word_boundaries.append((start_idx, len(word_ids)))

        return word_boundaries

    def _get_maskable_boundaries(
        self,
        word_boundaries: List[Tuple[int, int]],
        word_ids: List[Optional[int]],
        max_span_length: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        Get maskable word boundaries, EXCLUDING special tokens.
        """
        maskable = []
        for start, end in word_boundaries:
            if any(word_ids[j] is None for j in range(start, end)):
                continue
            maskable.append((start,end))
        return maskable

    def _apply_token_masking(
        self,
        input_ids: torch.Tensor,
        start_idx: int,
        end_idx: int
    ) -> None:
        """
        Apply token-level masking.
        """
        for idx in range(start_idx, end_idx):
            prob = random.random()
            if prob < 0.8:
                input_ids[idx] = self.tokenizer.mask_token_id
            elif prob < 0.9:
                input_ids[idx] = random.choice(self.valid_vocab_ids)

    def _create_labels(
        self,
        original_ids: torch.Tensor,
        masked_positions: Set[int]
    ) -> torch.Tensor:
        """
        Create masking labels.
        """
        labels = torch.full_like(original_ids, -100)
        for pos in masked_positions:
            labels[pos] = original_ids[pos]
        return labels

class WholeWordMaskingModule(MaskingModule):
    """Whole word masking."""

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies whole word masking to the input batch.
        """
        input_ids = batch['input_ids']
        word_ids = batch['word_ids']
        special_tokens_mask = batch['special_tokens_mask']

        if input_ids.dim() != 1:
            raise ValueError(f"Expected 1D input tensor, got: {input_ids.shape}")

        input_ids = self.tensor_manager.create_cpu_tensor(input_ids.clone(), dtype=torch.long)
        original_ids = input_ids.clone()

        word_ids_list = [None if i == -1 or m == 1 else i for i, m in zip(word_ids.tolist(), special_tokens_mask.tolist())]
        word_boundaries = self._get_word_boundaries(input_ids, word_ids_list)
        maskable_boundaries = self._get_maskable_boundaries(word_boundaries, word_ids_list)

        if not maskable_boundaries:
            logger.warning("No maskable word boundaries found.")
            return input_ids, torch.full_like(input_ids, -100)

        num_to_mask = max(1, int(len(maskable_boundaries) * self.mask_prob))
        words_to_mask = random.sample(maskable_boundaries, num_to_mask)

        masked_positions = set()
        for start, end in words_to_mask:
            self._apply_token_masking(input_ids, start, end)
            masked_positions.update(range(start, end))

        labels = self._create_labels(original_ids, masked_positions)
        return input_ids, labels

class SpanMaskingModule(MaskingModule):
    """Span-based masking, prioritizing longer spans, up to a target ratio."""

    MIN_SPAN_LENGTH = 1
    MAX_SPAN_LENGTH = 10

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        mask_prob: float = 0.15,
        max_span_length: int = 10,
        max_predictions: int = 20,
        worker_id: Optional[int] = None
    ):
        """
        Initialize SpanMaskingModule.
        """
        super().__init__(tokenizer, mask_prob, max_predictions, worker_id)
        if not self.MIN_SPAN_LENGTH <= max_span_length <= self.MAX_SPAN_LENGTH:
            logger.warning(f"max_span_length {max_span_length} outside recommended range")
        self.max_span_length = max_span_length

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies span masking.
        """
        input_ids = batch['input_ids']
        word_ids = batch['word_ids']
        special_tokens_mask = batch['special_tokens_mask']

        if input_ids.dim() != 1:
            raise ValueError(f"Expected 1D input tensor, got shape: {input_ids.shape}")

        input_ids = self.tensor_manager.create_cpu_tensor(input_ids.clone(), dtype=torch.long)
        original_ids = input_ids.clone()

        word_ids_list = [None if id == -1 or mask == 1 else id for id, mask in zip(word_ids.tolist(), special_tokens_mask.tolist())]


        word_boundaries = self._get_word_boundaries(input_ids, word_ids_list)
        maskable_boundaries = self._get_maskable_boundaries(word_boundaries, word_ids_list, self.max_span_length)

        if not maskable_boundaries:
            return input_ids, torch.full_like(input_ids, -100)


        seq_length = len(input_ids)
        target_masked = int(seq_length * self.mask_prob)

        token_counts = {}
        for start, end in maskable_boundaries:
            for i in range(start, end):
                token_counts[i] = token_counts.get(i, 0) + 1

        processed_boundaries = []
        for i, (start, end, length, overlap_score) in enumerate(maskable_boundaries):
            length = end - start
            overlap_score = 0
            for j in range(start,end):
                for start_other, end_other in maskable_boundaries:
                    if start_other <= j < end_other and (start,end) != (start_other, end_other):
                        overlap_score +=1
                        break
            processed_boundaries.append((i, start, end, length, overlap_score))

        processed_boundaries.sort(key=lambda x: (x[3], x[4]), reverse=True)

        masked_positions = set()
        masked_count = 0

        for idx, start, end, length, overlap_score in processed_boundaries:
            for i in range(start, end):
                if masked_count < target_masked and i not in masked_positions:
                    self._apply_token_masking(input_ids, i, i + 1)
                    masked_positions.add(i)
                    masked_count += 1

        labels = self._create_labels(original_ids, masked_positions)
        num_masked = len(masked_positions)
        mask_ratio = num_masked / seq_length
        logger.info(
            f"Masking results for index {batch.get('index', 'N/A')}:\n"
            f"- Mask ratio achieved: {mask_ratio:.2%}\n"
            f"- Total tokens: {len(input_ids)}\n"
            f"- Masked tokens: {num_masked}\n"
            f"- Sequence length: {seq_length}"
         )
        return input_ids, labels

def create_attention_mask(input_ids: torch.Tensor, padding_idx: int = 0) -> torch.Tensor:
    """Create attention mask from input ids."""
    return (input_ids != padding_idx).float()