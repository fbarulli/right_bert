# src/embedding/masking.py
from __future__ import annotations
import torch
import random
import logging
from typing import Tuple, Optional, List, Set, Dict
from transformers import PreTrainedTokenizerFast

from src.common.managers import (  # Absolute import
    get_tensor_manager,
    get_tokenizer_manager
)

# Get manager instances (assuming these are defined in src/common/managers.py)
tensor_manager = get_tensor_manager()
tokenizer_manager = get_tokenizer_manager()

logger = logging.getLogger(__name__)

class MaskingModule:
    """Base class for masking strategies used in embedding learning."""

    # Hyperparameter ranges from config
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

        Args:
            tokenizer (PreTrainedTokenizerFast): BERT tokenizer.
            mask_prob (float): Probability of masking each word (between 0.1 and 0.3).
            max_predictions (int): Maximum number of tokens to mask.
            worker_id (Optional[int]): Optional worker ID for process-specific resources.
        """
        # Validate mask probability
        if not self.MIN_MASK_PROB <= mask_prob <= self.MAX_MASK_PROB:
            logger.warning(
                f"Mask probability {mask_prob} outside recommended range "
                f"[{self.MIN_MASK_PROB}, {self.MAX_MASK_PROB}]"
            )
        self.mask_prob = max(self.MIN_MASK_PROB, min(self.MAX_MASK_PROB, mask_prob))

        self.tokenizer = tokenizer
        self.base_max_predictions = max_predictions
        self.max_predictions = max_predictions

        # Get special token IDs
        self.special_token_ids = set([
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
            tokenizer.unk_token_id
        ])

        # Get valid vocabulary for random token selection
        self.valid_vocab_ids = [
            i for i in range(tokenizer.vocab_size)
            if i not in self.special_token_ids
        ]
        #Removed prints

    def _get_word_boundaries(
        self,
        input_ids: torch.Tensor,
        word_ids: List[Optional[int]]
    ) -> List[Tuple[int, int]]:
        """
        Get word boundaries respecting word pieces.  Correctly handles
        subword tokens that are part of the same word.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            word_ids (List[Optional[int]]): List of word IDs, with None for special tokens/padding.

        Returns:
            List[Tuple[int, int]]: A list of (start, end) tuples representing word boundaries.
        """
        word_boundaries = []
        start_idx = None

        for i, word_id in enumerate(word_ids):
            if word_id is None:  # Handle special tokens and padding
                if start_idx is not None:
                    word_boundaries.append((start_idx, i))
                    start_idx = None
                continue

            if start_idx is None:  # Start of a new word
                start_idx = i
            elif word_id != word_ids[i - 1]:  # End of the *current* word (ID change)
                word_boundaries.append((start_idx, i))
                start_idx = i  # *Crucially* reset start_idx

        # Handle the last word (if any)
        if start_idx is not None:
            word_boundaries.append((start_idx, len(word_ids)))

        return word_boundaries

    def _get_maskable_boundaries(
        self,
        word_boundaries: List[Tuple[int, int]],
        word_ids: List[Optional[int]],
        max_span_length: Optional[int] = None  # Not used in this basic version
    ) -> List[Tuple[int, int]]:
        """
        Get maskable word boundaries, EXCLUDING any that contain special tokens.

        Args:
            word_boundaries (List[Tuple[int, int]]): Word boundaries from _get_word_boundaries.
            word_ids (List[Optional[int]]): List of word IDs.
            max_span_length (Optional[int]): Maximum span length (not used here).

        Returns:
            List[Tuple[int, int]]:  Maskable boundaries.
        """
        maskable = []
        for start, end in word_boundaries:
            # CRITICAL: Check for special tokens within the boundary.
            if any(word_ids[j] is None for j in range(start, end)):
                continue  # Skip this boundary if it contains a special token
            maskable.append((start,end))
        return maskable

    def _apply_token_masking(
        self,
        input_ids: torch.Tensor,
        start_idx: int,
        end_idx: int
    ) -> None:
        """
        Apply token-level masking (80% mask, 10% random, 10% unchanged).

        Args:
            input_ids (torch.Tensor): The input token IDs (modified in place).
            start_idx (int): Start index of the span to mask.
            end_idx (int): End index of the span to mask.
        """
        for idx in range(start_idx, end_idx):
            prob = random.random()
            if prob < 0.8:  # 80% mask token
                input_ids[idx] = self.tokenizer.mask_token_id
            elif prob < 0.9:  # 10% random token (excluding special tokens)
                input_ids[idx] = random.choice(self.valid_vocab_ids)
            # 10% unchanged (implicitly)

    def _create_labels(
        self,
        original_ids: torch.Tensor,
        masked_positions: Set[int]
    ) -> torch.Tensor:
        """
        Create masking labels (-100 for unmasked, original ID for masked).

        Args:
            original_ids (torch.Tensor): Original input token IDs.
            masked_positions (Set[int]): Set of masked token positions.

        Returns:
            torch.Tensor:  Labels tensor (-100 for unmasked, original ID for masked).
        """
        labels = torch.full_like(original_ids, -100)  # Initialize with -100
        for pos in masked_positions:
            labels[pos] = original_ids[pos]  # Original ID at masked positions
        return labels

class WholeWordMaskingModule(MaskingModule):
    """Whole word masking (mask all tokens in a randomly chosen word)."""

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies whole word masking to the input batch.

        Args:
            batch (Dict[str, torch.Tensor]): Input batch containing 'input_ids', 'word_ids', and 'special_tokens_mask'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Masked input IDs and labels.
        """
        input_ids = batch['input_ids']
        word_ids = batch['word_ids']
        special_tokens_mask = batch['special_tokens_mask']

        if input_ids.dim() != 1:
            raise ValueError(f"Expected 1D input tensor, got: {input_ids.shape}")

        input_ids = tensor_manager.create_cpu_tensor(input_ids.clone(), dtype=torch.long)
        original_ids = input_ids.clone()

        # Convert word_ids to list, handling special tokens/padding
        word_ids_list = [None if i == -1 or m == 1 else i for i, m in zip(word_ids.tolist(), special_tokens_mask.tolist())]
        word_boundaries = self._get_word_boundaries(input_ids, word_ids_list)
        maskable_boundaries = self._get_maskable_boundaries(word_boundaries, word_ids_list)

        if not maskable_boundaries:
            logger.warning("No maskable word boundaries found.")
            return input_ids, torch.full_like(input_ids, -100)  # No masking

        # Choose words to mask (up to mask_prob)
        num_to_mask = max(1, int(len(maskable_boundaries) * self.mask_prob))
        words_to_mask = random.sample(maskable_boundaries, num_to_mask)

        masked_positions = set()
        for start, end in words_to_mask:
            self._apply_token_masking(input_ids, start, end)  # Mask the entire word
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
        max_span_length: int = 10,  # Default, can be overridden.
        max_predictions: int = 20,
        worker_id: Optional[int] = None
    ):
        """
        Initialize SpanMaskingModule.

        Args:
            tokenizer (PreTrainedTokenizerFast): BERT tokenizer.
            mask_prob (float): Target masking probability.
            max_span_length (int): Maximum length of a masked span.
            max_predictions (int): Maximum number of predictions (masked tokens).
            worker_id (Optional[int]): Worker ID (for process isolation).
        """
        super().__init__(tokenizer, mask_prob, max_predictions, worker_id)
        if not self.MIN_SPAN_LENGTH <= max_span_length <= self.MAX_SPAN_LENGTH:
            logger.warning(f"max_span_length {max_span_length} outside recommended range")
        self.max_span_length = max_span_length # Use provided value


    def __call__(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies span masking.

        Args:
            batch (Dict[str, torch.Tensor]): Input batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Masked input IDs and labels.

        """
        input_ids = batch['input_ids']
        word_ids = batch['word_ids']
        special_tokens_mask = batch['special_tokens_mask']

        if input_ids.dim() != 1:
            raise ValueError(f"Expected 1D input tensor, got shape: {input_ids.shape}")

        # Use CPU tensors for masking operations (avoids device issues)
        input_ids = tensor_manager.create_cpu_tensor(input_ids.clone(), dtype=torch.long)
        original_ids = input_ids.clone() # Keep a copy for labels

        word_ids_list = [None if id == -1 or mask == 1 else id for id, mask in zip(word_ids.tolist(), special_tokens_mask.tolist())]


        word_boundaries = self._get_word_boundaries(input_ids, word_ids_list)
        maskable_boundaries = self._get_maskable_boundaries(word_boundaries, word_ids_list, self.max_span_length)

        if not maskable_boundaries:
            # logger.warning("No maskable word boundaries found") #Removed
            return input_ids, torch.full_like(input_ids, -100) # Return original and -100 labels


        seq_length = len(input_ids)
        target_masked = int(seq_length * self.mask_prob) # Calculate target

        # Analyze and pre-process boundaries (as before)
        token_counts = {}
        for start, end in maskable_boundaries:
            for i in range(start, end):
                token_counts[i] = token_counts.get(i, 0) + 1

        processed_boundaries = []
        for i, (start, end) in enumerate(maskable_boundaries):
            length = end - start
            overlap_score = 0
            for j in range(start,end): #For each token in boundary
                for start_other, end_other in maskable_boundaries: #Check all other boundaries
                    if start_other <= j < end_other and (start,end) != (start_other, end_other): #If it overlaps
                        overlap_score +=1 #Increment
                        break
            processed_boundaries.append((i, start, end, length, overlap_score))

        # Sort by length (descending) then overlap (also descending)
        processed_boundaries.sort(key=lambda x: (x[3], x[4]), reverse=True)

        masked_positions = set()
        masked_count = 0

        # Iterate and mask until target is reached or we run out of boundaries
        for idx, start, end, length, overlap_score in processed_boundaries:
            #NEW LOOP
            for i in range(start, end):  # Iterate through *tokens* in the boundary
                if masked_count < target_masked and i not in masked_positions:
                    self._apply_token_masking(input_ids, i, i + 1)  # Mask *single token*
                    masked_positions.add(i)
                    masked_count += 1

        labels = self._create_labels(original_ids, masked_positions) # Create labels
        num_masked = len(masked_positions)
        mask_ratio = num_masked / seq_length
        logger.info(
            f"Masking results for index {batch.get('index', 'N/A')}:\n"  # Added batch index
            f"- Mask ratio achieved: {mask_ratio:.2%}\n"
            f"- Total tokens: {len(input_ids)}\n"
            f"- Masked tokens: {num_masked}\n"
            f"- Sequence length: {seq_length}"
         )
        return input_ids, labels

def create_attention_mask(input_ids: torch.Tensor, padding_idx: int = 0) -> torch.Tensor:
    """Create attention mask from input ids."""
    return (input_ids != padding_idx).float()