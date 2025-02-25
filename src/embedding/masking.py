"""
Masking strategies for embedding learning, including word and span-based masking.
"""
from src.embedding.imports import (
    torch,
    random,
    os,
    dataclass,
    Tuple, Optional, List, Set, Dict,
    Tensor,
    PreTrainedTokenizerFast,
    logger,
    log_function,
    LogConfig,
)

@dataclass
class MaskingConfig:
    """Configuration for masking modules."""
    mask_prob: float
    max_predictions: int
    max_span_length: int = 1
    worker_id: Optional[int] = None
    log_level: str = 'log'

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0 < self.mask_prob < 1:
            raise ValueError(f"Mask probability must be between 0 and 1, got {self.mask_prob}")
        if self.max_predictions < 1:
            raise ValueError(f"Max predictions must be positive, got {self.max_predictions}")
        if self.max_span_length < 1:
            raise ValueError(f"Max span length must be positive, got {self.max_span_length}")
        if self.log_level not in ['debug', 'log', 'none']:
            raise ValueError(f"Invalid log level: {self.log_level}")

class MaskingModule:
    """Base class for masking strategies used in embedding learning."""

    MIN_MASK_PROB = 0.1
    MAX_MASK_PROB = 0.3

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        config: MaskingConfig
    ) -> None:
        """Initialize masking module."""
        from src.common import get_tensor_manager  # Local import
        
        if not self.MIN_MASK_PROB <= config.mask_prob <= self.MAX_MASK_PROB:
            logger.warning(
                f"Mask probability {config.mask_prob} outside recommended range "
                f"[{self.MIN_MASK_PROB}, {self.MAX_MASK_PROB}]"
            )
        self.mask_prob = max(self.MIN_MASK_PROB, min(self.MAX_MASK_PROB, config.mask_prob))

        self.tokenizer = tokenizer
        self.max_predictions = config.max_predictions
        self.tensor_manager = get_tensor_manager()
        self.log_config = LogConfig(level=config.log_level)

        # Special tokens
        self.special_token_ids = {
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
            tokenizer.unk_token_id
        }

        # Valid vocabulary IDs (excluding special tokens)
        self.valid_vocab_ids = [
            i for i in range(tokenizer.vocab_size)
            if i not in self.special_token_ids
        ]

        logger.info(
            f"Initialized masking module with:\n"
            f"- Mask probability: {self.mask_prob}\n"
            f"- Max predictions: {self.max_predictions}\n"
            f"- Log level: {self.log_config.level}"
        )

    @log_function()
    def _get_word_boundaries(
        self,
        input_ids: Tensor,
        word_ids: List[Optional[int]]
    ) -> List[Tuple[int, int]]:
        """Get word boundaries respecting word pieces."""
        word_boundaries: List[Tuple[int, int]] = []
        start_idx: Optional[int] = None

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

    @log_function()
    def _get_maskable_boundaries(
        self,
        word_boundaries: List[Tuple[int, int]],
        word_ids: List[Optional[int]],
        max_span_length: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """Get maskable word boundaries, excluding special tokens."""
        return [
            (start, end) for start, end in word_boundaries
            if not any(word_ids[j] is None for j in range(start, end))
        ]

    @log_function()
    def _apply_token_masking(
        self,
        input_ids: Tensor,
        start_idx: int,
        end_idx: int
    ) -> None:
        """Apply token-level masking."""
        for idx in range(start_idx, end_idx):
            prob = random.random()
            if prob < 0.8:  # 80% chance to mask
                input_ids[idx] = self.tokenizer.mask_token_id
            elif prob < 0.9:  # 10% chance to replace with random token
                input_ids[idx] = random.choice(self.valid_vocab_ids)
            # 10% chance to keep original token

    @log_function()
    def _create_labels(
        self,
        original_ids: Tensor,
        masked_positions: Set[int]
    ) -> Tensor:
        """Create masking labels."""
        labels = torch.full_like(original_ids, -100)
        for pos in masked_positions:
            labels[pos] = original_ids[pos]
        return labels

class WholeWordMaskingModule(MaskingModule):
    """Whole word masking strategy."""

    @log_function()
    def __call__(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """Apply whole word masking to the input batch."""
        input_ids = batch['input_ids']
        word_ids = batch['word_ids']
        special_tokens_mask = batch['special_tokens_mask']

        if input_ids.dim() != 1:
            raise ValueError(f"Expected 1D input tensor, got: {input_ids.shape}")

        input_ids = self.tensor_manager.create_cpu_tensor(input_ids.clone(), dtype=torch.long)
        original_ids = input_ids.clone()

        # Convert to list for processing
        word_ids_list = [
            None if i == -1 or m == 1 else i 
            for i, m in zip(word_ids.tolist(), special_tokens_mask.tolist())
        ]
        
        # Get word boundaries
        word_boundaries = self._get_word_boundaries(input_ids, word_ids_list)
        maskable_boundaries = self._get_maskable_boundaries(word_boundaries, word_ids_list)

        if not maskable_boundaries:
            logger.warning("No maskable word boundaries found")
            return input_ids, torch.full_like(input_ids, -100)

        # Select words to mask
        num_to_mask = max(1, int(len(maskable_boundaries) * self.mask_prob))
        words_to_mask = random.sample(maskable_boundaries, num_to_mask)

        # Apply masking
        masked_positions: Set[int] = set()
        for start, end in words_to_mask:
            self._apply_token_masking(input_ids, start, end)
            masked_positions.update(range(start, end))

        labels = self._create_labels(original_ids, masked_positions)
        return input_ids, labels

class SpanMaskingModule(MaskingModule):
    """Span-based masking, prioritizing longer spans."""

    MIN_SPAN_LENGTH = 1
    MAX_SPAN_LENGTH = 10

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        config: MaskingConfig
    ) -> None:
        """Initialize span masking module."""
        super().__init__(tokenizer, config)
        
        if not self.MIN_SPAN_LENGTH <= config.max_span_length <= self.MAX_SPAN_LENGTH:
            raise ValueError(
                f"max_span_length must be between {self.MIN_SPAN_LENGTH} and "
                f"{self.MAX_SPAN_LENGTH}, got {config.max_span_length}"
            )
        self.max_span_length = config.max_span_length

    @log_function()
    def __call__(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """Apply span masking."""
        input_ids = batch['input_ids']
        word_ids = batch['word_ids']
        special_tokens_mask = batch['special_tokens_mask']

        if input_ids.dim() != 1:
            raise ValueError(f"Expected 1D input tensor, got shape: {input_ids.shape}")

        # Prepare tensors
        input_ids = self.tensor_manager.create_cpu_tensor(input_ids.clone(), dtype=torch.long)
        original_ids = input_ids.clone()

        # Process word IDs
        word_ids_list = [
            None if id == -1 or mask == 1 else id 
            for id, mask in zip(word_ids.tolist(), special_tokens_mask.tolist())
        ]

        # Get boundaries
        word_boundaries = self._get_word_boundaries(input_ids, word_ids_list)
        maskable_boundaries = self._get_maskable_boundaries(
            word_boundaries, word_ids_list, self.max_span_length
        )

        if not maskable_boundaries:
            return input_ids, torch.full_like(input_ids, -100)

        # Calculate masking targets
        seq_length = len(input_ids)
        target_masked = int(seq_length * self.mask_prob)

        # Process boundaries with overlap scores
        processed_boundaries = []
        for boundary in maskable_boundaries:
            start, end = boundary
            length = end - start
            overlap_score = 0
            
            # Calculate overlap score
            for j in range(start, end):
                for other_start, other_end in maskable_boundaries:
                    if (other_start <= j < other_end and 
                        (start, end) != (other_start, other_end)):
                        overlap_score += 1
                        break
                        
            processed_boundaries.append((start, end, length, overlap_score))

        # Sort by length and overlap score
        processed_boundaries.sort(key=lambda x: (x[2], -x[3]), reverse=True)

        # Apply masking
        masked_positions: Set[int] = set()
        masked_count = 0

        for start, end, _, _ in processed_boundaries:
            for i in range(start, end):
                if masked_count < target_masked and i not in masked_positions:
                    self._apply_token_masking(input_ids, i, i + 1)
                    masked_positions.add(i)
                    masked_count += 1

        # Create labels and log results
        labels = self._create_labels(original_ids, masked_positions)
        num_masked = len(masked_positions)
        mask_ratio = num_masked / seq_length
        
        logger.info(
            f"Masking results:\n"
            f"- Mask ratio achieved: {mask_ratio:.2%}\n"
            f"- Total tokens: {len(input_ids)}\n"
            f"- Masked tokens: {num_masked}\n"
            f"- Sequence length: {seq_length}"
        )
        
        return input_ids, labels

@log_function()
def create_attention_mask(input_ids: Tensor, padding_idx: int = 0) -> Tensor:
    """Create attention mask from input IDs."""
    return (input_ids != padding_idx).float()

__all__ = [
    'MaskingConfig',
    'MaskingModule',
    'WholeWordMaskingModule',
    'SpanMaskingModule',
    'create_attention_mask',
]