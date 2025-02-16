# tests/embedding/test_masking.py

import unittest
import torch
from transformers import BertTokenizerFast
from src.embedding.masking import WholeWordMaskingModule, SpanMaskingModule, create_attention_mask, MaskingModule  # Absolute import
from src.common.managers import TensorManager # Absolute import
from typing import Dict
import logging


# Set up logging (for test output)
logging.basicConfig(level=logging.DEBUG)


# Mock TensorManager for testing purposes
class MockTensorManager(TensorManager):

  def create_cpu_tensor(self, data, dtype):
    return torch.tensor(data, dtype=dtype)

class TestMasking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        cls.tensor_manager = MockTensorManager()
    def _create_test_batch(self, text: str) -> Dict[str, torch.Tensor]:
        """Helper function to create a test batch."""
        encoding = self.tokenizer(
            text,
            return_tensors='np',
            return_special_tokens_mask=True,
            return_word_ids=True,
            truncation=True,
            padding='max_length',
            max_length = 16
        )
        batch = {
            'input_ids': self.tensor_manager.create_cpu_tensor(encoding['input_ids'][0], dtype=torch.long),
            'attention_mask': self.tensor_manager.create_cpu_tensor(encoding['attention_mask'][0], dtype=torch.long),
            'special_tokens_mask': self.tensor_manager.create_cpu_tensor(encoding['special_tokens_mask'][0], dtype=torch.long),
            'word_ids': self.tensor_manager.create_cpu_tensor(encoding.word_ids(0), dtype=torch.long),
            'index': torch.tensor(0, dtype=torch.long)  # Include index
        }

        return batch
    def test_whole_word_masking(self):
        batch = self._create_test_batch("This is a test sentence for whole word masking.")
        masking = WholeWordMaskingModule(self.tokenizer, mask_prob=0.5)  # High mask prob for testing
        masked_input, labels = masking(batch)

        self.assertEqual(masked_input.shape, (16,))
        self.assertEqual(labels.shape, (16,))
        self.assertTrue((labels != -100).any())  # Check if *something* was masked

    def test_span_masking(self):
        batch = self._create_test_batch("This is another test sentence for span masking.")
        masking = SpanMaskingModule(self.tokenizer, mask_prob=0.5, max_span_length=3)
        masked_input, labels = masking(batch)

        self.assertEqual(masked_input.shape, (16,))
        self.assertEqual(labels.shape, (16,))
        self.assertTrue((labels != -100).any())

    def test_no_maskable_words(self):
        # Test case where no words can be masked (all special tokens)
        batch = self._create_test_batch("[CLS] [SEP]")
        masking = WholeWordMaskingModule(self.tokenizer, mask_prob=0.5)
        masked_input, labels = masking(batch)
        self.assertTrue((labels == -100).all())  # Nothing should be masked

        masking = SpanMaskingModule(self.tokenizer, mask_prob=0.5, max_span_length=3)
        masked_input, labels = masking(batch)
        self.assertTrue((labels == -100).all())


    def test_create_attention_mask(self):
        input_ids = torch.tensor([1, 2, 0, 3, 0, 0])  # 0 is the padding token
        attention_mask = create_attention_mask(input_ids)
        self.assertTrue(torch.equal(attention_mask, torch.tensor([1, 1, 0, 1, 0, 0], dtype=torch.float)))

    def test_masking_module_initialization(self):
      """Test the base MaskingModule's initialization and utility methods."""

      tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
      masking_module = MaskingModule(tokenizer=tokenizer, mask_prob=0.2)

      # Check mask probability clamping
      self.assertEqual(masking_module.mask_prob, 0.2)

      # Check with out-of-range mask probabilities
      masking_module_low = MaskingModule(tokenizer=tokenizer, mask_prob=0.05)
      self.assertEqual(masking_module_low.mask_prob, masking_module_low.MIN_MASK_PROB)

      masking_module_high = MaskingModule(tokenizer=tokenizer, mask_prob=0.35)
      self.assertEqual(masking_module_high.mask_prob, masking_module_high.MAX_MASK_PROB)

      # Check special token IDs
      self.assertIsInstance(masking_module.special_token_ids, set)
      self.assertTrue(tokenizer.cls_token_id in masking_module.special_token_ids)

      # Check valid vocab IDs
      self.assertIsInstance(masking_module.valid_vocab_ids, list)
      self.assertTrue(all(token_id not in masking_module.special_token_ids for token_id in masking_module.valid_vocab_ids))
    
    def test_get_word_boundaries(self):
      tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
      masking_module = MaskingModule(tokenizer=tokenizer)

      # Test case 1: Simple sentence
      batch = self._create_test_batch("This is a simple test.")
      word_boundaries = masking_module._get_word_boundaries(batch['input_ids'], batch['word_ids'].tolist())
      self.assertEqual(len(word_boundaries), 5) #Five words

      # Test case 2: Sentence with subwords
      batch = self._create_test_batch("This sentence has subword tokens.")
      word_boundaries = masking_module._get_word_boundaries(batch['input_ids'], batch['word_ids'].tolist())
      self.assertEqual(len(word_boundaries), 6) # six words

      # Test case 3: Sentence with special tokens
      batch = self._create_test_batch("[CLS] This is a test. [SEP]")
      word_ids_list = [None if i == -1 or m == 1 else i for i, m in zip(batch['word_ids'].tolist(), batch['special_tokens_mask'].tolist())]
      word_boundaries = masking_module._get_word_boundaries(batch['input_ids'], word_ids_list)
      self.assertEqual(len(word_boundaries), 4) # four words, excluding special tokens

    def test_get_maskable_boundaries(self):
      tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
      masking_module = MaskingModule(tokenizer=tokenizer)

      # Test case 1: Simple sentence
      batch = self._create_test_batch("This is a simple test.")
      word_ids_list = [None if i == -1 or m == 1 else i for i, m in zip(batch['word_ids'].tolist(), batch['special_tokens_mask'].tolist())]
      word_boundaries = masking_module._get_word_boundaries(batch['input_ids'], word_ids_list)

      maskable_boundaries = masking_module._get_maskable_boundaries(word_boundaries,word_ids_list)
      self.assertEqual(len(maskable_boundaries), len(word_boundaries)) # All are maskable

      # Test case 2: Sentence with special tokens
      batch = self._create_test_batch("[CLS] This is a test. [SEP]")
      word_ids_list = [None if i == -1 or m == 1 else i for i, m in zip(batch['word_ids'].tolist(), batch['special_tokens_mask'].tolist())]
      word_boundaries = masking_module._get_word_boundaries(batch['input_ids'], word_ids_list)
      maskable_boundaries = masking_module._get_maskable_boundaries(word_boundaries, word_ids_list)
      self.assertEqual(len(maskable_boundaries), 4) # Excluding the special tokens

    def test_apply_token_masking(self):
      tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
      masking_module = MaskingModule(tokenizer=tokenizer)

      # Test case: Mask a single token
      input_ids = torch.tensor([101, 1000, 102])  # [CLS] some_token [SEP]
      masking_module._apply_token_masking(input_ids, 1, 2)
      self.assertTrue(input_ids[1] == tokenizer.mask_token_id or input_ids[1] in masking_module.valid_vocab_ids)

    def test_create_labels(self):
      tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
      masking_module = MaskingModule(tokenizer=tokenizer)

      # Test case 1: No masking
      original_ids = torch.tensor([1, 2, 3, 4])
      masked_positions: Set[int] = set()
      labels = masking_module._create_labels(original_ids, masked_positions)
      self.assertTrue(torch.equal(labels, torch.tensor([-100, -100, -100, -100])))

      # Test case 2: Masking some positions
      original_ids = torch.tensor([1, 2, 3, 4])
      masked_positions = {1, 3}
      labels = masking_module._create_labels(original_ids, masked_positions)
      