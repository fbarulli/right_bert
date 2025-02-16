# test_csv_dataset.py

import unittest
import tempfile
import os
import csv
import numpy as np
import torch
from transformers import BertTokenizerFast
from common.csv_dataset import CSVDataset, csv_dataset_factory, process_batch_parallel, TokenizedBatch  # Replace 'your_module'
import logging

# Set up logging for the tests
logging.basicConfig(level=logging.DEBUG)

class TestCSVDataset(unittest.TestCase):
    """Unit tests for the CSVDataset class and related functions."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment before running any tests."""
        # Create a temporary directory for test data and cache
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.data_dir = os.path.join(cls.temp_dir.name, "data")
        cls.cache_dir = os.path.join(cls.temp_dir.name, "cache")
        os.makedirs(cls.data_dir, exist_ok=True)
        os.makedirs(cls.cache_dir, exist_ok=True)

        # Create a dummy CSV file
        cls.csv_file = os.path.join(cls.data_dir, "test.csv")
        with open(cls.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'rating'])
            writer.writerow(['This is a great movie.', '5'])
            writer.writerow(['I did not like this movie.', '1'])
            writer.writerow(['Average film.', '3'])
            writer.writerow(['It was ok', '4'])
            writer.writerow(['amazing!', '5'])

        # Create a dummy tokenizer
        cls.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        cls.max_length = 16

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests have run."""
        # Clean up the temporary directory
        cls.temp_dir.cleanup()


    def test_create_memmap_arrays(self):
        """Test the creation of memory-mapped arrays."""
        dataset = CSVDataset(self.csv_file, self.tokenizer, self.max_length, cache_dir=self.cache_dir)
        self.assertEqual(len(dataset), 4)  # train ratio default 0.9 -> 4 training examples
        self.assertEqual(dataset.input_ids.shape, (5, self.max_length))
        self.assertEqual(dataset.attention_mask.shape, (5, self.max_length))
        self.assertEqual(dataset.labels.shape, (5,))
        self.assertEqual(dataset.special_tokens_mask.shape, (5, self.max_length))
        self.assertEqual(dataset.word_ids.shape, (5, self.max_length))
        self.assertTrue(np.all(dataset.labels[:4] >= 0))
        self.assertEqual(dataset.labels[4], -1) #val example


    def test_load_memmap_arrays(self):
        """Test loading existing memory-mapped arrays."""
        # First, create the dataset to generate the memmap files
        _ = CSVDataset(self.csv_file, self.tokenizer, self.max_length, cache_dir=self.cache_dir)
        # Then, create a new dataset, which should load the existing files
        dataset = CSVDataset(self.csv_file, self.tokenizer, self.max_length, cache_dir=self.cache_dir)
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.input_ids.shape, (5, self.max_length))
        self.assertEqual(dataset.attention_mask.shape, (5, self.max_length))
        self.assertEqual(dataset.labels.shape, (5,))
        self.assertTrue(np.all(dataset.labels[:4] >= 0))

    def test_dataset_splits(self):
        """Test the train/validation split functionality."""
        dataset_train = CSVDataset(self.csv_file, self.tokenizer, self.max_length, split='train', train_ratio=0.8, cache_dir=self.cache_dir)
        dataset_val = CSVDataset(self.csv_file, self.tokenizer, self.max_length, split='val', train_ratio=0.8, cache_dir=self.cache_dir)
        self.assertEqual(len(dataset_train), 4)
        self.assertEqual(len(dataset_val), 1)

    def test_getitem(self):
        """Test retrieving a single item from the dataset."""
        dataset = CSVDataset(self.csv_file, self.tokenizer, self.max_length, cache_dir=self.cache_dir)
        item = dataset[0]
        self.assertIsInstance(item, dict)
        self.assertIsInstance(item['input_ids'], torch.Tensor)
        self.assertIsInstance(item['attention_mask'], torch.Tensor)
        self.assertIsInstance(item['special_tokens_mask'], torch.Tensor)
        self.assertIsInstance(item['word_ids'], torch.Tensor)
        self.assertIsInstance(item['index'], torch.Tensor)
        self.assertIsInstance(item['labels'], torch.Tensor)
        self.assertEqual(item['input_ids'].shape, (self.max_length,))
        self.assertEqual(item['attention_mask'].shape, (self.max_length,))
        self.assertEqual(item['index'], 0)

    def test_getitem_no_label(self):
        """Test retrieving an item when no labels are present in the CSV."""
        # Create CSV data *without* labels
        csv_file_no_labels = os.path.join(self.data_dir, 'test_no_labels.csv')
        with open(csv_file_no_labels, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['text'])  # Header without 'rating'
            writer.writerow(['This is a test.'])

        dataset = CSVDataset(csv_file_no_labels, self.tokenizer, self.max_length, cache_dir=self.cache_dir, train_ratio=1.0) #train ratio = 1.0 so that it is not empty
        item = dataset[0]
        self.assertNotIn('labels', item)  # Check that 'labels' is not present

    def test_factory(self):
        """Test the csv_dataset_factory function."""
        dataset = csv_dataset_factory(self.csv_file, self.tokenizer, self.max_length, cache_dir=self.cache_dir)
        self.assertIsInstance(dataset, CSVDataset)

    def test_process_batch_parallel(self):
        """Test the process_batch_parallel function."""
        texts = ["This is a test.", "Another test sentence."]
        encoding = process_batch_parallel(self.tokenizer, texts, self.max_length)
        self.assertIsInstance(encoding, dict)
        self.assertIsInstance(encoding['input_ids'], np.ndarray)
        self.assertIsInstance(encoding['attention_mask'], np.ndarray)
        self.assertIsInstance(encoding['special_tokens_mask'], np.ndarray)
        self.assertIsInstance(encoding['word_ids'], np.ndarray)
        self.assertEqual(encoding['input_ids'].shape, (2, self.max_length))
        self.assertEqual(encoding['attention_mask'].shape, (2, self.max_length))

    def test_invalid_rating(self):
        """Test handling of invalid ratings in the CSV data."""
        # Test handling of invalid ratings
        invalid_csv_file = os.path.join(self.data_dir, 'invalid_test.csv')
        with open(invalid_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'rating'])
            writer.writerow(['This is a great movie.', '5'])
            writer.writerow(['Invalid rating', '6'])  # Invalid: > 5
            writer.writerow(['Another test', 'abc']) # Invalid: not an int
            writer.writerow(['OK movie', '3'])

        with self.assertLogs(level='ERROR') as cm:
          dataset = CSVDataset(invalid_csv_file, self.tokenizer, self.max_length, train_ratio = 1.0, cache_dir=self.cache_dir)
          self.assertEqual(len(dataset), 2)  # Only valid lines
        self.assertTrue(any("Rating must be between 1-5" in log for log in cm.output))
        self.assertTrue(any("Rating must be an integer" in log for log in cm.output))
    
    def test_tokenized_batch_dataclass(self):
        """Test the TokenizedBatch dataclass."""
        input_ids = np.array([1, 2, 3])
        attention_mask = np.array([1, 1, 0])
        special_tokens_mask = np.array([0, 1, 0])
        word_ids = np.array([0, 1, -1])
        labels = np.array([1])
        index = np.array([0])

        batch = TokenizedBatch(input_ids=input_ids, attention_mask=attention_mask,
                            special_tokens_mask=special_tokens_mask, word_ids=word_ids,
                            labels=labels, index=index)
        self.assertTrue(hasattr(batch, 'input_ids'))
        self.assertTrue(hasattr(batch, 'attention_mask'))
        self.assertTrue(hasattr(batch, 'special_tokens_mask'))
        self.assertTrue(hasattr(batch, 'word_ids'))
        self.assertTrue(hasattr(batch, 'labels'))
        self.assertTrue(hasattr(batch, 'index'))
        self.assertIsInstance(batch, TokenizedBatch)

if __name__ == '__main__':
    unittest.main()