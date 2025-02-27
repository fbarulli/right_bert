"""Wrapper class for DataLoader to ensure consistent batch formatting."""
import torch
import logging
from typing import Iterator, Dict, Any

logger = logging.getLogger(__name__)

class LabelEnsureDataLoader:
    """Wrapper around DataLoader to ensure all batches have labels."""

    def __init__(self, data_loader):
        """
        Initialize the wrapper.
        
        Args:
            data_loader: The DataLoader to wrap
        """
        self.data_loader = data_loader
        from src.common.fix_batch_labels import ensure_batch_has_labels
        self.ensure_batch_has_labels = ensure_batch_has_labels
        
        # Count batches with missing labels
        self.missing_labels_count = 0
        
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through the DataLoader, ensuring each batch has labels."""
        for batch in self.data_loader:
            # Check if batch has labels before applying fix
            had_labels = 'labels' in batch
            
            # Apply fix regardless
            fixed_batch = self.ensure_batch_has_labels(batch)
            
            # Validate fix worked
            if not had_labels and 'labels' in fixed_batch:
                self.missing_labels_count += 1
                
            # Count fixes in batches
            if self.missing_labels_count > 0 and self.missing_labels_count % 10 == 0:
                logger.warning(f"Added labels to {self.missing_labels_count} batches so far")
                
            yield fixed_batch
            
    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.data_loader)
        
    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return self.data_loader.batch_size
        
    @property
    def dataset(self):
        """Get the dataset."""
        return self.data_loader.dataset
