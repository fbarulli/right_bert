"""Utilities for fixing batch data issues in embedding training."""
import torch
import logging

logger = logging.getLogger(__name__)

def ensure_batch_has_labels(batch):
    """
    Ensure the batch has proper labels for loss computation.
    
    Args:
        batch: The input batch dictionary
        
    Returns:
        Updated batch with labels if needed
    """
    if 'labels' not in batch and 'input_ids' in batch:
        # Generate labels from input_ids with MLM strategy
        logger.warning("Batch missing labels, generating MLM labels from input_ids")
        input_ids = batch['input_ids']
        
        # Create labels: -100 means "ignore this token in loss computation"
        labels = input_ids.clone()
        
        # For MLM, we only want to compute loss on masked tokens
        # We'll use a conservative approach and only mask 15% of tokens
        prob_matrix = torch.full(labels.shape, 0.15, device=labels.device)
        
        # Don't mask special tokens
        special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
        
        # Usual special tokens positions
        if labels.dim() > 1:
            special_tokens_mask[:, 0] = True  # Usually the CLS token
            if labels.size(1) > 1:
                special_tokens_mask[:, -1] = True  # Usually the SEP token
        else:
            special_tokens_mask[0] = True
            if labels.size(0) > 1:
                special_tokens_mask[-1] = True
                
        prob_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Sample tokens to mask
        masked_indices = torch.bernoulli(prob_matrix).bool()
        
        # Set labels to -100 for unmasked tokens
        labels[~masked_indices] = -100
        
        batch['labels'] = labels
        logger.debug(f"Created labels with {masked_indices.sum().item()} tokens for loss computation")
        
    return batch

def ensure_batch_device_consistency(batch, device):
    """
    Ensure all tensors in batch are on the same device.
    
    Args:
        batch: The input batch dictionary
        device: Target device
        
    Returns:
        Batch with all tensors on the same device
    """
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.device != device:
            logger.debug(f"Moving '{key}' from {value.device} to {device}")
            result[key] = value.to(device)
        else:
            result[key] = value
            
    return result
