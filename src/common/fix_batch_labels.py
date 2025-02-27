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
    try:
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
            
            # Double check that we have at least one masked token
            if masked_indices.sum() == 0:
                # Force-mask at least one non-special token
                non_special_indices = ~special_tokens_mask
                if non_special_indices.sum() > 0:
                    # Find indices of non-special tokens
                    non_special_positions = torch.nonzero(non_special_indices, as_tuple=True)[0]
                    if len(non_special_positions) > 0:
                        # Randomly mask one non-special token
                        idx_to_mask = non_special_positions[torch.randint(0, len(non_special_positions), (1,))]
                        masked_indices[idx_to_mask] = True
                        logger.warning(f"No tokens were randomly masked, forcing mask on position {idx_to_mask}")
            
            # Set labels to -100 for unmasked tokens
            labels[~masked_indices] = -100
            
            batch['labels'] = labels
            logger.debug(f"Created labels with {masked_indices.sum().item()} tokens for loss computation")
            
            # Sanity check - ensure labels exist now
            if 'labels' not in batch:
                logger.error("Failed to add labels to batch despite trying")
                # Emergency backup - create simple identity labels
                batch['labels'] = input_ids.clone()
        
        # Return batch whether modified or not
        return batch
    except Exception as e:
        logger.error(f"Error in ensure_batch_has_labels: {e}")
        # Don't propagate the exception - return the original batch
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
