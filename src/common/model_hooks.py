"""Utility functions for adding hooks to models."""
import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def add_label_check_hook(model):
    """
    Add forward hook to model to check and add labels if missing.
    
    Args:
        model: The PyTorch model
        
    Returns:
        Model with hook added
    """
    from src.common.fix_batch_labels import ensure_batch_has_labels
    
    # Store the original forward method
    original_forward = model.forward
    
    # Define the new forward method with label checking
    def forward_with_label_check(*args, **kwargs):
        # Check if labels are in kwargs
        if 'labels' not in kwargs and len(args) <= 3:  # Typically position args would be input_ids, attn_mask, token_type_ids
            logger.warning("Model.forward: No labels in kwargs, adding them")
            
            # Find input_ids from args or kwargs
            input_ids = None
            if len(args) > 0:
                input_ids = args[0]  # Assume input_ids is first positional arg
            elif 'input_ids' in kwargs:
                input_ids = kwargs['input_ids']
                
            if input_ids is not None:
                # Create a batch-like dict to pass to ensure_batch_has_labels
                temp_batch = {'input_ids': input_ids}
                if 'attention_mask' in kwargs:
                    temp_batch['attention_mask'] = kwargs['attention_mask']
                
                # Ensure the batch has labels
                fixed_batch = ensure_batch_has_labels(temp_batch)
                
                # Add labels to kwargs
                if 'labels' in fixed_batch:
                    kwargs['labels'] = fixed_batch['labels']
                    logger.info("Added labels directly in forward_with_label_check")
        
        # Call the original forward method
        return original_forward(*args, **kwargs)
    
    # Replace the forward method
    model.forward = forward_with_label_check
    
    logger.info("Added label check hook to model")
    
    return model
