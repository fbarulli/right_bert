"""
Main training script for embedding learning with trial management.
"""
from src.embedding.imports import (
    torch,
    Path,
    Dict, Any, Optional,
    PreTrainedModel,
    DataLoader, Dataset,
    logger,
    log_function,
    LogConfig,
    create_progress_bar,
    WandbManager,
    optuna,
)
# Add direct import for nn
from torch import nn

# Update to import the missing EmbeddingTrainer class
import logging
from typing import Dict, Any, Optional

# Import the EmbeddingTrainer class
from src.embedding.trainers import EmbeddingTrainer

logger = logging.getLogger(__name__)

@log_function()
def train_embeddings(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    metrics_dir: str,
    is_trial: bool = False,
    trial: Optional[optuna.Trial] = None,
    wandb_manager: Optional[Any] = None,
    job_id: int = 0,
    train_dataset: Optional[Dataset] = None,
    val_dataset: Optional[Dataset] = None
) -> Dict[str, Any]:
    """Train embedding model."""
    logger.info("Entering train_embeddings")
    
    # Import the fix_dataloader_config function
    from src.common.fix_dataloader import fix_dataloader_config
    from src.common.fix_batch_labels import ensure_batch_has_labels
    from src.common.dataloader_wrapper import LabelEnsureDataLoader
    
    # Fix config to avoid multiprocessing issues
    config = fix_dataloader_config(config)
    
    # Ensure model is on the correct device
    device = next(model.parameters()).device
    logger.info(f"Model is initially on device: {device}")
    
    # Force model to GPU if available but not being used
    if str(device) == "cpu" and torch.cuda.is_available():
        logger.warning("CUDA is available but model is on CPU. Moving model to GPU...")
        
        # Use CUDA manager if possible
        try:
            from src.common.managers import get_cuda_manager
            cuda_manager = get_cuda_manager()
            device = cuda_manager.get_device()
        except Exception as e:
            logger.warning(f"Error using CUDA manager: {e}, falling back to default CUDA device")
            device = torch.device("cuda")
            
        # Move model to GPU
        model = model.to(device)
        new_device = next(model.parameters()).device
        logger.info(f"Model moved to device: {new_device}")
    
    # Log key model and dataset information for diagnosis
    logger.info(f"Training dataset size: {len(train_dataset) if train_dataset else 'unknown'}")
    logger.info(f"Validation dataset size: {len(val_dataset) if val_dataset else 'unknown'}")
    
    # Wrap data loaders to ensure all batches have labels
    logger.info("Wrapping data loaders to ensure batches have labels")
    train_loader = LabelEnsureDataLoader(train_loader)
    val_loader = LabelEnsureDataLoader(val_loader)
    
    # Add debug check on sample batch
    sample_batch = next(iter(train_loader))  # This should now have labels due to wrapper
    has_labels = 'labels' in sample_batch
    logger.info(f"Sample batch keys after wrapping: {list(sample_batch.keys())}")
    logger.info(f"Batch includes labels after wrapping: {has_labels}")
    
    if not has_labels:
        logger.critical("STILL NO LABELS AFTER WRAPPING! Applying emergency fix")
        # Add emergency check for labels in forward pass hooks
    
    # Get all required managers from the factory
    from src.common.managers import (
        get_cuda_manager,
        get_batch_manager,
        get_amp_manager,
        get_tokenizer_manager,
        get_metrics_manager,
        get_storage_manager
    )
    
    try:
        # Get instances of all required managers
        cuda_manager = get_cuda_manager()
        batch_manager = get_batch_manager()
        amp_manager = get_amp_manager()
        tokenizer_manager = get_tokenizer_manager()
        metrics_manager = get_metrics_manager()
        storage_manager = get_storage_manager()
        
        # Create trainer with all required dependencies
        trainer = EmbeddingTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            metrics_dir=metrics_dir,
            # Pass all managers (now safe since we disabled multiprocessing)
            batch_manager=batch_manager,
            amp_manager=amp_manager,
            metrics_manager=metrics_manager,
            cuda_manager=cuda_manager,
            tokenizer_manager=tokenizer_manager,
            storage_manager=storage_manager,
            is_trial=is_trial,
            trial=trial,
            wandb_manager=wandb_manager,
            job_id=job_id
        )
        
        # Train the model
        results = trainer.train()
        return results
        
    except Exception as e:
        logger.error(f"Fatal error in embedding training: {str(e)}")
        raise

def validate_embeddings(
    model_path: str,
    tokenizer_name: str,
    output_dir: str,
    device: Optional[torch.device] = None
) -> None:
    """
    Validates the quality of trained embeddings.
    
    Args:
        model_path: Path to the saved model
        tokenizer_name: Name/path of tokenizer
        output_dir: Directory for validation outputs
        device: Optional torch device
    """
    # Validation logic here
    pass

__all__ = [
    'train_embeddings',
    'validate_embeddings',
]