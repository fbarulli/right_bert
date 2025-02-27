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
    
    # Fix config to avoid multiprocessing issues
    config = fix_dataloader_config(config)
    
    # Log key model and dataset information for diagnosis
    device = next(model.parameters()).device
    logger.info(f"Model is on device: {device}")
    logger.info(f"Training dataset size: {len(train_dataset) if train_dataset else 'unknown'}")
    logger.info(f"Validation dataset size: {len(val_dataset) if val_dataset else 'unknown'}")
    
    # Check if batch preparation properly includes labels
    sample_batch = next(iter(train_loader))
    has_labels = 'labels' in sample_batch
    logger.info(f"Sample batch keys: {list(sample_batch.keys())}")
    logger.info(f"Batch includes labels: {has_labels}")
    
    # Monkey patch the DataLoader's __iter__ method to ensure labels
    original_iter = train_loader.__iter__
    
    def patched_iter():
        for batch in original_iter():
            yield ensure_batch_has_labels(batch)
            
    train_loader.__iter__ = patched_iter
    
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