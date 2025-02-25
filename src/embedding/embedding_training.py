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

@log_function()
def train_embeddings(
    model: PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    metrics_dir: Optional[str] = None,
    is_trial: bool = False,
    trial: Optional['optuna.Trial'] = None,
    wandb_manager: Optional[WandbManager] = None,
    job_id: Optional[int] = None,
    train_dataset: Optional[Dataset] = None,
    val_dataset: Optional[Dataset] = None
) -> None:
    """
    Train embedding model with masked language modeling.
    
    Args:
        model: The pre-trained model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        metrics_dir: Optional directory for saving metrics
        is_trial: Whether this is an Optuna trial
        trial: Optional Optuna trial object
        wandb_manager: Optional WandB manager for logging
        job_id: Optional job ID for distributed training
        train_dataset: Training dataset (for cleanup)
        val_dataset: Validation dataset (for cleanup)
    """
    try:
        from src.embedding.embedding_trainer import EmbeddingTrainer

        # Create trainer instance with trial info
        trainer = EmbeddingTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            metrics_dir=metrics_dir,
            is_trial=is_trial,
            trial=trial,
            wandb_manager=wandb_manager,
            job_id=job_id,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )

        num_epochs = config['training']['num_epochs']
        
        try:
            trainer.train(num_epochs)
            
            # Handle trial completion metrics
            if trial and metrics_dir:
                metrics_path = Path(metrics_dir)
                metrics_path.mkdir(parents=True, exist_ok=True)
                
                try:
                    from src.common.study.trial_analyzer import TrialAnalyzer
                    analyzer = TrialAnalyzer(metrics_path)
                    analyzer.plot_trial_curves([trial], "Embedding Training")
                except Exception as e:
                    logger.warning(f"Failed to plot trial metrics: {str(e)}")
                    
        except optuna.exceptions.TrialPruned:
            # Trial was pruned - cleanup will be handled by trainer
            raise
            
        except Exception as e:
            logger.error(f"Error in embedding training: {str(e)}")
            raise
            
        finally:
            # Let the trainer handle its own cleanup
            # It will know whether this is a trial and clean up accordingly
            pass

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