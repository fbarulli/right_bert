
# src/embedding/embedding_training.py
#src/embedding/embedding_training.py
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# Corrected: Use absolute import
from src.common.managers import get_wandb_manager

logger = logging.getLogger(__name__)

def train_embeddings(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    metrics_dir: Optional[str] = None,
    is_trial: bool = False,
    trial: Optional['optuna.Trial'] = None,
    wandb_manager: Optional['WandbManager'] = None,  # Corrected type
    job_id: Optional[int] = None,
    train_dataset: Optional[Dataset] = None,
    val_dataset: Optional[Dataset] = None
) -> None:
    """Train embedding model with masked language modeling."""
    try:
        from src.embedding.embedding_trainer import EmbeddingTrainer  # Corrected import

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
        trainer.train(num_epochs)

        if trial:
            try:
                from src.common.study.trial_analyzer import TrialAnalyzer # Corrected import
                metrics_path = Path(metrics_dir) if metrics_dir else Path.cwd() / 'metrics'
                metrics_path.mkdir(parents=True, exist_ok=True)
                analyzer = TrialAnalyzer(metrics_path)
                analyzer.plot_trial_curves([trial], "Embedding Training")
            except Exception as e:
                logger.warning(f"Failed to plot trial metrics: {str(e)}")

        trainer.cleanup_memory(aggressive=True)

    except Exception as e:
        logger.error(f"Error in embedding training: {str(e)}")
        raise

def validate_embeddings(
    model_path: str,
    tokenizer_name: str,
    output_dir: str,
    words_to_check: Optional[List[str]] = None,  # Provide a default
    top_k: int = 10
):
    if words_to_check is None:
        words_to_check = []
    """
    Validates the quality of trained embeddings using nearest neighbors and t-SNE.

    Args:
        model_path: Path to the directory containing the saved model.
        tokenizer_name: Name/path of the tokenizer.
        output_dir: Where to save visualization plots.
        words_to_check: Optional list of words to find nearest neighbors for.
        top_k: Number of nearest neighbors to retrieve.
    """
    pass