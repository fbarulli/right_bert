"""
Trainer implementation for embedding learning with progress tracking and metrics.
"""
from src.embedding.imports import (
    torch,
    dataclass,
    Dict, Any, Optional, TypeVar, cast, List,
    Dataset, DataLoader,
    PreTrainedModel,
    Optimizer,
    get_linear_schedule_with_warmup,
    logger,
    log_function,
    LogConfig,
    create_progress_bar,
    # Memory management
    GPUMemoryManager,
    TensorPool,
    MemoryTracker,
    CachingDict,
    # Managers
    CUDAManager,
    BatchManager,
    AMPManager,
    TokenizerManager,
    MetricsManager,
    StorageManager,
    WandbManager,
    # Base trainer
    BaseTrainer,
)

T = TypeVar('T', bound=PreTrainedModel)

@dataclass
class EmbeddingTrainerConfig:
    """Configuration for embedding trainer."""
    max_grad_norm: float
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_epochs: int
    scheduler: Dict[str, Any]
    log_level: str = 'log'
    gc_threshold: float = 0.8
    cache_size: int = 1000
    tensor_pool_size: int = 1000

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.log_level not in ['debug', 'log', 'none']:
            raise ValueError(f"Invalid log level: {self.log_level}")

class EmbeddingTrainer(BaseTrainer):
    """Trainer for learning embeddings through masked language modeling."""

    def __init__(
        self,
        cuda_manager: CUDAManager,
        batch_manager: BatchManager,
        amp_manager: AMPManager,
        tokenizer_manager: TokenizerManager,
        metrics_manager: MetricsManager,
        storage_manager: StorageManager,
        model: T,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        metrics_dir: Optional[str] = None,
        is_trial: bool = False,
        trial: Optional['optuna.Trial'] = None,
        wandb_manager: Optional[WandbManager] = None,
        job_id: Optional[int] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None
    ) -> None:
        """Initialize embedding trainer."""
        trainer_config = EmbeddingTrainerConfig(
            max_grad_norm=config['training']['max_grad_norm'],
            batch_size=config['training']['batch_size'],
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            learning_rate=config['training']['learning_rate'],
            num_epochs=config['training']['num_epochs'],
            scheduler=config['training']['scheduler'],
            log_level=config['training'].get('log_level', 'log'),
            gc_threshold=config['training'].get('gc_threshold', 0.8),
            cache_size=config['training'].get('cache_size', 1000),
            tensor_pool_size=config['training'].get('tensor_pool_size', 1000)
        )
        
        # Generate unique trial ID if this is a trial
        trial_id = f"trial_{trial.number}" if trial else None
        
        # Initialize GPU memory manager first
        self.gpu_manager = GPUMemoryManager(trial_id=trial_id)
        
        self.max_grad_norm = trainer_config.max_grad_norm
        self.log_config = LogConfig(level=trainer_config.log_level)
        self.gradient_accumulation_steps = trainer_config.gradient_accumulation_steps
        self.trial = trial
        
        # Initialize memory management with trial ID
        self.memory_tracker = MemoryTracker(
            gc_threshold=trainer_config.gc_threshold,
            log_config=self.log_config,
            trial_id=trial_id,
            gpu_manager=self.gpu_manager
        )
        self.tensor_pool = TensorPool(
            max_size=trainer_config.tensor_pool_size,
            trial_id=trial_id,
            gpu_manager=self.gpu_manager
        )
        self.cache = CachingDict(
            maxsize=trainer_config.cache_size,
            trial_id=trial_id,
            memory_tracker=self.memory_tracker,
            gpu_manager=self.gpu_manager
        )

        super().__init__(
            cuda_manager=cuda_manager,
            batch_manager=batch_manager,
            amp_manager=amp_manager,
            tokenizer_manager=tokenizer_manager,
            metrics_manager=metrics_manager,
            storage_manager=storage_manager,
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

        self.best_embedding_loss = float('inf')
        self.best_val_acc = 0.0

        # Scale learning rate based on batch size
        base_batch_size = 32
        current_batch_size = trainer_config.batch_size
        effective_batch_size = current_batch_size * trainer_config.gradient_accumulation_steps

        if effective_batch_size != base_batch_size:
            scale_factor = effective_batch_size / base_batch_size
            trainer_config.learning_rate *= scale_factor
            logger.info(
                f"Scaled learning rate by {scale_factor:.3f}:\n"
                f"- Batch size: {current_batch_size}\n"
                f"- Gradient accumulation: {trainer_config.gradient_accumulation_steps}\n"
                f"- Effective batch size: {effective_batch_size}"
            )

        self._optimizer = self.create_optimizer()

        if trainer_config.scheduler['use_scheduler']:
            if not train_loader:
                raise ValueError("train_loader is required when using scheduler")
                
            num_training_steps = len(train_loader) * trainer_config.num_epochs
            num_warmup_steps = int(num_training_steps * trainer_config.scheduler['warmup_ratio'])

            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self._optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            logger.info(
                f"Created linear scheduler with warmup:\n"
                f"- Warmup steps: {num_warmup_steps}\n"
                f"- Total steps: {num_training_steps}"
            )

    def train(self, num_epochs: int) -> None:
        """Train the model for the specified number of epochs."""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        try:
            progress_bar = create_progress_bar(
                range(num_epochs),
                desc="Training Progress",
                enabled=self.log_config.tqdm_enabled
            )

            for epoch in progress_bar:
                # Clear only tensor cache at epoch start
                self.gpu_manager.clear_tensor_cache()
                
                # Train epoch
                self.train_epoch(epoch)
                
                # Validate
                val_metrics = self.validate()
                
                # Log metrics
                if isinstance(progress_bar, type(range(0))):
                    logger.info(
                        f"Epoch {epoch + 1}/{num_epochs}:\n"
                        f"Validation metrics: {val_metrics}"
                    )
                else:
                    progress_bar.set_postfix(**val_metrics)

        except optuna.exceptions.TrialPruned as e:
            logger.info(f"Trial pruned: {e}")
            self.cleanup_trial_resources()
            raise
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            self.cleanup_trial_resources()
            raise
            
        finally:
            # Clean up trial resources if this is a trial
            if self.trial:
                self.cleanup_trial_resources()

    def cleanup_trial_resources(self) -> None:
        """Clean up resources specific to this trial."""
        if self.trial:
            logger.info(f"Cleaning up resources for trial {self.trial.number}")
            self.tensor_pool.clear()
            self.cache.clear()
            self.memory_tracker.reset_peaks()
            self.gpu_manager.clear_trial_memory()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if hasattr(self, 'trial') and self.trial:
            self.cleanup_trial_resources()

    # [Rest of the class implementation remains the same]

__all__ = [
    'EmbeddingTrainer',
    'EmbeddingTrainerConfig',
]