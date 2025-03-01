"""EmbeddingTrainer class for handling embedding model training."""
import os
import logging
import torch
import torch.nn as nn
import time
import threading
import traceback  # Add explicit import for traceback
import inspect  # Add explicit import for inspect
from typing import Dict, Any, Optional, List, Tuple, Set  # Add Set to typing imports
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

from src.common.managers.amp_manager import AMPManager
from src.common.managers.wandb_manager import WandbManager
from src.common.managers.metrics_manager import MetricsLogger
from src.common.cuda_utils import ensure_model_on_device  # Import utility function

logger = logging.getLogger(__name__)

class EmbeddingTrainer:
    """Trainer for embedding models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        metrics_dir: str,
        is_trial: bool = False,
        trial = None,
        wandb_manager: Optional[WandbManager] = None,
        batch_manager = None,
        metrics_manager = None,
        amp_manager: Optional[AMPManager] = None,
        job_id: Optional[int] = None,
        # Add missing parameters that are being passed to the constructor
        cuda_manager = None,
        tokenizer_manager = None,
        storage_manager = None,
        train_dataset = None,
        val_dataset = None,
        # Accept additional kwargs for future compatibility
        **kwargs
    ):
        """
        Initialize the EmbeddingTrainer.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            metrics_dir: Directory to save metrics
            is_trial: Whether this is a trial run
            trial: Optuna trial object
            wandb_manager: WandbManager for logging
            batch_manager: BatchManager for batch handling
            metrics_manager: MetricsManager for metrics computation
            amp_manager: AMPManager for mixed precision
            job_id: Optional job ID
            cuda_manager: CUDA manager instance
            tokenizer_manager: Tokenizer manager instance
            storage_manager: Storage manager instance
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.is_trial = is_trial
        self.trial = trial
        self.wandb_manager = wandb_manager
        self.batch_manager = batch_manager
        self.metrics_manager = metrics_manager
        self.amp_manager = amp_manager
        self.job_id = job_id
        
        # Set up device
        self.device = self.model.device if hasattr(self.model, 'device') else next(model.parameters()).device
        logger.info(f"Training on device: {self.device}")
        
        # Set up metrics logger
        self.metrics_logger = MetricsLogger(
            metrics_dir=Path(metrics_dir),
            is_trial=is_trial,
            trial=trial,
            wandb_manager=wandb_manager,
            job_id=job_id
        )
        
        # Set up optimizer
        self.setup_optimizer()
        
        # Set up scheduler
        self.setup_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.total_train_time = 0
        
        logger.info(f"EmbeddingTrainer initialized with config: {json.dumps(self._get_printable_config(), indent=2)}")
    
    def _get_printable_config(self) -> Dict[str, Any]:
        """Get a simplified config for printing."""
        return {
            'training': {
                'learning_rate': self.config['training'].get('learning_rate', 'N/A'),
                'batch_size': self.config['training'].get('batch_size', 'N/A'),
                'num_epochs': self.config['training'].get('num_epochs', 'N/A'),
                'scheduler': self.config['training'].get('scheduler', {}).get('type', 'N/A')
            }
        }
    
    def setup_optimizer(self) -> None:
        """Set up optimizer based on config."""
        learning_rate = self.config['training'].get('learning_rate', 5e-5)
        weight_decay = self.config['training'].get('weight_decay', 0.01)
        
        # Get optimizer settings
        optimizer_cfg = self.config['training'].get('optimizer', {})
        optimizer_type = optimizer_cfg.get('type', 'adamw')
        
        # Create optimizer
        if optimizer_type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            logger.warning(f"Unknown optimizer type: {optimizer_type}. Using AdamW.")
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
            
        logger.info(f"Optimizer: {self.optimizer.__class__.__name__}, LR: {learning_rate}, Weight Decay: {weight_decay}")
    
    def setup_scheduler(self) -> None:
        """Set up learning rate scheduler based on config."""
        scheduler_cfg = self.config['training'].get('scheduler', {})
        scheduler_type = scheduler_cfg.get('type', 'linear')
        warmup_ratio = self.config['training'].get('warmup_ratio', 0.1)
        
        # Calculate total steps and warmup steps
        epochs = self.config['training'].get('num_epochs', 10)
        steps_per_epoch = len(self.train_loader)
        total_steps = epochs * steps_per_epoch
        warmup_steps = int(total_steps * warmup_ratio)
        
        # Create scheduler
        if scheduler_type.lower() == 'linear':
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif scheduler_type.lower() == 'cosine':
            from transformers import get_cosine_schedule_with_warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}. Using linear.")
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            
        logger.info(f"Scheduler: {scheduler_type}, Warmup Steps: {warmup_steps}, Total Steps: {total_steps}")
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns:
            Dict[str, Any]: Training results
        """
        max_epochs = self.config['training'].get('num_epochs', 10)
        early_stopping_patience = self.config['training'].get('early_stopping', {}).get('patience', 3)
        grad_clip = self.config['training'].get('grad_clip', None)
        
        logger.info(f"Starting training for {max_epochs} epochs")
        
        try:
            start_time = time.time()
            
            # Verify model is on the right device first
            self.model = ensure_model_on_device(self.model, self.device)
            logger.info(f"Verified model is on device: {self.device}")
            
            for epoch in range(max_epochs):
                self.current_epoch = epoch
                
                # Training
                train_results = self.train_epoch()
                
                # Validation
                val_results = self.validate()
                
                # Log epoch results
                self.log_epoch_results(epoch, train_results, val_results)
                
                # Save model if best
                if val_results['loss'] < self.best_val_loss:
                    self.best_val_loss = val_results['loss']
                    self.save_model('best')
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                
                # Check early stopping
                if early_stopping_patience > 0 and self.early_stop_counter >= early_stopping_patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break
                
                # Save checkpoint
                if (epoch + 1) % self.config['training'].get('checkpoint_interval', max_epochs) == 0:
                    self.save_model(f"checkpoint-{epoch+1}")
                    
            self.total_train_time = time.time() - start_time
            logger.info(f"Training completed in {self.total_train_time:.2f} seconds")
            
            # Final save
            self.save_model('final')
            
            # Log trial result if in trial mode
            if self.is_trial and self.trial:
                self.trial.set_user_attr("best_val_loss", self.best_val_loss)
                
            return {
                'best_val_loss': self.best_val_loss,
                'total_train_time': self.total_train_time,
                'epochs': self.current_epoch + 1,
                'global_step': self.global_step
            }
                
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise
    
    def _safe_wandb_log(self, metrics: Dict[str, float], step: int) -> None:
        """
        Safely log metrics to WandB with comprehensive error handling.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
        """
        if not self.wandb_manager:
            return
            
        try:
            # Multiple ways to check if WandB is enabled
            wandb_enabled = False
            
            # Method 1: Check is_enabled() method
            if hasattr(self.wandb_manager, 'is_enabled') and callable(self.wandb_manager.is_enabled):
                wandb_enabled = self.wandb_manager.is_enabled()
            # Method 2: Check enabled property
            elif hasattr(self.wandb_manager, 'enabled'):
                wandb_enabled = self.wandb_manager.enabled
            # Method 3: Check _enabled attribute
            elif hasattr(self.wandb_manager, '_enabled'):
                wandb_enabled = self.wandb_manager._enabled
                
            if wandb_enabled:
                # Also check for log_metrics method
                if hasattr(self.wandb_manager, 'log_metrics') and callable(self.wandb_manager.log_metrics):
                    self.wandb_manager.log_metrics(metrics, step=step)
                else:
                    logger.warning("WandB manager has no log_metrics method")
        except Exception as e:
            logger.warning(f"Error logging to WandB: {e}")
            # Continue execution - logging failure should not interrupt training
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dict[str, float]: Training metrics
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch+1} [Train]",
            leave=False
        )
        
        for batch in progress_bar:
            # Move batch to device safely
            try:
                # First check if batch_manager is properly initialized
                if hasattr(self, 'batch_manager') and self.batch_manager is not None:
                    try:
                        # Try to use the batch manager, but fall back if it fails
                        batch = self.batch_manager.prepare_batch(batch)
                    except Exception as e:
                        logger.warning(f"BatchManager error: {e}, falling back to basic device movement")
                        batch = {k: v.to(self.device) if hasattr(v, 'to') else v 
                                 for k, v in batch.items()}
                else:
                    # No batch manager, use basic device movement
                    logger.debug("No BatchManager available, using basic device movement")
                    batch = {k: v.to(self.device) if hasattr(v, 'to') else v 
                             for k, v in batch.items()}
            except Exception as e:
                logger.error(f"Error preparing batch: {e}")
                # Last resort - simple implementation
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                         for k, v in batch.items()}
                
            # Forward pass with potential mixed precision
            try:
                # Filter batch to only include keys expected by the model
                model_inputs = self._filter_batch_for_model(batch)
                    
                if hasattr(self, 'amp_manager') and self.amp_manager and self.amp_manager.is_enabled():
                    with self.amp_manager.autocast():
                        outputs = self.model(**model_inputs)
                else:
                    outputs = self.model(**model_inputs)
                    
                # Get loss from outputs with enhanced debugging and robustness
                if isinstance(outputs, dict):
                    # Add more detailed debugging
                    logger.debug(f"Model output keys: {list(outputs.keys())}")
                    
                    if 'loss' in outputs:
                        loss = outputs['loss']
                    else:
                        logger.warning("Model output doesn't contain 'loss' key, computing manually")
                        # Compute loss manually if it's missing
                        if 'logits' in outputs and 'labels' in batch:
                            logger.debug("Computing loss from logits and labels")
                            loss_fct = nn.CrossEntropyLoss()
                            logits = outputs['logits']
                            labels = batch['labels']
                            # Handle labels of -100 (masked tokens)
                            mask = labels != -100
                            if mask.sum() > 0:  # Only compute loss on non-masked tokens
                                loss = loss_fct(
                                    logits.view(-1, logits.size(-1))[mask.view(-1)],
                                    labels.view(-1)[mask.view(-1)]
                                )
                            else:
                                # If all tokens are masked, use dummy loss
                                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                        else:
                            # Create dummy loss as last resort
                            logger.error("Cannot compute loss: missing logits or labels")
                            loss = torch.tensor(1.0, device=self.device, requires_grad=True)
                else:
                    # Compute loss using metrics manager if possible
                    if hasattr(self, 'metrics_manager') and self.metrics_manager:
                        try:
                            metrics = self.metrics_manager.compute_embedding_metrics(outputs, batch)
                            loss = torch.tensor(metrics['loss'], device=self.device)
                        except Exception as metrics_err:
                            logger.error(f"Error computing metrics: {metrics_err}")
                            # Default behavior - assume outputs[0] is loss if it's a tuple
                            if isinstance(outputs, tuple) and len(outputs) > 0:
                                loss = outputs[0]
                            else:
                                logger.error("Cannot determine loss value from outputs")
                                # Create a dummy loss to avoid crashing - training will be bad but not crash
                                loss = torch.tensor(1.0, device=self.device, requires_grad=True)
                    else:
                        # Default behavior - assume outputs[0] is loss
                        loss = outputs[0]
                        
                # Check that loss is not None before proceeding
                if loss is None:
                    logger.error("Model returned None loss - constructing dummy loss")
                    loss = torch.tensor(1.0, device=self.device, requires_grad=True)
                    
                # Additional safety check
                if not isinstance(loss, torch.Tensor):
                    logger.error(f"Loss is not a tensor (got {type(loss)}) - skipping this batch")
                    continue
                    
                # Backward pass and optimization
                self.optimizer.zero_grad()
                
                if self.amp_manager:
                    # Safe backward step with amp manager
                    try:
                        self.amp_manager.backward_step(loss, self.model, self.optimizer)
                    except Exception as amp_err:
                        logger.error(f"Error in AMP backward_step: {amp_err}")
                        # Fallback to normal backward
                        if isinstance(loss, torch.Tensor) and loss.requires_grad:
                            loss.backward()
                            # Clip gradients if specified
                            grad_clip = self.config['training'].get('grad_clip')
                            if grad_clip:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                            self.optimizer.step()
                else:
                    # Normal backward pass
                    loss.backward()
                    # Clip gradients if specified
                    grad_clip = self.config['training'].get('grad_clip')
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()
                    
                # Update scheduler
                self.scheduler.step()
                
                # Update metrics
                epoch_loss += loss.item()
                # TODO: Add accuracy calculation if needed
                
                # Update progress bar
                progress_bar.set_postfix(loss=loss.item())
                
                # Log metrics every N steps - with safe WandB handling
                log_interval = self.config['training'].get('log_interval', 50)
                if self.global_step % log_interval == 0:
                    # Compute batch metrics
                    if self.metrics_manager:
                        batch_metrics = self.metrics_manager.compute_embedding_metrics(outputs, batch)
                    else:
                        batch_metrics = {'loss': loss.item()}
                        
                    # Log metrics
                    self.metrics_logger.log_metrics(
                        batch_metrics, 
                        phase=f'train_step',
                        step=self.global_step
                    )
                    
                    # Update WandB - use our safer method
                    self._safe_wandb_log(batch_metrics, self.global_step)
                
                # Increment counters
                self.global_step += 1
                num_batches += 1
                
            except Exception as e:
                logger.error(f"Error during forward/backward pass: {e}")
                logger.error(traceback.format_exc())  # Add full traceback
                continue
            
        # Compute epoch metrics
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        
        return {
            'loss': avg_loss,
            'accuracy': epoch_accuracy / num_batches if num_batches > 0 else 0.0,
            'lr': self.scheduler.get_last_lr()[0]
        }

    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dict[str, float]: Validation metrics
        """
        self.model.eval()
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        all_metrics = {}
        
        progress_bar = tqdm(
            self.val_loader, 
            desc=f"Epoch {self.current_epoch+1} [Val]",
            leave=False
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                if self.batch_manager:
                    batch = self.batch_manager.prepare_batch(batch)
                else:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                # Filter batch keys for model
                model_inputs = self._filter_batch_for_model(batch)
                
                # Forward pass with potential mixed precision
                if self.amp_manager and self.amp_manager.is_enabled():
                    with self.amp_manager.autocast():
                        outputs = self.model(**model_inputs)
                else:
                    outputs = self.model(**model_inputs)
                    
                # Get loss from outputs
                if isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    # Compute loss using metrics manager if possible
                    if self.metrics_manager:
                        metrics = self.metrics_manager.compute_embedding_metrics(outputs, batch)
                        loss = torch.tensor(metrics['loss'], device=self.device)
                        
                        # Update all metrics
                        for k, v in metrics.items():
                            if k not in all_metrics:
                                all_metrics[k] = 0.0
                            all_metrics[k] += v
                    else:
                        # Default behavior - assume outputs[0] is loss
                        loss = outputs[0]
                        
                epoch_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix(loss=loss.item())
                
                num_batches += 1
                
        # Compute overall metrics
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        
        metrics = {'loss': avg_loss}
        
        # Add other metrics if available
        for k, v in all_metrics.items():
            metrics[k] = v / num_batches if num_batches > 0 else 0.0
            
        return metrics
    
    def log_epoch_results(
        self, 
        epoch: int, 
        train_results: Dict[str, float], 
        val_results: Dict[str, float]
    ) -> None:
        """
        Log epoch results.
        
        Args:
            epoch: Current epoch number
            train_results: Training results
            val_results: Validation results
        """
        # Prepare metrics
        train_metrics = {f'train_{k}': v for k, v in train_results.items()}
        val_metrics = {f'val_{k}': v for k, v in val_results.items()}
        
        # Combine metrics
        combined_metrics = {**train_metrics, **val_metrics}
        combined_metrics['epoch'] = epoch + 1
        combined_metrics['best_val_loss'] = self.best_val_loss
        combined_metrics['early_stop_counter'] = self.early_stop_counter
        
        # Log to console
        logger.info(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_results['loss']:.4f} | "
            f"Val Loss: {val_results['loss']:.4f} | "
            f"Best Val Loss: {self.best_val_loss:.4f} | "
            f"LR: {train_results.get('lr', 0):.6f}"
        )
        
        # Log to metrics logger
        self.metrics_logger.log_metrics(
            combined_metrics, 
            phase=f'epoch_{epoch+1}', 
            step=self.global_step
        )
        
        # Log to WandB - use our safer method
        self._safe_wandb_log(combined_metrics, self.global_step)
    
    def save_model(self, tag: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            tag: Tag to identify the checkpoint
        """
        try:
            from src.common.managers import get_model_manager
            model_manager = get_model_manager()
            
            checkpoint_dir = Path(self.config['output']['dir']) / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Define checkpoint path
            checkpoint_path = checkpoint_dir / f"{tag}.pt"
            
            # Save model with appropriate manager if available
            if hasattr(model_manager, 'save_model'):
                model_manager.save_model(self.model, checkpoint_path)
            else:
                # Fallback - save directly
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'epoch': self.current_epoch,
                    'global_step': self.global_step,
                    'best_val_loss': self.best_val_loss,
                    'config': self.config,
                }
                torch.save(checkpoint, checkpoint_path)
                
            logger.info(f"Model saved to {checkpoint_path}")
            
            # Save metadata
            metadata = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_val_loss': float(self.best_val_loss),
                'time': time.time()
            }
            with open(checkpoint_dir / f"{tag}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
    
    def _filter_batch_for_model(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter the batch to only include keys that the model expects,
        and ensure all tensors are on the correct device.
        
        Args:
            batch: The input batch
            
        Returns:
            Dict[str, Any]: The filtered batch with tensors on correct device
        """
        # Common model input keys for transformer models
        expected_keys = {
            'input_ids', 
            'attention_mask', 
            'token_type_ids',
            'position_ids',
            'labels'
        }
        
        # Check if we have model forward signature information
        if hasattr(self.model, 'forward'):
            import inspect
            try:
                # Get parameter names from model's forward method
                signature = inspect.signature(self.model.forward)
                # Update expected_keys with actual parameters from model
                expected_keys.update(set(signature.parameters.keys()))
            except Exception as e:
                logger.debug(f"Could not inspect model signature: {e}")
        
        # Filter batch and ensure all tensors are on the model device
        model_inputs = {}
        
        # First check model device to ensure consistency
        model_device = self.device
        
        for key, value in batch.items():
            # Only include expected keys
            if key in expected_keys:
                # Ensure tensor is on the right device
                if torch.is_tensor(value):
                    # Check and move to model device if needed
                    if value.device != model_device:
                        logger.debug(f"Moving tensor '{key}' from {value.device} to {model_device}")
                        try:
                            model_inputs[key] = value.to(model_device)
                        except Exception as e:
                            logger.error(f"Error moving tensor {key} to {model_device}: {e}")
                            # Try creating a new tensor on the correct device
                            try:
                                model_inputs[key] = torch.tensor(value.cpu().numpy(), device=model_device)
                            except:
                                # Last resort - skip this tensor
                                logger.error(f"Failed to move tensor {key}, skipping it")
                                continue
                    else:
                        # Already on right device
                        model_inputs[key] = value
                elif isinstance(value, list) and all(torch.is_tensor(t) for t in value):
                    # Handle list of tensors
                    model_inputs[key] = [t.to(model_device) if t.device != model_device else t for t in value]
                else:
                    # Non-tensor values can be kept as is
                    model_inputs[key] = value
        
        # Debug log what we're excluding
        excluded = set(batch.keys()) - set(model_inputs.keys())
        if excluded:
            logger.debug(f"Excluded keys from model input: {excluded}")
        
        # Double-check device consistency before returning
        device_mismatch = False
        for k, v in model_inputs.items():
            if torch.is_tensor(v) and v.device != model_device:
                device_mismatch = True
                logger.error(f"Tensor {k} is on {v.device} instead of {model_device} after filtering!")
        
        if device_mismatch:
            logger.warning(f"Device mismatch detected in batch! Model expects {model_device}")
        
        return model_inputs

__all__ = ['EmbeddingTrainer']
