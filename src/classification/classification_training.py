# src/classification/classification_training.py
from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
import math

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import optuna

from src.common.managers import (
    get_tokenizer_manager,
    get_data_manager,
    get_model_manager,
    get_parameter_manager,
    get_directory_manager,
    get_optuna_manager,
    get_shared_tokenizer,
    get_wandb_manager
)

from src.common.utils import seed_everything, create_optimizer, create_scheduler
from src.classification.classification_trainer import ClassificationTrainer
from src.common.utils import load_yaml_config


logger = logging.getLogger(__name__)

def get_classification_model():
    from src.classification.model import ClassificationBert
    return ClassificationBert

def run_classification_optimization(embedding_model_path: str, config_path: str, study_name: Optional[str] = None) -> Dict[str, Any]:
    config = load_yaml_config(config_path)
    n_jobs = config['training']['n_jobs']
    n_trials = config['training']['num_trials']
    study_name = study_name or 'classification_optimization'

    output_dir = Path(config['output']['dir']).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading labels for classification training")

    study_manager = get_optuna_manager()

    def objective(trial):
        local_vars: Dict[str, Any] = {
            'model': None,
            'tokenizer': None,
            'optimizer': None,
            'scheduler': None,
            'trainer': None,
            'train_loader': None,
            'val_loader': None
        }

        try:
            parameter_manager = get_parameter_manager()
            trial_config = parameter_manager.get_trial_config(trial)
            trial_config['model']['name'] = embedding_model_path
            if trial_config["training"]["num_trials"] > 1:
                wandb_manager = get_wandb_manager()
                wandb_manager.init_trial(trial.number)
            data_manager = get_data_manager()
            tokenizer_manager = get_tokenizer_manager()
            model_manager = get_model_manager()
            directory_manager = get_directory_manager()
            tokenizer = tokenizer_manager.get_worker_tokenizer(
                worker_id=trial.number,
                model_name=embedding_model_path,
                model_type='classification'
            )
            local_vars['model'] = get_classification_model()(
                config=trial_config,
                num_labels=trial_config['model']['num_labels']
            )

            train_dataset, val_dataset = data_manager.create_datasets(
                trial_config,
                stage='classification'
            )
            train_loader, val_loader = data_manager.create_dataloaders(
                trial_config,
                train_dataset,
                val_dataset
            )
            local_vars['train_loader'] = train_loader
            local_vars['val_loader'] = val_loader

            local_vars['optimizer'] = create_optimizer(local_vars['model'], trial_config['training'])
            local_vars['scheduler'] = create_scheduler(local_vars['optimizer'], len(local_vars['train_loader'].dataset), trial_config)

            trial_output_dir = directory_manager.base_dir / "trials" / f"trial_{trial.number}"
            metrics_dir = trial_output_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)

            local_vars['trainer'] = ClassificationTrainer(
                model=local_vars['model'],
                train_loader=local_vars['train_loader'],
                val_loader=local_vars['val_loader'],
                config=trial_config,
                metrics_dir= metrics_dir,
                optimizer=local_vars['optimizer'],
                scheduler=local_vars['scheduler'],
                is_trial=True,
                trial=trial,
                wandb_manager= wandb_manager if trial_config["training"]["num_trials"] > 1 else None,
                job_id=trial.number,
                train_dataset=train_dataset,
                val_dataset=val_dataset
            )

            try:
                local_vars['trainer'].train(int(trial_config['training']['num_epochs']))
                val_metrics = local_vars['trainer'].validate()

                val_loss = val_metrics['loss']
                val_accuracy = val_metrics['accuracy']

                objective_value = (
                    0.6 * val_loss +
                    0.4 * (1 - val_accuracy)
                )

                if local_vars['trainer'].early_stopping_triggered:
                    raise optuna.TrialPruned("Early stopping triggered")

                if not math.isfinite(val_metrics['loss']):
                    raise optuna.TrialPruned("Non-finite loss detected")

                if torch.cuda.is_available():
                    memory_used = torch.cuda.max_memory_allocated() / (1024**3)
                    if memory_used > config['resources']['max_memory_gb']:
                        raise optuna.TrialPruned(f"Memory limit exceeded: {memory_used:.2f}GB")

                return objective_value

            finally:
                if torch.cuda.is_available():
                    for var_name, var in local_vars.items():
                        if var is not None:
                            del var
                    torch.cuda.empty_cache()
                    logger.info("GPU memory cleaned after classification trial")

        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise optuna.TrialPruned(f"Trial failed: {str(e)}")
        finally:
            if config["training"]["num_trials"] > 1:
                wandb_manager.finish_trial(trial.number)
            if torch.cuda.is_available():
                for var_name, var in local_vars.items():
                    if var is not None:
                        del var
                torch.cuda.empty_cache()
                logger.info("Cleaned up trial resources")

    best_trial = study_manager.optimize(objective)
    logger.info(f"Best classification trial parameters: {best_trial.params}")

    return best_trial.params

def train_final_model(embedding_model_path: str, best_params: Dict[str, Any], config_path: str, output_dir: Optional[Path] = None) -> None:
    config = load_yaml_config(config_path)
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config['training']['seed'])
    parameter_manager = get_parameter_manager()
    data_manager = get_data_manager()
    tokenizer_manager = get_tokenizer_manager()
    directory_manager = get_directory_manager()
    model_manager = get_model_manager()
    if config["training"]["num_trials"] > 1:
        wandb_manager = get_wandb_manager()
    tokenizer = tokenizer_manager.get_worker_tokenizer(
            worker_id=0,
            model_name=embedding_model_path,
            model_type='classification'
    )
    config.update(best_params)
    config['model']['name'] = embedding_model_path
    config['training']['batch_size'] = 16

    train_dataset, val_dataset = data_manager.create_datasets(
        config,
        stage='classification'
    )
    train_loader, val_loader = data_manager.create_dataloaders(
        config,
        train_dataset,
        val_dataset
    )

    model = get_classification_model()(config=config, num_labels=config['model']['num_labels'])
    optimizer = create_optimizer(model, config['training'])
    scheduler = create_scheduler(optimizer, len(train_loader.dataset), config['training'])

    if config["training"]["num_trials"] > 1:
        wandb_manager.init_final_training()
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        metrics_dir= output_dir / 'metrics',
        optimizer=optimizer,
        scheduler=scheduler,
        is_trial=False,
        trial=None,
        wandb_manager= wandb_manager if config["training"]["num_trials"] > 1 else None,
        job_id=0,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )

    try:
        trainer.train(int(config['training']['num_epochs']))

        val_metrics = trainer.validate()
        logger.info("Final classification metrics:")
        logger.info(f"- Loss: {val_metrics['loss']:.4f}")
        logger.info(f"- Accuracy: {val_metrics['accuracy']:.4f}")

    finally:
        if config["training"]["num_trials"] > 1:
            wandb_manager.finish()
        if torch.cuda.is_available():
            del model
            del optimizer
            del scheduler
            del trainer
            del tokenizer
            torch.cuda.empty_cache()
            logger.info("GPU memory cleaned after final training")