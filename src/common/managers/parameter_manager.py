# src/common/managers/parameter_manager.py (Refactored)
from __future__ import annotations
import logging
from typing import Dict, Any
import copy
import optuna
from optuna.trial import FixedTrial

from src.common.managers.base_manager import BaseManager  # Corrected import

logger = logging.getLogger(__name__)

class ParameterManager(BaseManager):
    """Centralized manager for handling trial parameters and configuration."""
    
    def __init__(self, base_config: Dict[str, Any]):
        super().__init__()
        self.base_config = base_config
        
        # Define parameter mappings (parameter name -> config section)
        self.parameter_mappings = {
            'learning_rate': 'training',
            'weight_decay': 'training',
            'warmup_ratio': 'training',
            'hidden_dropout_prob': 'training',
            'attention_probs_dropout_prob': 'training',
            'embedding_mask_probability': 'data',
            'max_span_length': 'data'
        }
        
        # Define parameter ranges and types from hyperparameters section
        self.parameter_ranges = {}
        if 'hyperparameters' in base_config:
            for param_name, param_config in base_config['hyperparameters'].items():
                self.parameter_ranges[param_name] = {
                    'type': param_config['type'],
                    'min': param_config['min'],
                    'max': param_config['max']
                }

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes."""
        # For ParameterManager, we likely don't need process-local storage,
        # but we still call the superclass's _initialize_process_local for consistency.
        super()._initialize_process_local(config)
    
    def copy_base_config(self) -> Dict[str, Any]:
        """Create a deep copy of base config."""
        return copy.deepcopy(self.base_config)
    
    def apply_fixed_params(self, config: Dict[str, Any], params: Dict[str, Any]) -> None:
        """Apply parameters from a FixedTrial to config."""
        logger.info(f"Applying fixed parameters: {params}")
        for param_name, value in params.items():
            if param_name in self.parameter_mappings:
                section = self.parameter_mappings[param_name]
                config[section][param_name] = value
                logger.debug(f"Set {section}.{param_name} = {value}")
            else:
                logger.warning(f"Unknown parameter: {param_name}")
    
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters using Optuna trial and return complete config."""
        config = self.copy_base_config()
        self._suggest_params_internal(config, trial)
        self.validate_config(config)
        return config
        
    def _suggest_params_internal(self, config: Dict[str, Any], trial: optuna.Trial) -> None:
        """Internal method to suggest parameters and update config."""
        logger.info("Suggesting new parameters for trial")
        
        for param_name, param_range in self.parameter_ranges.items():
            if param_name not in self.parameter_mappings:
                continue
                
            section = self.parameter_mappings[param_name]
            param_type = param_range['type']
            min_val = param_range['min']
            max_val = param_range['max']
            
            if param_type == 'log':
                value = trial.suggest_float(param_name, min_val, max_val, log=True)
            elif param_type == 'float':
                value = trial.suggest_float(param_name, min_val, max_val)
            elif param_type == 'int':
                value = trial.suggest_int(param_name, min_val, max_val)
            else:
                logger.warning(f"Unknown parameter type: {param_type}")
                continue
                
            config[section][param_name] = value
            logger.debug(f"Suggested {section}.{param_name} = {value}")
    
    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Validate all required parameters are present."""
        required_params = {
            'training': [
                'learning_rate', 'weight_decay', 'warmup_ratio',
                'hidden_dropout_prob', 'attention_probs_dropout_prob',
                'batch_size', 'num_epochs', 'gradient_accumulation_steps',
                'log_every_n_steps', 'optimizer_type', 'early_stopping_patience',
                'early_stopping_min_delta', 'max_grad_norm'
            ],
            'data': [
                'embedding_mask_probability', 'max_span_length', 'csv_path',
                'train_ratio', 'max_length', 'max_predictions', 'num_workers'
            ],
            'model': ['name', 'type'],
            'resources': [
                'max_memory_gb', 'garbage_collection_threshold',
                'max_split_size_mb', 'max_time_hours', 'cache_cleanup_days'
            ]
        }
        
        config_to_validate = config if config is not None else self.base_config
        
        missing_params = []
        for section, params in required_params.items():
            for param in params:
                if param not in config_to_validate[section]:
                    missing_params.append(f"{section}.{param}")
        
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
    
    def get_trial_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get complete trial configuration."""
        config = self.copy_base_config()
        
        if isinstance(trial, FixedTrial):
            logger.info(f"Using FixedTrial with params: {trial.params}")
            self.apply_fixed_params(config, trial.params)
        else:
            self._suggest_params_internal(config, trial)
            
        self.validate_config(config)
        self.log_config(config)
        return config
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log the current configuration state."""
        logger.info("Using trial parameters:")
        logger.info("\nTraining dynamics:")
        logger.info(f"- learning_rate: {config['training']['learning_rate']}")
        logger.info(f"- weight_decay: {config['training']['weight_decay']}")
        logger.info(f"- warmup_ratio: {config['training']['warmup_ratio']}")
        logger.info(f"- batch_size: {config['training']['batch_size']} (fixed)")
        
        logger.info("\nModel architecture:")
        logger.info(f"- hidden_dropout_prob: {config['training']['hidden_dropout_prob']}")
        logger.info(f"- attention_probs_dropout_prob: {config['training']['attention_probs_dropout_prob']}")
        
        logger.info("\nEmbedding-specific:")
        logger.info(f"- embedding_mask_probability: {config['data']['embedding_mask_probability']}")
        logger.info(f"- max_span_length: {config['data']['max_span_length']}")

__all__ = ['ParameterManager']