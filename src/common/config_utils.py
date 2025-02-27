# config_utils.py
# src/common/config_utils.py
from __future__ import annotations
import logging
import yaml
from typing import Dict, Any

logger = logging.getLogger(__name__)

PARAMETER_TYPES = {
    'data': {
        'csv_path': str,
        'train_ratio': float,
        'max_length': int,
        'embedding_mask_probability': float,
        'max_predictions': int,
        'max_span_length': int
    },
    'training': {
        'batch_size': int,
        'num_epochs': int,
        'seed': int,
        'n_jobs': int,
        'num_trials': int,
        'n_startup_trials': int,
        'learning_rate': float,
        'weight_decay': float,
        'fp16': bool,
        'optimizer_type': str,
        'save_every_n_epochs': int,
        'early_stopping_patience': int,
        'early_stopping_min_delta': float,
        'max_grad_norm': float,
        'hidden_dropout_prob': float,
        'attention_probs_dropout_prob': float,
        'gradient_accumulation_steps': int,
        'logging_steps': int,
        'eval_steps': int,
        'save_top_k': int,
        'log_every_n_steps': int,
        'debug_logging': bool,
        'study_name': str,
        'warmup_ratio': float,
        'num_workers': int
    },
    'model': {
        'name': str,
        'type': str,
        'stage': str,
        'tie_weights': bool,
        'hidden_dropout_prob': float,
        'attention_probs_dropout_prob': float
    },
    'optimizer': {
        'learning_rate': float,
        'weight_decay': float,
        'adam_beta1': float,
        'adam_beta2': float,
        'adam_epsilon': float,
        'max_grad_norm': float
    },
    'output': {
        'dir': str,
        'storage_dir': str,
        'save_model': bool,
        'save_optimizer': bool,
        'save_scheduler': bool,
        'wandb': {
            'enabled': bool,
            'project': str,
            'api_key': str,
            'tags': list
        }
    },
    'hyperparameters': {
        'learning_rate': dict,
        'weight_decay': dict,
        'warmup_ratio': dict,
        'embedding_mask_probability': dict,
        'max_span_length': dict,
        'hidden_dropout_prob': dict,
        'attention_probs_dropout_prob': dict
    },
    'resources': {
        'max_memory_gb': float,
        'gpu_memory_gb': float,
        'garbage_collection_threshold': float,
        'max_split_size_mb': float,
        'max_time_hours': float,
        'cache_cleanup_days': float
    },
    'scheduler': {
        'type': str,
        'warmup_ratio': float,
        'min_lr_ratio': float,
        'use_scheduler': bool
    }
}

def _convert_value(value: Any, target_type: type) -> Any:
    try:
        if target_type == bool:
            if isinstance(value, str):
                return value.lower() == 'true'
            return bool(value)
        return target_type(value)
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting {value} to {target_type}: {str(e)}")
        raise

def _convert_config_types(config: Dict[str, Any], type_map: Dict[str, Any]) -> Dict[str, Any]:
    converted: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            if key in type_map and isinstance(type_map[key], dict):
                converted[key] = _convert_config_types(value, type_map[key])
            else:
                converted[key] = value
        elif key in type_map:
            try:
                converted[key] = _convert_value(value, type_map[key])
                logger.debug(f"Converted {key}={value} to {type(converted[key])}")
            except Exception as e:
                logger.error(f"Error converting {key}={value}: {str(e)}")
                raise
        else:
            converted[key] = value
    return converted

def load_yaml_config(filepath: str) -> Dict[str, Any]:
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:
                logging.warning(f"YAML file at {filepath} is empty.")
                return {}
            config =  _convert_config_types(config, PARAMETER_TYPES)
            logger.debug(f"Loaded YAML configuration: {config}")
            return config
    except FileNotFoundError:
        logging.error(f"YAML file not found: {filepath}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {filepath}: {e}")
        return {}
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading YAML: {filepath}: {e}")
        return {}

__all__ = ['load_yaml_config']