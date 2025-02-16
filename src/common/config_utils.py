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
        'num_workers': int
    },
    'training': {
        'batch_size': int,
        'num_epochs': int,
        'num_workers': int,
        'seed': int,
        'n_jobs': int,
        'num_trials': int,
        'n_startup_trials': int,
        'learning_rate': float,
        'weight_decay': float,
        'fp16': bool,
        "optimizer_type": str,
        "save_every_n_epochs": int,
        'early_stopping_patience': int,
        'early_stopping_min_delta': float,
        "max_grad_norm": float
    },
    'model': {
        'name': str,
        'type': str, #Added type
        "stage":str # Added stage
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
        "storage_dir": str, #Added storage
        "wandb": dict
    },
    'study_name': str,
    'hyperparameters': dict,
    'resources': {
        'max_memory_gb': float,
        'gpu_memory_gb': float, #Added GPU
        'garbage_collection_threshold': float,
        'max_split_size_mb': float, #Not used for now
        'max_time_hours': float,
        'cache_cleanup_days': float #Not used for now
    },
    'scheduler': dict
}

def _convert_value(value: Any, target_type: type) -> Any:
    """Convert value to target type."""
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
    """Recursively convert config values to correct types."""
    converted: Dict[str, Any] = {}  # Initialize to correct type
    for key, value in config.items():
        if isinstance(value, dict):
            if key in type_map and isinstance(type_map[key], dict):
                converted[key] = _convert_config_types(value, type_map[key])
            else:
                converted[key] = value  # Keep as is if no type spec
        elif key in type_map:
            try:
                converted[key] = _convert_value(value, type_map[key])
                logger.debug(f"Converted {key}={value} to {type(converted[key])}")
            except Exception as e:
                logger.error(f"Error converting {key}={value}: {str(e)}")
                raise
        else:
            converted[key] = value  # Keep as is if not in type map
    return converted

def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """Loads a YAML configuration file and converts types."""
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
            # Add basic validation, check if config is None or empty
            if config is None:
                logging.warning(f"YAML file at {filepath} is empty.")
                return {}
            # Convert the config types
            config =  _convert_config_types(config, PARAMETER_TYPES)
            return config
    except FileNotFoundError:
        logging.error(f"YAML file not found: {filepath}")
        return {}  # Return an empty dict to signal failure
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {filepath}: {e}")
        return {}
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading YAML: {filepath}: {e}")
        return {}

__all__ = ['load_yaml_config']