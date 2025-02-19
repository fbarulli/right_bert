# src/common/managers/config_validator.py
from __future__ import annotations
import logging
import traceback
from typing import Dict, Any, Set, Optional

logger = logging.getLogger(__name__)

class ConfigValidator:
    """
    Configuration validator for dependency injection system.
    
    This class handles:
    - Required configuration validation
    - Type checking
    - Cross-section validation
    - Dependency configuration validation
    """

    def __init__(self):
        """Initialize validator with required configuration definitions."""
        # Define required sections and their fields
        self._required_sections: Dict[str, Set[str]] = {
            'output': {
                'dir',
                'storage_dir',
                'wandb'
            },
            'training': {
                'batch_size',
                'num_epochs',
                'num_workers',
                'learning_rate',
                'seed',
                'optimizer_type',
                'scheduler',
                'fp16',
                'profiler',
                'cuda_graph',
                'gradient_accumulation_steps',
                'max_grad_norm',
                'early_stopping_patience',
                'early_stopping_min_delta'
            },
            'model': {
                'name',
                'type',
                'stage',
                'config'
            },
            'data': {
                'csv_path',
                'train_ratio',
                'max_length',
                'embedding_mask_probability',
                'max_predictions',
                'max_span_length'
            },
            'resources': {
                'max_memory_gb',
                'gpu_memory_gb',
                'garbage_collection_threshold',
                'max_split_size_mb'
            }
        }

        # Define nested section requirements
        self._nested_requirements: Dict[str, Dict[str, Set[str]]] = {
            'output.wandb': {
                'enabled',
                'project',
                'api_key',
                'tags'
            },
            'training.scheduler': {
                'use_scheduler',
                'warmup_ratio'
            },
            'training.profiler': {
                'enabled',
                'activities',
                'schedule',
                'record_shapes',
                'profile_memory',
                'with_stack',
                'with_flops',
                'export_chrome_trace'
            },
            'training.cuda_graph': {
                'enabled',
                'warmup_steps'
            },
            'model.config': {
                'hidden_size',
                'num_hidden_layers',
                'num_attention_heads',
                'intermediate_size'
            }
        }

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration for manager initialization.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If any required sections or fields are missing
            TypeError: If field types are incorrect
        """
        try:
            # Basic validation
            if not config:
                raise ValueError("Configuration cannot be empty")

            # Validate top-level sections
            self._validate_sections(config)

            # Validate each section's fields
            for section, required_fields in self._required_sections.items():
                self._validate_section_fields(config, section, required_fields)

            # Validate nested sections
            for nested_path, required_fields in self._nested_requirements.items():
                self._validate_nested_section(config, nested_path, required_fields)

            # Validate specific field types and values
            self._validate_field_types(config)
            self._validate_field_values(config)

            # Cross-section validation
            self._validate_cross_section_dependencies(config)

            logger.info(
                "Configuration validation successful:\n" +
                "\n".join(f"- {section}: {len(fields)} fields"
                         for section, fields in self._required_sections.items())
            )

        except Exception as e:
            logger.error("Configuration validation failed")
            logger.error(traceback.format_exc())
            raise

    def _validate_sections(self, config: Dict[str, Any]) -> None:
        """Validate top-level sections exist."""
        missing_sections = set(self._required_sections.keys()) - set(config.keys())
        if missing_sections:
            raise ValueError(f"Missing required config sections: {missing_sections}")

    def _validate_section_fields(
        self,
        config: Dict[str, Any],
        section: str,
        required_fields: Set[str]
    ) -> None:
        """Validate fields in a section exist."""
        section_config = config.get(section, {})
        missing_fields = required_fields - set(section_config.keys())
        if missing_fields:
            raise ValueError(
                f"Missing required fields in {section} section: {missing_fields}"
            )

    def _validate_nested_section(
        self,
        config: Dict[str, Any],
        nested_path: str,
        required_fields: Set[str]
    ) -> None:
        """Validate nested section fields exist."""
        try:
            # Navigate to nested section
            current = config
            for part in nested_path.split('.'):
                current = current.get(part, {})

            # Check required fields
            missing_fields = required_fields - set(current.keys())
            if missing_fields:
                raise ValueError(
                    f"Missing required fields in {nested_path}: {missing_fields}"
                )

        except Exception as e:
            raise ValueError(
                f"Error validating nested section {nested_path}: {str(e)}"
            )

    def _validate_field_types(self, config: Dict[str, Any]) -> None:
        """Validate field types are correct."""
        # Training section types
        training = config['training']
        if not isinstance(training['batch_size'], int):
            raise TypeError("batch_size must be an integer")
        if not isinstance(training['learning_rate'], (int, float)):
            raise TypeError("learning_rate must be a number")
        if not isinstance(training['fp16'], bool):
            raise TypeError("fp16 must be a boolean")

        # Resource section types
        resources = config['resources']
        if not isinstance(resources['max_memory_gb'], (int, float)):
            raise TypeError("max_memory_gb must be a number")
        if not isinstance(resources['gpu_memory_gb'], (int, float)):
            raise TypeError("gpu_memory_gb must be a number")

        # Data section types
        data = config['data']
        if not isinstance(data['train_ratio'], float):
            raise TypeError("train_ratio must be a float")
        if not isinstance(data['max_length'], int):
            raise TypeError("max_length must be an integer")

    def _validate_field_values(self, config: Dict[str, Any]) -> None:
        """Validate field values are in acceptable ranges."""
        # Training values
        training = config['training']
        if training['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")
        if training['learning_rate'] <= 0:
            raise ValueError("learning_rate must be positive")
        if training['num_workers'] < 0:
            raise ValueError("num_workers cannot be negative")

        # Resource values
        resources = config['resources']
        if resources['max_memory_gb'] <= 0:
            raise ValueError("max_memory_gb must be positive")
        if resources['gpu_memory_gb'] <= 0:
            raise ValueError("gpu_memory_gb must be positive")

        # Data values
        data = config['data']
        if not 0 < data['train_ratio'] < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if data['max_length'] <= 0:
            raise ValueError("max_length must be positive")

    def _validate_cross_section_dependencies(self, config: Dict[str, Any]) -> None:
        """Validate dependencies between different sections."""
        # Model stage dependencies
        stage = config['model']['stage']
        if stage == 'embedding':
            if 'embedding_mask_probability' not in config['data']:
                raise ValueError(
                    "embedding_mask_probability required for embedding stage"
                )
        elif stage == 'classification':
            if 'num_labels' not in config['model']:
                raise ValueError("num_labels required for classification stage")

        # Resource dependencies
        if config['training']['fp16']:
            if not config['resources'].get('gpu_memory_gb'):
                raise ValueError("gpu_memory_gb required when fp16 is enabled")

        # Scheduler dependencies
        if config['training']['scheduler']['use_scheduler']:
            if 'warmup_ratio' not in config['training']['scheduler']:
                raise ValueError("warmup_ratio required when scheduler is enabled")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Convenience function to validate configuration.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If validation fails
    """
    validator = ConfigValidator()
    validator.validate_config(config)


__all__ = ['validate_config', 'ConfigValidator']
