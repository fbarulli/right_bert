# src/common/managers/parameter_manager.py
from __future__ import annotations
import logging
from typing import Dict, Any, Optional, Set
import copy
import optuna
from optuna.trial import FixedTrial

from src.common.managers.base_manager import BaseManager
from src.common.managers.config_validator import validate_config

logger = logging.getLogger(__name__)

class ParameterManager(BaseManager):
    """
    Centralized manager for handling trial parameters and configuration.

    This manager handles:
    - Parameter mapping and ranges
    - Trial parameter suggestion
    - Configuration validation
    - Parameter logging
    """

    # Parameter mappings (parameter name -> config section)
    PARAMETER_MAPPINGS: Dict[str, str] = {
        'learning_rate': 'training',
        'weight_decay': 'training',
        'warmup_ratio': 'training',
        'hidden_dropout_prob': 'training',
        'attention_probs_dropout_prob': 'training',
        'embedding_mask_probability': 'data',
        'max_span_length': 'data'
    }

    def __init__(
        self,
        base_config: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ParameterManager.

        Args:
            base_config: Base configuration dictionary
            config: Optional additional configuration dictionary

        Raises:
            ValueError: If base_config is invalid
        """
        if not base_config:
            raise ValueError("base_config cannot be empty")

        super().__init__(config)
        self._base_config = base_config
        self._local.parameter_ranges = {}

        # Initialize parameter ranges from hyperparameters section
        if 'hyperparameters' in base_config:
            for param_name, param_config in base_config['hyperparameters'].items():
                self._local.parameter_ranges[param_name] = {
                    'type': param_config['type'],
                    'min': param_config['min'],
                    'max': param_config['max']
                }

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local attributes.

        Args:
            config: Optional configuration dictionary that overrides the one from constructor
        """
        try:
            super()._initialize_process_local(config)
            logger.info(f"ParameterManager initialized for process {self._local.pid}")

        except Exception as e:
            logger.error(f"Failed to initialize ParameterManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def copy_base_config(self) -> Dict[str, Any]:
        """
        Create a deep copy of base config.

        Returns:
            Dict[str, Any]: Copy of base configuration
        """
        return copy.deepcopy(self._base_config)

    def apply_fixed_params(
        self,
        config: Dict[str, Any],
        params: Dict[str, Any]
    ) -> None:
        """
        Apply parameters from a FixedTrial to config.

        Args:
            config: Configuration dictionary to modify
            params: Parameters to apply

        Raises:
            ValueError: If a parameter section is missing
        """
        try:
            logger.info(f"Applying fixed parameters: {params}")
            for param_name, value in params.items():
                if param_name in self.PARAMETER_MAPPINGS:
                    section = self.PARAMETER_MAPPINGS[param_name]
                    if section not in config:
                        raise ValueError(f"Missing config section: {section}")
                    config[section][param_name] = value
                    logger.debug(f"Set {section}.{param_name} = {value}")
                else:
                    logger.warning(f"Unknown parameter: {param_name}")

        except Exception as e:
            logger.error(f"Error applying fixed parameters: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest parameters using Optuna trial.

        Args:
            trial: Optuna trial instance

        Returns:
            Dict[str, Any]: Complete configuration with suggested parameters
        """
        self.ensure_initialized()
        try:
            config = self.copy_base_config()
            self._suggest_params_internal(config, trial)
            validate_config(config)
            return config

        except Exception as e:
            logger.error(f"Error suggesting parameters: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _suggest_params_internal(
        self,
        config: Dict[str, Any],
        trial: optuna.Trial
    ) -> None:
        """
        Internal method to suggest parameters and update config.

        Args:
            config: Configuration dictionary to update
            trial: Optuna trial instance
        """
        try:
            logger.info("Suggesting new parameters for trial")

            for param_name, param_range in self._local.parameter_ranges.items():
                if param_name not in self.PARAMETER_MAPPINGS:
                    continue

                section = self.PARAMETER_MAPPINGS[param_name]
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

        except Exception as e:
            logger.error(f"Error suggesting parameters internally: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_trial_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Get complete trial configuration.

        Args:
            trial: Optuna trial instance

        Returns:
            Dict[str, Any]: Complete configuration for the trial
        """
        self.ensure_initialized()
        try:
            config = self.copy_base_config()

            if isinstance(trial, FixedTrial):
                logger.info(f"Using FixedTrial with params: {trial.params}")
                self.apply_fixed_params(config, trial.params)
            else:
                self._suggest_params_internal(config, trial)

            validate_config(config)
            self.log_config(config)
            return config

        except Exception as e:
            logger.error(f"Error getting trial config: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log the current configuration state.

        Args:
            config: Configuration dictionary to log
        """
        try:
            logger.info("\nTrial Parameters:")

            logger.info("\nTraining dynamics:")
            logger.info(f"- learning_rate: {config['training']['learning_rate']}")
            logger.info(f"- weight_decay: {config['training']['weight_decay']}")
            logger.info(f"- warmup_ratio: {config['training']['warmup_ratio']}")
            logger.info(f"- batch_size: {config['training']['batch_size']} (fixed)")

            logger.info("\nModel architecture:")
            logger.info(
                f"- hidden_dropout_prob: {config['training']['hidden_dropout_prob']}"
            )
            logger.info(
                f"- attention_probs_dropout_prob: "
                f"{config['training']['attention_probs_dropout_prob']}"
            )

            logger.info("\nEmbedding-specific:")
            logger.info(
                f"- embedding_mask_probability: "
                f"{config['data']['embedding_mask_probability']}"
            )
            logger.info(f"- max_span_length: {config['data']['max_span_length']}")

        except Exception as e:
            logger.error(f"Error logging config: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def cleanup(self) -> None:
        """Clean up parameter manager resources."""
        try:
            self._local.parameter_ranges.clear()
            logger.info(f"Cleaned up ParameterManager for process {self._local.pid}")
            super().cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up ParameterManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def validate_config(self, config):
        """Validate the configuration parameters.

        Args:
            config: The configuration to validate

        Raises:
            ValueError: If the configuration is invalid
        """
        if not config:
            raise ValueError("Configuration cannot be empty")
        
        required_fields = ['model_name', 'batch_size', 'epochs']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")

__all__ = ['ParameterManager']