# src/common/managers/parameter_manager.py
from __future__ import annotations
import logging
import os  # Add this import
import traceback  # Add this import
from typing import Dict, Any, Optional, Set
import copy
import optuna
from optuna.trial import FixedTrial
import threading

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
        base_config: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ParameterManager.

        Args:
            base_config: Base configuration dictionary (falls back to config if None)
            config: Optional additional configuration dictionary

        Raises:
            ValueError: If neither base_config nor config is provided
        """
        self._local = threading.local()
        self._base_config = base_config if base_config is not None else (config or {})
        super().__init__(config)
        
        if not self._base_config:
            raise ValueError("No configuration provided - both base_config and config cannot be empty")

        self._local.parameter_ranges = {}

        # Initialize parameter ranges from hyperparameters section
        if 'hyperparameters' in self._base_config:
            for param_name, param_config in self._base_config['hyperparameters'].items():
                self._local.parameter_ranges[param_name] = {
                    'type': param_config['type'],
                    'min': param_config['min'],
                    'max': param_config['max']
                }

    def _setup_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Setup process-local parameter ranges."""
        self._local.parameter_ranges = {}

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local attributes.

        Args:
            config: Optional configuration dictionary that overrides the one from constructor
        """
        if hasattr(self._local, 'initialized') and self._local.initialized:
            logger.debug(f"ParameterManager already initialized for process {os.getpid()}")
            return

        try:
            logger.debug(f"Initializing ParameterManager for process {os.getpid()} and thread {threading.current_thread().name}")
            super()._initialize_process_local(config)

            effective_config = config if config is not None else self._config
            parameter_config = self.get_config_section(effective_config, 'parameters')

            # CRITICAL: Always set base_config to avoid AttributeError later
            self._local.base_config = self._base_config or effective_config
            
            # Set necessary attributes
            self._local.parameter_ranges = parameter_config.get('ranges', {})
            self._local.search_space = {}
            self._local.hyperparameters = {}
            self._local.initialized = True

            self._log_process_info()

        except Exception as e:
            logger.error(f"Failed to initialize ParameterManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _log_process_info(self) -> None:
        """Log current process and ParameterManager state information."""
        if hasattr(self._local, 'pid'):
            logger.debug(
                f"\nProcess Info:\n"
                f"- PID: {self._local.pid}\n"
                f"- PPID: {os.getppid()}\n"
                f"- Thread: {threading.current_thread().name}\n"
                f"\nParameterManager State:\n"
                f"- initialized: {getattr(self._local, 'initialized', False)}\n"
                f"- parameter_ranges: {getattr(self._local, 'parameter_ranges', None)}\n"
            )
        else:
            logger.debug("PID attribute not found in _local")

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
                if (param_name in self.PARAMETER_MAPPINGS):
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
        """Generate a configuration for the given trial by sampling from the search space."""
        try:
            # Safety check for initialization
            if not self.is_initialized():
                logger.warning("ParameterManager not initialized, forcing initialization")
                self._local = threading.local()
                self._local.pid = os.getpid()
                self._local.initialized = True
                self._local.base_config = self._config or {}  # Use stored config with empty dict fallback
                self._local.search_space = {}  # Empty but safe
                self._local.param_ranges = {}   # Empty but safe
                self._local.hyperparameters = {}
            
            # Safety check for base_config attribute
            if not hasattr(self._local, 'base_config') or self._local.base_config is None:
                logger.warning("Missing base_config attribute in ParameterManager, using stored config")
                self._local.base_config = self._config or {}
            
            self.ensure_initialized()
            
            # Create a deep copy of the base config
            config = copy.deepcopy(self._local.base_config)
            
            # Sample parameters for this trial - REPLACE _sample_params with direct implementation
            params = {}
            try:
                # Just use the direct implementation since _sample_params doesn't exist
                logger.info("Sampling parameters for trial")
                
                # Use parameter ranges from config if available
                parameter_ranges = self._local.parameter_ranges if hasattr(self._local, 'parameter_ranges') else {}
                
                for param_name, param_range in parameter_ranges.items():
                    if param_name not in self.PARAMETER_MAPPINGS:
                        continue
                        
                    param_type = param_range.get('type', 'float')
                    min_val = param_range.get('min', 0.0)
                    max_val = param_range.get('max', 1.0)
                    
                    if param_type == 'log':
                        value = trial.suggest_float(param_name, min_val, max_val, log=True)
                    elif param_type == 'float':
                        value = trial.suggest_float(param_name, min_val, max_val)
                    elif param_type == 'int':
                        value = trial.suggest_int(param_name, min_val, max_val)
                    else:
                        continue  # Skip unknown parameter types
                        
                    # Map parameter to the right section
                    section = self.PARAMETER_MAPPINGS[param_name]
                    
                    # Create section if it doesn't exist
                    if section not in params:
                        params[section] = {}
                        
                    # Add parameter to its section
                    params[section][param_name] = value
                    
                logger.debug(f"Sampled parameters: {params}")
            except Exception as e:
                logger.error(f"Error sampling parameters: {str(e)}")
                logger.error("Stack trace:", exc_info=True)
                # Continue with empty params rather than failing
            
            # Add trial number to config for tracking
            if 'training' not in config:
                config['training'] = {}
            config['training']['trial_number'] = trial.number
            
            # Update config with sampled parameters
            for section_name, section_params in params.items():
                # Create section if it doesn't exist
                if section_name not in config:
                    config[section_name] = {}
                
                # Update all parameters in this section
                for param_name, param_value in section_params.items():
                    config[section_name][param_name] = param_value
                    
            # Log the generated configuration
            logger.debug(f"Generated trial config for trial {trial.number}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error generating trial config: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            # Return base config as fallback
            return copy.deepcopy(self._config)

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
            if hasattr(self, '_local'):
                if hasattr(self._local, 'parameter_ranges'):
                    del self._local.parameter_ranges
                else:
                    logger.debug("parameter_ranges attribute not found in _local")
                
                # Use a safety check for the pid attribute
                pid = getattr(self._local, 'pid', os.getpid())
                logger.info(f"Cleaned up ParameterManager for process {pid}")
                
            super().cleanup()
        except AttributeError as e:
            logger.debug(f"AttributeError during ParameterManager cleanup: {e}")
            # Still try to call the parent cleanup
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
        
        # Check model configuration
        if not config['model']['name']:
            raise ValueError("Missing required configuration field: model.name")
            
        # Check training configuration
        training_config = config['training']
        if 'batch_size' not in training_config:
            raise ValueError("Missing required configuration field: training.batch_size")
            
        # Check for either epochs or num_epochs
        has_epochs = 'epochs' in training_config or 'num_epochs' in training_config
        if not has_epochs:
            raise ValueError("Missing required configuration field: training.epochs or training.num_epochs")

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get all hyperparameters.
        
        Returns:
            Dict[str, Any]: All hyperparameters
        """
        self.ensure_initialized()
        return self._local.hyperparameters.copy()
    
    def get_hyperparameter(self, name: str, default: Any = None) -> Any:
        """Get a single hyperparameter by name.
        
        Args:
            name: The hyperparameter name
            default: Default value if the parameter doesn't exist
            
        Returns:
            Any: The hyperparameter value or default
        """
        self.ensure_initialized()
        return self._local.hyperparameters.get(name, default)
    
    def set_hyperparameter(self, name: str, value: Any) -> None:
        """Set a hyperparameter value.
        
        Args:
            name: The hyperparameter name
            value: The hyperparameter value
        """
        self.ensure_initialized()
        self._local.hyperparameters[name] = value
        logger.debug(f"Set hyperparameter '{name}' to {value}")
    
    def get_model_parameter(self, name: str, default: Any = None) -> Any:
        """Get a model parameter by name.
        
        Args:
            name: The parameter name
            default: Default value if the parameter doesn't exist
            
        Returns:
            Any: The parameter value or default
        """
        self.ensure_initialized()
        return self._local.model.get(name, default)
    
    def get_training_parameter(self, name: str, default: Any = None) -> Any:
        """Get a training parameter by name.
        
        Args:
            name: The parameter name
            default: Default value if the parameter doesn't exist
            
        Returns:
            Any: The parameter value or default
        """
        self.ensure_initialized()
        return self._local.training.get(name, default)
    
    def get_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get all parameter ranges for hyperparameter tuning.
        
        Returns:
            Dict[str, Dict[str, Any]]: Parameter ranges by category
        """
        self.ensure_initialized()
        ranges = {}
        
        # Get ranges from config if they exist
        if 'parameter_ranges' in self._local.config:
            ranges = self._local.config['parameter_ranges'].copy()
        
        return ranges
    
    def get_parameter_range(self, name: str) -> Dict[str, Any]:
        """Get a specific parameter range.
        
        Args:
            name: Parameter name
            
        Returns:
            Dict[str, Any]: Range specification for the parameter
        """
        self.ensure_initialized()
        ranges = self.get_parameter_ranges()
        
        for category, params in ranges.items():
            if name in params:
                return params[name]
        
        return {}
    
    def save_parameters(self, filepath: str) -> None:
        """Save all parameters to a JSON file.
        
        Args:
            filepath: Path to save the parameters
        """
        self.ensure_initialized()
        
        params = {
            'hyperparameters': self._local.hyperparameters,
            'model': self._local.model,
            'training': self._local.training
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(params, f, indent=2)
            logger.info(f"Parameters saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save parameters: {str(e)}")
            raise
    
    def load_parameters(self, filepath: str) -> Dict[str, Any]:
        """Load parameters from a JSON file.
        
        Args:
            filepath: Path to load the parameters from
            
        Returns:
            Dict[str, Any]: Loaded parameters
        """
        self.ensure_initialized()
        
        try:
            with open(filepath, 'r') as f:
                params = json.load(f)
                
            if 'hyperparameters' in params:
                self._local.hyperparameters.update(params['hyperparameters'])
            
            if 'model' in params:
                self._local.model.update(params['model'])
            
            if 'training' in params:
                self._local.training.update(params['training'])
                
            logger.info(f"Parameters loaded from {filepath}")
            return params
        except Exception as e:
            logger.error(f"Failed to load parameters: {str(e)}")
            raise
    
    def update_from_trial(self, trial_params: Dict[str, Any]) -> None:
        """Update parameters from a trial.
        
        Args:
            trial_params: Parameters from a trial
        """
        self.ensure_initialized()
        
        for name, value in trial_params.items():
            self.set_hyperparameter(name, value)
            
        logger.info(f"Parameters updated from trial: {trial_params}")

__all__ = ['ParameterManager']