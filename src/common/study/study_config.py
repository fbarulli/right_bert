# src/common/study/study_config.py
# src/common/study/study_config.py
from __future__ import annotations
import logging
from typing import Dict, Any
import optuna
from optuna.samplers import TPESampler
from src.common.managers.parameter_manager import ParameterManager

logger = logging.getLogger(__name__)

class StudyConfig:
    """Manages study configuration and parameter distributions."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.param_manager = ParameterManager(config)  # Use ParameterManager
        self.sampler = self._create_sampler()

        logger.info("\n=== Study Configuration ===")
        logger.info(f"Study Name: {self.config['training']['study_name']}")
        logger.info(f"Number of Jobs: {self.config['training']['n_jobs']}")
        logger.info(f"Total Trials: {self.config['training']['num_trials']}")
        logger.info(f"Epochs per Trial: {self.config['training']['num_epochs']}")
        logger.info(f"Seed: {self.config['training']['seed']}")

    def _create_sampler(self) -> TPESampler:
        """Create Optuna sampler with configuration."""
        return TPESampler(
            n_startup_trials=self.config['training']['n_startup_trials'],
            seed=self.config['training']['seed'],
            consider_prior=False,
            consider_magic_clip=False,
            consider_endpoints=False,
            multivariate=False
        )

    def validate_config(self) -> None:
        """Validate configuration using parameter manager."""
        self.param_manager.validate_config(self.config)

    def suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters using parameter manager."""
        return self.param_manager.get_trial_config(trial)

    def create_minimal_config(self) -> Dict[str, Any]:
        """Create minimal config for worker processes."""
        return self.param_manager.copy_base_config()

__all__ = ['StudyConfig']