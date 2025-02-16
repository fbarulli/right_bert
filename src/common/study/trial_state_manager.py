# src/common/study/trial_state_manager.py
from __future__ import annotations
import logging
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
import optuna

logger = logging.getLogger(__name__)

class TrialStatus(Enum):
    INITIALIZING = "initializing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"

class TrialStateManager:
    """Manages the state of a single Optuna trial."""

    def __init__(
        self,
        trial: optuna.Trial,
        config: Dict[str, Any]
    ):
        self.trial = trial
        self.trial_number = trial.number
        self.max_memory_gb = config['resources']['max_memory_gb']
        self.status = TrialStatus.INITIALIZING
        self.start_time = datetime.now()

        # Set initial state attributes on the trial
        self.trial.set_user_attr('current_status', self.status.value)
        self.trial.set_user_attr('start_time', self.start_time.isoformat())
        self.trial.set_user_attr('completed', False)

    def update_state(self, new_status: TrialStatus, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the trial's state and log metrics.

        Args:
            new_status: The new status of the trial (from the TrialStatus enum).
            metrics: Optional dictionary of metrics to log.
        """
        old_status = self.status
        self.status = new_status

        # Update the trial's user attributes (Optuna's way to store custom info)
        self.trial.set_user_attr('current_status', new_status.value)
        if metrics:
            for key, value in metrics.items():
                self.trial.set_user_attr(key, value)

        # Handle completion/failure
        if new_status == TrialStatus.COMPLETED:
            self.trial.set_user_attr('completed', True)
            self.trial.set_user_attr('end_time', datetime.now().isoformat())
            duration = (datetime.now() - self.start_time).total_seconds()
            self.trial.set_user_attr('duration_seconds', duration)

        elif new_status == TrialStatus.FAILED:
            self.trial.set_user_attr('completed', False)  # Explicitly set
            self.trial.set_user_attr('end_time', datetime.now().isoformat())
            duration = (datetime.now() - self.start_time).total_seconds()
            self.trial.set_user_attr('duration_seconds', duration)
__all__ = ['TrialStateManager', 'TrialStatus']