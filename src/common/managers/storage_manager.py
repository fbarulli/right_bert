# src/common/managers/storage_manager.py
from __future__ import annotations
import sqlite3
import contextlib
import logging
from pathlib import Path
from filelock import FileLock
import json
from typing import Dict, Any, Optional
import optuna
from optuna.samplers import TPESampler
import shutil
import traceback

from src.common.managers.base_manager import BaseManager
from src.common.managers.directory_manager import DirectoryManager

logger = logging.getLogger(__name__)

class StorageManager(BaseManager):
    """
    Manages Optuna storage and study persistence.
    
    This manager handles:
    - Database initialization and configuration
    - Study creation and persistence
    - Trial history management
    - Directory structure for trials and profiling
    - Resource cleanup
    """

    def __init__(
        self,
        directory_manager: DirectoryManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize StorageManager.

        Args:
            directory_manager: Injected DirectoryManager instance
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self._directory_manager = directory_manager
        self._local.storage_dir = None
        self._local.storage_path = None
        self._local.lock_path = None
        self._local.history_path = None
        self._local.trials_dir = None
        self._local.profiler_dir = None

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local attributes.

        Args:
            config: Optional configuration dictionary that overrides the one from constructor
        """
        try:
            super()._initialize_process_local(config)

            if not self._directory_manager.is_initialized():
                raise RuntimeError("DirectoryManager must be initialized before StorageManager")

            # Initialize paths
            effective_config = config if config is not None else self._config
            output_config = self.get_config_section(effective_config, 'output')
            base_dir = Path(output_config['dir'])
            
            self._local.storage_dir = base_dir / output_config['storage_dir']
            self._local.storage_path = self._local.storage_dir / 'optuna.db'
            self._local.lock_path = self._local.storage_dir / 'optuna.lock'
            self._local.history_path = self._local.storage_dir / 'trial_history.json'
            self._local.trials_dir = self._local.storage_dir / 'trials'
            self._local.profiler_dir = self._local.storage_dir / 'profiler'

            # Create directories
            self._local.storage_dir.mkdir(parents=True, exist_ok=True)
            self._local.trials_dir.mkdir(exist_ok=True)
            self._local.profiler_dir.mkdir(exist_ok=True)

            # Initialize database
            self._initialize_storage()

            logger.info(
                f"StorageManager initialized for process {self._local.pid}:\n"
                f"- Storage dir: {self._local.storage_dir}\n"
                f"- Trials dir: {self._local.trials_dir}\n"
                f"- Profiler dir: {self._local.profiler_dir}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize StorageManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _initialize_storage(self) -> None:
        """Initialize SQLite storage with proper settings."""
        self.ensure_initialized()
        try:
            with contextlib.closing(sqlite3.connect(self._local.storage_path)) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA busy_timeout=10000")
                conn.commit()
            logger.debug(f"Initialized SQLite storage at {self._local.storage_path}")

        except Exception as e:
            logger.error(f"Error initializing storage: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_study(
        self,
        study_name: str,
        sampler: TPESampler
    ) -> optuna.Study:
        """
        Create or load study with proper locking.

        Args:
            study_name: Name of the study
            sampler: Optuna sampler instance

        Returns:
            optuna.Study: Created or loaded study
        """
        self.ensure_initialized()
        try:
            storage_url = self.get_storage_url()
            with FileLock(self._local.lock_path):
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage_url,
                    sampler=sampler,
                    direction='minimize',
                    load_if_exists=True
                )
            logger.info(f"Created/loaded study '{study_name}'")
            return study

        except Exception as e:
            logger.error(f"Error creating study: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def save_trial_history(self, trials: list[optuna.Trial]) -> None:
        """
        Save trial history to JSON file.

        Args:
            trials: List of completed trials
        """
        self.ensure_initialized()
        try:
            history = {'trials': []}
            for trial in trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    trial_data = {
                        'number': trial.number,
                        'params': trial.params,
                        'value': trial.value,
                        'state': trial.state.name,
                        'datetime_start': (
                            trial.datetime_start.isoformat()
                            if trial.datetime_start else None
                        ),
                        'datetime_complete': (
                            trial.datetime_complete.isoformat()
                            if trial.datetime_complete else None
                        ),
                        'fail_reason': trial.user_attrs.get('fail_reason', None)
                    }
                    history['trials'].append(trial_data)

            with open(self._local.history_path, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"Saved trial history to {self._local.history_path}")

        except Exception as e:
            logger.error(f"Error saving trial history: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_storage_url(self) -> str:
        """
        Get SQLite storage URL.

        Returns:
            str: Storage URL with timeout configuration
        """
        self.ensure_initialized()
        return f"sqlite:///{self._local.storage_path}?timeout=60"

    def get_trial_dir(self, trial_number: int) -> Path:
        """
        Get directory for a specific trial.

        Args:
            trial_number: Trial number

        Returns:
            Path: Trial directory path
        """
        self.ensure_initialized()
        trial_dir = self._local.trials_dir / f'trial_{trial_number}'
        trial_dir.mkdir(parents=True, exist_ok=True)
        return trial_dir

    def get_profiler_dir(self, trial_number: Optional[int] = None) -> Path:
        """
        Get profiler directory.

        Args:
            trial_number: Optional trial number for trial-specific profiling

        Returns:
            Path: Profiler directory path
        """
        self.ensure_initialized()
        if trial_number is not None:
            profiler_dir = self.get_trial_dir(trial_number) / 'profiler'
        else:
            profiler_dir = self._local.profiler_dir
        profiler_dir.mkdir(parents=True, exist_ok=True)
        return profiler_dir

    def cleanup_trial(self, trial_number: int) -> None:
        """
        Clean up trial directory.

        Args:
            trial_number: Trial number to clean up
        """
        self.ensure_initialized()
        try:
            trial_dir = self._local.trials_dir / f'trial_{trial_number}'
            if trial_dir.exists():
                shutil.rmtree(trial_dir)
                logger.debug(f"Cleaned up trial directory: {trial_dir}")

        except Exception as e:
            logger.error(f"Error cleaning up trial {trial_number}: {str(e)}")
            logger.error(traceback.format_exc())

    def cleanup_profiler(self, trial_number: Optional[int] = None) -> None:
        """
        Clean up profiler directory.

        Args:
            trial_number: Optional trial number for trial-specific cleanup
        """
        self.ensure_initialized()
        try:
            if trial_number is not None:
                profiler_dir = self.get_trial_dir(trial_number) / 'profiler'
            else:
                profiler_dir = self._local.profiler_dir

            if profiler_dir.exists():
                shutil.rmtree(profiler_dir)
                profiler_dir.mkdir(exist_ok=True)
                logger.debug(f"Cleaned up profiler directory: {profiler_dir}")

        except Exception as e:
            logger.error(f"Error cleaning up profiler: {str(e)}")
            logger.error(traceback.format_exc())

    def log_database_status(self) -> None:
        """Log current database status for debugging."""
        self.ensure_initialized()
        try:
            conn = sqlite3.connect(self._local.storage_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM trials")
            total_trials = cursor.fetchone()[0]

            cursor.execute(
                "SELECT datetime_complete FROM trials "
                "WHERE datetime_complete IS NOT NULL "
                "ORDER BY datetime_complete DESC LIMIT 1"
            )
            last_modified = cursor.fetchone()

            logger.info(
                f"\nOptuna Database Status:\n"
                f"- Location: {self._local.storage_path}\n"
                f"- Total trials: {total_trials}\n"
                f"- Last modified: {last_modified[0] if last_modified else 'Never'}"
            )

            conn.close()

        except Exception as e:
            logger.error(f"Error checking database status: {str(e)}")
            logger.error(traceback.format_exc())

    def cleanup(self) -> None:
        """Clean up storage manager resources."""
        try:
            # Clean up trials
            if self._local.trials_dir and self._local.trials_dir.exists():
                shutil.rmtree(self._local.trials_dir)
                self._local.trials_dir.mkdir(exist_ok=True)

            # Clean up profiler
            if self._local.profiler_dir and self._local.profiler_dir.exists():
                shutil.rmtree(self._local.profiler_dir)
                self._local.profiler_dir.mkdir(exist_ok=True)

            # Clean up database
            if self._local.storage_path and self._local.storage_path.exists():
                self._local.storage_path.unlink()
                self._initialize_storage()

            # Clean up history
            if self._local.history_path and self._local.history_path.exists():
                self._local.history_path.unlink()

            logger.info(f"Cleaned up StorageManager for process {self._local.pid}")
            super().cleanup()

        except Exception as e:
            logger.error(f"Error cleaning up StorageManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise


__all__ = ['StorageManager']
