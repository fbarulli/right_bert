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

from .base_manager import BaseManager
logger = logging.getLogger(__name__)

class StorageManager(BaseManager):
    """Manages Optuna storage and study persistence."""

    def __init__(self, storage_dir: Path):
        super().__init__()
        self.storage_dir = storage_dir

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes."""
        if not hasattr(self, 'storage_dir') or not isinstance(self.storage_dir, Path):
            raise ValueError("StorageManager must be initialized with a storage_dir Path object.")

        self.storage_path = self.storage_dir / 'optuna.db'
        self.lock_path = self.storage_dir / 'optuna.lock'
        self.history_path = self.storage_dir / 'trial_history.json'
        self.trials_dir = self.storage_dir / "trials"
        self.profiler_dir = self.storage_dir / 'profiler'
        # Create directories
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.trials_dir.mkdir(exist_ok=True)
        self.profiler_dir.mkdir(exist_ok=True)

        self._initialize_storage() # Initialize DB

        logger.info(
            f"Initialized storage manager:\n"
            f"- Storage dir: {self.storage_dir}\n"
            f"- Trials dir: {self.trials_dir}\n"
            f"- Profiler dir: {self.profiler_dir}"
        )

    def _initialize_storage(self):
        """Initialize SQLite storage with proper settings."""
        with contextlib.closing(sqlite3.connect(self.storage_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=10000")
            conn.commit()

    def create_study(self, study_name: str, sampler: TPESampler) -> optuna.Study:
        """Create or load study with proper locking."""
        storage_url = f"sqlite:///{self.storage_path}?timeout=60"

        with FileLock(self.lock_path):
            return optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                sampler=sampler,
                direction='minimize',
                load_if_exists=True
            )

    def save_trial_history(self, study: optuna.Study):
        """Save trial history to JSON file."""
        history = {'trials': []}

        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_data = {
                    'number': trial.number,
                    'params': trial.params,
                    'value': trial.value,
                    'state': trial.state.name,
                    'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                    'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                    'fail_reason': trial.user_attrs.get('fail_reason', None),  # Store fail reason
                }
                history['trials'].append(trial_data)

        with open(self.history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"\nSaved trial history to {self.history_path}")
    
    def get_storage_url(self) -> str:
        return f"sqlite:///{self.storage_path}?timeout=60"

    def get_trial_dir(self, trial_number: int) -> Path:
        trial_dir = self.trials_dir / f'trial_{trial_number}'
        trial_dir.mkdir(parents=True, exist_ok=True)
        return trial_dir

    def get_profiler_dir(self, trial_number: Optional[int] = None) -> Path:
        if trial_number is not None:
            profiler_dir = self.get_trial_dir(trial_number) / 'profiler'
        else:
            profiler_dir = self.profiler_dir

        profiler_dir.mkdir(parents=True, exist_ok=True)
        return profiler_dir

    def cleanup_trial(self, trial_number: int):
        try:
            trial_dir = self.trials_dir / f'trial_{trial_number}'
            if trial_dir.exists():
                shutil.rmtree(trial_dir)
                logger.debug(f"Cleaned up trial directory: {trial_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up trial {trial_number}: {e}")

    def cleanup_profiler(self, trial_number: Optional[int] = None):
        try:
            if trial_number is not None:
                profiler_dir = self.get_trial_dir(trial_number) / 'profiler'
            else:
                profiler_dir = self.profiler_dir

            if profiler_dir.exists():
                shutil.rmtree(profiler_dir)
                profiler_dir.mkdir(exist_ok=True)
                logger.debug(f"Cleaned up profiler directory: {profiler_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up profiler: {e}")

    def cleanup_all(self):
        try:
            # Clean up trials
            if self.trials_dir.exists():
                shutil.rmtree(self.trials_dir)
                self.trials_dir.mkdir(exist_ok=True)

            # Clean up profiler
            if self.profiler_dir.exists():
                shutil.rmtree(self.profiler_dir)
                self.profiler_dir.mkdir(exist_ok=True)

            # Clean up database
            if self.storage_path.exists():
                self.storage_path.unlink()
                self._initialize_storage()

            # Clean up history
            if self.history_path.exists():
                self.history_path.unlink()

            logger.info("Cleaned up all storage")

        except Exception as e:
            logger.error(f"Error cleaning up storage: {e}")
            raise

    def log_database_status(self) -> None:
        """Log current database status (for debugging)."""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM trials")
            total_trials = cursor.fetchone()[0]

            cursor.execute("SELECT datetime_complete FROM trials WHERE datetime_complete IS NOT NULL ORDER BY datetime_complete DESC LIMIT 1")
            last_modified = cursor.fetchone()

            logger.info(f"\nOptuna Database Status:")
            logger.info(f"Location: {self.storage_path}")
            logger.info(f"Total trials in DB: {total_trials}")
            if last_modified:
                logger.info(f"Last modified: {last_modified[0]}")

            conn.close()
        except Exception as e:
            logger.error(f"Error checking database status: {e}")

__all__ = ['StorageManager']