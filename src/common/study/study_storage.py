from __future__ import annotations
import os
import logging
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from src.common.study.data_types import TrialData

logger = logging.getLogger(__name__)

class StudyStorage:
    """Storage for Optuna studies."""
    
    def __init__(self, base_dir: str):
        """Initialize the study storage.
        
        Args:
            base_dir: Base directory for storing study data
        """
        # Convert string path to Path object or use os.path.join
        self.storage_path = os.path.join(base_dir, 'storage', 'optuna.db')
        
        # Create directory if it doesn't exist
        storage_dir = os.path.dirname(self.storage_path)
        os.makedirs(storage_dir, exist_ok=True)
        
        # Path for trial history
        base_path = Path(base_dir)
        self.history_path = base_path / 'trial_history.json'
        
        logger.info(f"Initialized study storage at {self.storage_path}")
        self._init_database_schema()
        
    def _init_database_schema(self):
        """Initialize the database schema if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS studies (
                    study_id INTEGER PRIMARY KEY,
                    study_name TEXT UNIQUE,
                    direction TEXT,
                    system_attrs TEXT,
                    user_attrs TEXT,
                    datetime_start DATETIME,
                    datetime_complete DATETIME
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id INTEGER PRIMARY KEY,
                    study_id INTEGER,
                    number INTEGER UNIQUE,
                    state TEXT,
                    value REAL,
                    datetime_start DATETIME,
                    datetime_complete DATETIME,
                    params TEXT,
                    user_attrs TEXT,
                    system_attrs TEXT,
                    FOREIGN KEY (study_id) REFERENCES studies(study_id)
                )
            """)

            conn.commit()
            conn.close()
            logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}")
            raise
            
    def get_storage_url(self) -> str:
        """Get SQLite URL for Optuna storage."""
        return f"sqlite:///{self.storage_path}?timeout=60"
    
    def save_trial_history(self, trials: list) -> None:
        """Save trial history to JSON file."""
        try:
            history = {'trials': []}
            for trial in trials:
                if trial.state.name == 'COMPLETE':
                    trial_data = {
                        'number': trial.number,
                        'params': trial.params,
                        'value': trial.value,
                        'state': trial.state.name,
                        'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                        'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                        'fail_reason': trial.user_attrs.get('fail_reason', None),
                    }
                    history['trials'].append(trial_data)

            with open(self.history_path, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"Saved trial history to {self.history_path}")
        except Exception as e:
            logger.error(f"Error saving trial history: {str(e)}")
            
    def log_database_status(self) -> None:
        """Log current database status (for debugging)."""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM trials")
            total_trials = cursor.fetchone()[0]
            
            cursor.execute("SELECT datetime_complete FROM trials WHERE datetime_complete IS NOT NULL ORDER BY datetime_complete DESC LIMIT 1")
            last_modified = cursor.fetchone()
            
            logger.info(f"Optuna Database Status:")
            logger.info(f"Location: {self.storage_path}")
            logger.info(f"Total trials in DB: {total_trials}")
            if last_modified:
                logger.info(f"Last modified: {last_modified[0]}")

            conn.close()
        except Exception as e:
            logger.error(f"Error checking database status: {e}")

__all__ = ['StudyStorage']