# src/common/managers/optuna_manager.py (CORRECTED)
import logging
import os
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import optuna
from optuna.trial import TrialState
import multiprocessing as mp

from src.common.study.study_storage import StudyStorage
from src.common.study.study_config import StudyConfig
# DELAYED IMPORTS
# from src.common.process.worker_utils import run_worker
from src.common.managers.base_manager import BaseManager

logger = logging.getLogger(__name__)

class OptunaManager(BaseManager):
    """Manages optimization process using Optuna."""

    def __init__(self, study_name: str, config: Dict[str, Any], storage_dir: Optional[Path] = None):
        """
        Initialize the OptunaManager.

        Args:
            study_name: Name of the Optuna study.
            config: Configuration dictionary.
            storage_dir: Directory for Optuna storage.
        """
        # Assign attributes *before* calling super().__init__()
        self.study_name = study_name
        self.config = config
        self.storage_dir = storage_dir
        self.study = None # Initialize to None
        super().__init__(config) # Initialize base, calling _initialize

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        super()._initialize_process_local(config)
        if not self.study_name or not self.config:  # Now safe to access
            raise ValueError("OptunaManager requires study_name and config to be set.")

        self.study_config = StudyConfig(self.config)
        self.study_config.validate_config()
        self.storage = StudyStorage(self.storage_dir or Path.cwd())
        self.storage_url = self.storage.get_storage_url()

        logger.info("\n=== Creating/Loading Study ===")
        logger.info(f"Study name: {self.study_name}")
        logger.info(f"Storage URL: {self.storage_url}")
        logger.info(f"Sampler type: {type(self.study_config.sampler)}")
        logger.info(f"Sampler parameters: {self.study_config.sampler.__dict__}")

        try:
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage_url,
                sampler=self.study_config.sampler,
                direction='minimize',
                load_if_exists=True
            )
            logger.info("Study created/loaded successfully")
            logger.info(f"Study ID: {self.study._study_id}") # type: ignore
            logger.info(f"Study direction: {self.study.direction}") # type: ignore
            logger.info(f"Study system attrs: {self.study.system_attrs}") # type: ignore
        except Exception as e:
            logger.error(f"Failed to create/load study: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

        self.worker_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self._active_workers = {}
        self._local.initialized = True

        self._log_study_state()

    def _log_study_state(self) -> None:
        """Log current state of the study."""
        if self.study is not None:
            n_trials = len(self.study.trials)
            completed_trials = len([t for t in self.study.trials if t.state == TrialState.COMPLETE])
            failed_trials = len([t for t in self.study.trials if t.state == TrialState.FAIL])
            pruned_trials = len([t for t in self.study.trials if t.state == TrialState.PRUNED])

            logger.info("\n=== Study State ===")
            logger.info(f"Total trials: {n_trials}")
            logger.info(f"- Completed: {completed_trials}")
            logger.info(f"- Failed: {failed_trials}")
            logger.info(f"- Pruned: {pruned_trials}")

            if completed_trials > 0:
                best_trial = self.study.best_trial
                logger.info("\n=== Best Trial ===")
                logger.info(f"Number: {best_trial.number}")
                logger.info(f"Value: {best_trial.value:.4f}")
                logger.info(f"Duration: {best_trial.duration.total_seconds():.2f} seconds") # type: ignore
                logger.info("Parameters:")
                for k, v in best_trial.params.items():
                    logger.info(f"- {k}: {v}")

    def _start_workers(self, n_jobs: int) -> None:
        """Start worker processes."""
        logger.info(f"\n=== Starting {n_jobs} Worker Processes ===")

        for worker_id in range(n_jobs):
            logger.info(f"\nPreparing worker {worker_id}")
            from src.common.process.worker_utils import run_worker #DELAYED
            args = (
                worker_id,
                self.study_name,
                self.storage_url,
                self.worker_queue,
                self.result_queue
            )
            logger.info("Worker arguments:")
            logger.info(f"- worker_id: {type(args[0])} = {args[0]}")
            logger.info(f"- study_name: {type(args[1])} = {args[1]}")
            logger.info(f"- storage_url: {type(args[2])} = {args[2]}")
            logger.info(f"- worker_queue: {type(args[3])}")
            logger.info(f"- result_queue: {type(args[4])}")

            try:
                logger.info("Creating process...")
                process = mp.Process(
                    target=run_worker,
                    args=args,
                    daemon=True
                )

                logger.info("Starting process...")
                process.start()
                self._active_workers[worker_id] = process
                logger.info(f"Started worker {worker_id} with PID {process.pid}")

            except Exception as e:
                logger.error(f"Failed to start worker {worker_id}: {str(e)}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                raise

    def _cleanup_workers(self) -> None:
        """Clean up worker processes."""
        logger.info("\n=== Cleaning Up Worker Processes ===")
        logger.info(f"Active workers: {len(self._active_workers)}")

        logger.info("Sending exit signals to all workers...")
        for _ in range(len(self._active_workers)):
            self.worker_queue.put(None)

        for worker_id, process in self._active_workers.items():
            logger.info(f"\nWaiting for worker {worker_id} (PID: {process.pid}) to finish...")

            try:
                process.join(timeout=30)
                if process.is_alive():
                    logger.warning(f"Worker {worker_id} did not exit gracefully, terminating...")
                    process.terminate()
                    process.join(timeout=5)

                    if process.is_alive():
                        logger.error(f"Failed to terminate worker {worker_id}, killing...")
                        process.kill()
                        process.join(timeout=1)

                        if process.is_alive():
                            logger.error(f"Failed to kill worker {worker_id}")
                else:
                    logger.info(f"Worker {worker_id} exited successfully")

            except Exception as e:
                logger.error(f"Error cleaning up worker {worker_id}: {str(e)}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")

        self._active_workers.clear()
        logger.info("\nAll workers cleaned up")

    def optimize(self, config: Dict[str, Any], output_path: Path) -> Optional[optuna.trial.FrozenTrial]:
        """Run optimization with proper process isolation."""
        n_trials = config['training']['num_trials']
        n_jobs = config['training']['n_jobs']
        logger.info(f"\n=== Starting Optimization ===")
        logger.info(f"Main Process ID: {os.getpid()}")
        logger.info(f"Number of trials: {n_trials}")
        logger.info(f"Number of workers: {n_jobs}")

        try:
            logger.info("\n=== Worker Process Setup ===")
            logger.info(f"Study name: {self.study_name}")
            logger.info(f"Storage URL: {self.storage_url}")
            logger.info(f"Queue types: worker_queue={type(self.worker_queue)}, result_queue={type(self.result_queue)}")

            self._start_workers(n_jobs)
            logger.info("All workers started successfully")

            for trial_num in range(n_trials):
                logger.info(f"\n=== Creating Trial {trial_num} ===")
                try:
                    trial = self.study.ask()  # type: ignore
                    trial_config = self.study_config.suggest_parameters(trial)
                    logger.info(f"Trial created successfully:")
                    logger.info(f"- Number: {trial.number}")
                    logger.info(f"- Suggested parameters: {trial.params}")
                    logger.info(f"- User attrs: {trial.user_attrs}")
                    logger.info(f"- System attrs: {trial.system_attrs}")
                except Exception as e:
                    logger.error(f"Failed to create trial: {str(e)}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    raise

                trial_data = {
                    'trial_number': trial.number,
                    'trial_params': trial.params,
                    'config': trial_config,
                    'output_path': str(output_path)
                }

                logger.info("\n=== Trial Parameter Details ===")
                logger.info(f"Trial number: {trial.number}")
                logger.info(f"Trial parameters: {trial.params}")
                logger.info(f"Config structure:")
                for section in ['training', 'data']:
                    logger.info(f"- {section}: {trial_config[section]}")

                logger.info("\n=== Trial Data Structure ===")
                logger.info(f"Trial number: {trial.number}")
                logger.info(f"Config sections: {trial_config.keys()}")

                logger.info("\n=== Trial Data Types ===")
                for key, value in trial_data.items():
                    logger.info(f"{key}: {type(value)}")
                    if isinstance(value, dict):
                        logger.info(f"{key} keys: {value.keys()}")

                self.worker_queue.put(trial_data)
                logger.info(f"Successfully queued trial {trial.number}")

            completed_trials = 0
            while completed_trials < n_trials:
                logger.info(f"\n=== Waiting for Trial Result {completed_trials + 1}/{n_trials} ===")
                trial_num, result, error = self.result_queue.get()
                logger.info(f"Received result for trial {trial_num}")
                logger.info(f"Result type: {type(result)}")
                logger.info(f"Error: {error}")

                if error:
                    logger.error(f"Trial {trial_num} failed: {error}")
                    logger.info("Setting trial state to FAIL")
                    self.study.tell(trial_num, state=optuna.trial.TrialState.FAIL) # type: ignore
                else:
                    logger.info(f"Trial {trial_num} completed with value: {result}")
                    logger.info("Telling study about trial result")
                    self.study.tell(trial_num, result) # type: ignore
                    logger.info("Study updated successfully")

                completed_trials += 1
                logger.info(f"Completed {completed_trials}/{n_trials} trials")

            self.storage.save_trial_history(self.study.trials) # type: ignore

            if any(t.state == TrialState.COMPLETE for t in self.study.trials): # type: ignore
                return self.study.best_trial # type: ignore
            else:
                logger.warning("No trials completed successfully.")
                return None

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise
        finally:
            self._cleanup_workers()
            self.storage.log_database_status()
            logger.info("Optimization completed")
__all__ = ['OptunaManager']