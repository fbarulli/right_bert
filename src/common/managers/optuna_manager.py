# src/common/managers/optuna_manager.py
from __future__ import annotations
import logging
import os
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import optuna
from optuna.trial import TrialState
import multiprocessing as mp

from src.common.managers.base_manager import BaseManager
from src.common.managers.storage_manager import StorageManager
from src.common.study.study_storage import StudyStorage
from src.common.study.study_config import StudyConfig
from dependency_injector.providers import Singleton

logger = logging.getLogger(__name__)

class OptunaManager(BaseManager):
    """
    Manages optimization process using Optuna.

    This manager handles:
    - Study creation and configuration
    - Worker process management
    - Trial execution and monitoring
    - Result collection and storage
    """

    def __init__(self, config, storage_manager, parameter_manager):
        """Initialize OptunaManager with dependencies."""
        self._storage_manager = storage_manager
        self._parameter_manager = parameter_manager
        self._local = threading.local()
        self._setup_process_local(config)  # Call setup before super()
        super().__init__(config)
        self._initialize_process_local(config)  # Ensure initialization
    
    def _setup_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Setup process-local Optuna attributes."""
        if not hasattr(self, '_local'):
            self._local = threading.local()
        self._local.study = None
        self._local.study_config = None
        self._local.storage = None
        self._local.storage_url = None
        self._local.worker_queue = None
        self._local.result_queue = None
        self._local.active_workers = {}
        self._local.initialized = False
        self._local.pid = os.getpid()

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes."""
        try:
            super()._initialize_process_local(config)

            # Add validation with better error messages
            if not hasattr(self, '_storage_manager') or self._storage_manager is None:
                raise RuntimeError("StorageManager not provided to OptunaManager")
            if not hasattr(self, '_parameter_manager') or self._parameter_manager is None:
                raise RuntimeError("ParameterManager not provided to OptunaManager")
                
            # Validate initialization
            if isinstance(self._storage_manager, Singleton):
                if not self._storage_manager().is_initialized():
                    raise RuntimeError("Storage manager failed to initialize")
            else:
                if not self._storage_manager.is_initialized():
                    raise RuntimeError("Storage manager failed to initialize")
            if not self._parameter_manager.is_initialized():
                self._parameter_manager._initialize_process_local(config)
            
            # Initialize study configuration and storage
            effective_config = config if config is not None else self._config
            self._local.study_config = StudyConfig(effective_config)
            self._local.study_config.validate_config()
            self._local.storage = StudyStorage(self._storage_manager.storage_dir)
            self._local.storage_url = self._local.storage.get_storage_url()

            # Use instance variable directly to avoid circular dependency
            study_name = effective_config['training']['study_name']

            logger.info(
                f"\nInitializing OptunaManager for process {self._local.pid}:\n"
                f"- Study name: {study_name}\n"
                f"- Storage URL: {self._local.storage_url}\n"
                f"- Sampler type: {type(self._local.study_config.sampler)}\n"
                f"- Sampler parameters: {self._local.study_config.sampler.__dict__}"
            )

            # Create or load study
            self._local.study = optuna.create_study(
                study_name=study_name,
                storage=self._local.storage_url,
                sampler=self._local.study_config.sampler,
                direction='minimize',
                load_if_exists=True
            )

            # Initialize queues for worker communication
            self._local.worker_queue = mp.Queue()
            self._local.result_queue = mp.Queue()
            self._local.active_workers = {}

            self._log_study_state()

            # Set initialized flag to True
            self._local.initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize OptunaManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    @property
    def study_name(self) -> str:
        """Get the study name from config."""
        self.ensure_initialized()
        return self._config['training']['study_name']

    @property
    def study(self) -> optuna.Study:
        """Get the Optuna study object."""
        self.ensure_initialized()
        return self._local.study

    def _log_study_state(self) -> None:
        """Log current state of the study."""
        if self._local.study is not None:
            n_trials = len(self._local.study.trials)
            completed_trials = len([
                t for t in self._local.study.trials
                if t.state == TrialState.COMPLETE
            ])
            failed_trials = len([
                t for t in self._local.study.trials
                if t.state == TrialState.FAIL
            ])
            pruned_trials = len([
                t for t in self._local.study.trials
                if t.state == TrialState.PRUNED
            ])

            logger.info(
                f"\nStudy State:\n"
                f"- Total trials: {n_trials}\n"
                f"- Completed: {completed_trials}\n"
                f"- Failed: {failed_trials}\n"
                f"- Pruned: {pruned_trials}"
            )

            if completed_trials > 0:
                best_trial = self._local.study.best_trial
                logger.info(
                    f"\nBest Trial:\n"
                    f"- Number: {best_trial.number}\n"
                    f"- Value: {best_trial.value:.4f}\n"
                    f"- Duration: {best_trial.duration.total_seconds():.2f} seconds\n"
                    f"- Parameters:\n"
                    + "\n".join(f"  {k}: {v}" for k, v in best_trial.params.items())
                )

    def _start_workers(self, n_jobs: int) -> None:
        """
        Start worker processes.

        Args:
            n_jobs: Number of worker processes to start
        """
        self.ensure_initialized()
        logger.info(f"\nStarting {n_jobs} worker processes")

        for worker_id in range(n_jobs):
            try:
                from src.common.process.worker_utils import run_worker
                args = (
                    worker_id,
                    self.study_name,
                    self._local.storage_url,
                    self._local.worker_queue,
                    self._local.result_queue
                )

                process = mp.Process(
                    target=run_worker,
                    args=args,
                    daemon=True
                )
                process.start()
                self._local.active_workers[worker_id] = process

                logger.info(f"Started worker {worker_id} with PID {process.pid}")

            except Exception as e:
                logger.error(f"Failed to start worker {worker_id}: {str(e)}")
                logger.error(traceback.format_exc())
                raise

    def _cleanup_workers(self) -> None:
        """Clean up worker processes."""
        self.ensure_initialized()
        logger.info(f"\nCleaning up {len(self._local.active_workers)} worker processes")

        # Send exit signals
        for _ in range(len(self._local.active_workers)):
            self._local.worker_queue.put(None)

        # Wait for workers to finish
        for worker_id, process in self._local.active_workers.items():
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
                logger.error(traceback.format_exc())

        self._local.active_workers.clear()
        logger.info("All workers cleaned up")

    def optimize(
        self,
        config: Dict[str, Any],
        output_path: Path
    ) -> Optional[optuna.trial.FrozenTrial]:
        """
        Run optimization with proper process isolation.

        Args:
            config: Configuration dictionary
            output_path: Path for output files

        Returns:
            Optional[optuna.trial.FrozenTrial]: Best trial if any completed successfully
        """
        self.ensure_initialized()

        n_trials = config['training']['num_trials']
        n_jobs = config['training']['n_jobs']

        try:
            # Start workers
            self._start_workers(n_jobs)

            # Run trials
            for trial_num in range(n_trials):
                try:
                    # Create trial
                    trial = self._local.study.ask()
                    trial_config = self._local.study_config.suggest_parameters(trial)

                    # Prepare trial data
                    trial_data = {
                        'trial_number': trial.number,
                        'trial_params': trial.params,
                        'config': trial_config,
                        'output_path': str(output_path)
                    }

                    # Queue trial
                    self._local.worker_queue.put(trial_data)
                    logger.info(f"Queued trial {trial.number}")

                except Exception as e:
                    logger.error(f"Failed to create trial {trial_num}: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise

            # Collect results
            completed_trials = 0
            while completed_trials < n_trials:
                trial_num, result, error = self._local.result_queue.get()

                if error:
                    logger.error(f"Trial {trial_num} failed: {error}")
                    self._local.study.tell(trial_num, state=TrialState.FAIL)
                else:
                    logger.info(f"Trial {trial_num} completed with value: {result}")
                    self._local.study.tell(trial_num, result)

                completed_trials += 1
                logger.info(f"Completed {completed_trials}/{n_trials} trials")

            # Save results
            self._local.storage.save_trial_history(self._local.study.trials)

            # Return best trial if any completed
            if any(t.state == TrialState.COMPLETE for t in self._local.study.trials):
                return self._local.study.best_trial
            else:
                logger.warning("No trials completed successfully")
                return None

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        finally:
            self._cleanup_workers()
            self._local.storage.log_database_status()
            logger.info("Optimization completed")

    def cleanup(self) -> None:
        """Clean up OptunaManager resources."""
        try:
            if hasattr(self, '_local'):
                if hasattr(self._local, 'active_workers'):
                    self._cleanup_workers()
                self._local.study = None
                self._local.worker_queue = None
                self._local.result_queue = None
                logger.info(f"Cleaned up OptunaManager for process {self._local.pid}")
            super().cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up OptunaManager: {str(e)}")
            raise


__all__ = ['OptunaManager']