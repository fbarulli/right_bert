
# src/common/managers/wandb_manager.py
from __future__ import annotations
import logging
import time
import json
import threading
import gc
import torch
import os
import traceback
from typing import Dict, Any, Optional, List

from src.common.managers.base_manager import BaseManager

logger = logging.getLogger(__name__)

# Configure Weights & Biases to use thread-based initialization
os.environ["WANDB_START_METHOD"] = "thread"

# Import Weights & Biases with error handling
try:
    import wandb
    WANDB_AVAILABLE = True
    logger.debug("Module level wandb import successful")
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    logger.debug("Module level wandb import failed")

class WandbManager(BaseManager):
    """
    Manages Weights & Biases logging and experiment tracking.

    This manager handles:
    - W&B initialization and authentication
    - Run management and cleanup
    - Metric logging and progress tracking
    - Trial tracking for optimization
    """

    def __init__(
        self,
        config: Dict[str, Any],
        study_name: str
    ):
        """
        Initialize WandbManager.

        Args:
            config: Configuration dictionary
            study_name: Name of the study for grouping runs
        """
        super().__init__(config)
        self._study_name = study_name
        self._local.start_time = None
        self._local.run = None
        self._local.enabled = False

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local attributes.

        Args:
            config: Optional configuration dictionary that overrides the one from constructor
        """
        try:
            super()._initialize_process_local(config)

            effective_config = config if config is not None else self._config
            wandb_config = self.get_config_section(effective_config, 'output')['wandb']

            # Check if W&B is available and enabled
            self._local.enabled = WANDB_AVAILABLE and wandb_config.get('enabled', False)

            if self._local.enabled:
                api_key = wandb_config.get('api_key')
                if not api_key:
                    logger.warning("No wandb API key provided, disabling wandb logging")
                    self._local.enabled = False
                else:
                    try:
                        wandb.login(key=api_key)
                        logger.info("Successfully logged in to wandb")
                    except Exception as e:
                        logger.error(f"Failed to login to wandb: {str(e)}")
                        logger.error(traceback.format_exc())
                        self._local.enabled = False

            self._log_process_info()

        except Exception as e:
            logger.error(f"Failed to initialize WandbManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _log_process_info(self) -> None:
        """Log current process and W&B state information."""
        logger.debug(
            f"\nProcess Info:\n"
            f"- PID: {self._local.pid}\n"
            f"- PPID: {os.getppid()}\n"
            f"- Thread: {threading.current_thread().name}\n"
            f"\nWandb State:\n"
            f"- WANDB_AVAILABLE: {WANDB_AVAILABLE}\n"
            f"- enabled: {self._local.enabled}\n"
            f"- current run: {self._local.run.id if self._local.run else None}\n"
            f"\nEnvironment Variables:\n"
            f"{json.dumps({k: v for k, v in os.environ.items() if 'WANDB' in k}, indent=2)}"
        )

    def cleanup_run(self) -> None:
        """Clean up current W&B run."""
        logger.debug("\nCleaning up W&B run")
        self._log_process_info()

        if self._local.enabled and self._local.run:
            try:
                self._local.run.finish()
                logger.debug("Run finished successfully")
            except Exception as e:
                logger.warning(f"Error finishing run: {str(e)}")
                logger.error(traceback.format_exc())
            finally:
                self._local.run = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def init_optimization(self) -> None:
        """Initialize W&B run for optimization."""
        logger.debug("\nInitializing W&B for optimization")
        self._log_process_info()

        if not self._local.enabled:
            logger.debug("Wandb disabled, skipping initialization")
            return

        try:
            self.cleanup_run()

            wandb_config = self.get_config_section(self._config, 'output')['wandb']
            group_name = f"optimization_{self._study_name}"

            self._local.run = wandb.init(
                project=wandb_config.get('project', self._study_name),
                group=group_name,
                name="optimization_main",
                job_type="optimization",
                tags=wandb_config.get('tags', []),
                config=self._config,
                reinit=True
            )
            self._local.start_time = time.time()
            logger.info("Initialized wandb run for optimization")

        except Exception as e:
            logger.error(f"Failed to initialize wandb: {str(e)}")
            logger.error(traceback.format_exc())
            self.cleanup_run()

    def init_trial(self, trial_number: int) -> None:
        """
        Initialize W&B run for a trial.

        Args:
            trial_number: Trial number to track
        """
        logger.debug(f"\nInitializing W&B for trial {trial_number}")
        self._log_process_info()

        if not self._local.enabled:
            return

        try:
            self.cleanup_run()

            wandb_config = self.get_config_section(self._config, 'output')['wandb']
            group_name = f"optimization_{self._study_name}"

            self._local.run = wandb.init(
                project=wandb_config.get('project', self._study_name),
                group=group_name,
                name=f"trial_{trial_number}",
                job_type="trial",
                tags=wandb_config.get('tags', []),
                config=self._config,
                reinit=True
            )

            self._local.run.log({
                'trial_number': trial_number,
                'trial_status': 'started',
                'process_id': self._local.pid
            }, step=0)

            logger.info(f"Successfully initialized trial {trial_number}")

        except Exception as e:
            logger.error(f"Failed to initialize trial: {str(e)}")
            logger.error(traceback.format_exc())
            self.cleanup_run()

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None
    ) -> None:
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        if not self._local.enabled or self._local.run is None:
            return

        if step is None:
            logger.warning("Step must be provided for logging metrics")
            return

        try:
            self._local.run.log(metrics, step=step)
            logger.debug(f"Logged metrics at step {step}")

        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")
            logger.error(traceback.format_exc())

    def finish_trial(self, trial_number: int) -> None:
        """
        Finish W&B run for a trial.

        Args:
            trial_number: Trial number that completed
        """
        logger.debug(f"\nFinishing W&B for trial {trial_number}")
        self._log_process_info()

        if not self._local.enabled or self._local.run is None:
            return

        try:
            self._local.run.log({
                'trial_number': trial_number,
                'trial_status': 'completed',
                'process_id': self._local.pid
            }, step=0)
            logger.info(f"Successfully logged trial completion")
            self.cleanup_run()

        except Exception as e:
            logger.error(f"Failed to log trial completion: {str(e)}")
            logger.error(traceback.format_exc())

    def finish(self) -> None:
        """Finish W&B logging."""
        logger.debug("\nFinishing W&B logging")
        self._log_process_info()

        if not self._local.enabled:
            return

        try:
            if self._local.pid == os.getppid():
                logger.debug("Main process finishing run")
                self.cleanup_run()
                logger.info("Finished wandb optimization run")
            else:
                logger.debug("Child process skipping run finish")

        except Exception as e:
            logger.error(f"Error during finish: {str(e)}")
            logger.error(traceback.format_exc())

    def flush(self) -> None:
        """Flush W&B logs."""
        logger.debug("\nFlushing W&B logs")
        self._log_process_info()

        if not self._local.enabled or self._local.run is None:
            return

        try:
            self._local.run.flush()
            logger.debug("Logs flushed successfully")

        except Exception as e:
            logger.error(f"Failed to flush logs: {str(e)}")
            logger.error(traceback.format_exc())

    def init_final_training(self) -> None:
        """Initialize W&B run for final training."""
        logger.debug("\nInitializing W&B for final training")
        self._log_process_info()

        if not self._local.enabled:
            return

        try:
            self.cleanup_run()

            wandb_config = self.get_config_section(self._config, 'output')['wandb']
            self._local.run = wandb.init(
                project=wandb_config.get('project', self._study_name),
                name="final_training",
                job_type="final_training",
                tags=wandb_config.get('tags', []) + ["final"],
                config=self._config,
                reinit=True
            )
            self._local.start_time = time.time()
            logger.info("Initialized wandb for final training")

        except Exception as e:
            logger.error(f"Failed to initialize final training: {str(e)}")
            logger.error(traceback.format_exc())
            self.cleanup_run()

    def log_progress(
        self,
        current: int,
        total: int,
        prefix: str = '',
        step: Optional[int] = None
    ) -> None:
        """
        Log progress metrics to W&B.

        Args:
            current: Current progress value
            total: Total progress value
            prefix: Optional prefix for metric names
            step: Optional step number
        """
        if not self._local.enabled or self._local.run is None:
            return

        try:
            elapsed = time.time() - self._local.start_time if self._local.start_time else 0
            progress = current / total if total > 0 else 0
            remaining = (elapsed / progress) * (1 - progress) if progress > 0 else 0

            metrics = {
                f'{prefix}progress': progress * 100,
                f'{prefix}current': current,
                f'{prefix}total': total,
                f'{prefix}elapsed_time': elapsed,
                f'{prefix}remaining_time': remaining,
                f'{prefix}speed': current / elapsed if elapsed > 0 else 0
            }

            if step is not None:
                self._local.run.log(metrics, step=step)
            else:
                self._local.run.log(metrics)

            logger.debug(f"Logged progress metrics: {current}/{total}")

        except Exception as e:
            logger.error(f"Failed to log progress: {str(e)}")
            logger.error(traceback.format_exc())

    def cleanup(self) -> None:
        """Clean up wandb manager resources."""
        try:
            self.cleanup_run()
            self._local.start_time = None
            self._local.enabled = False
            logger.info(f"Cleaned up WandbManager for process {self._local.pid}")
            super().cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up WandbManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise


__all__ = ['WandbManager']