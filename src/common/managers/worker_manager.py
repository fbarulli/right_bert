# src/common/managers/worker_manager.py
from __future__ import annotations
import logging
import os
import gc
import pickle
import traceback
import threading
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import optuna
from optuna.trial import Trial

from src.common.managers.base_manager import BaseManager
from src.common.managers.cuda_manager import CUDAManager
from src.common.managers.model_manager import ModelManager
from src.common.managers.tokenizer_manager import TokenizerManager
from src.common.study.objective_factory import ObjectiveFactory
from src.common.process.multiprocessing_setup import verify_spawn_method
from src.common.resource.resource_initializer import ResourceInitializer

logger = logging.getLogger(__name__)

class WorkerManager(BaseManager):
    """
    Manages worker processes for parallel optimization.
    
    This manager handles:
    - Worker process creation and management
    - Resource monitoring and health checks
    - Trial queuing and execution
    - Process cleanup and scaling
    """

    def __init__(
        self,
        cuda_manager: CUDAManager,
        model_manager: ModelManager,
        tokenizer_manager: TokenizerManager,
        config: Dict[str, Any],
        study_name: str,
        storage_url: str,
        n_jobs: int = 2,
        max_workers: int = 32,
        health_check_interval: int = 60
    ):
        """
        Initialize WorkerManager.

        Args:
            cuda_manager: Injected CUDAManager instance
            model_manager: Injected ModelManager instance
            tokenizer_manager: Injected TokenizerManager instance
            config: Configuration dictionary
            study_name: Name of the study
            storage_url: URL for Optuna storage
            n_jobs: Number of worker processes to start
            max_workers: Maximum number of workers allowed
            health_check_interval: Interval for health checks in seconds
        """
        super().__init__(config)
        self._cuda_manager = cuda_manager
        self._model_manager = model_manager
        self._tokenizer_manager = tokenizer_manager
        self._study_name = study_name
        self._storage_url = storage_url
        self._n_jobs = min(n_jobs, max_workers)
        self._max_workers = max_workers
        self._health_check_interval = health_check_interval
        self._last_health_check = time.time()
        self._health_check_thread = None

        # Resource limits from config
        self._resource_limits = {
            'memory_gb': config['resources']['max_memory_gb'],
            'gpu_memory_gb': config['resources']['gpu_memory_gb'],
            'max_cpu_percent': 80.0
        }

    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize process-local attributes.

        Args:
            config: Optional configuration dictionary that overrides the one from constructor
        """
        try:
            super()._initialize_process_local(config)

            if not self._cuda_manager.is_initialized():
                raise RuntimeError("CUDAManager must be initialized before WorkerManager")
            if not self._model_manager.is_initialized():
                raise RuntimeError("ModelManager must be initialized before WorkerManager")
            if not self._tokenizer_manager.is_initialized():
                raise RuntimeError("TokenizerManager must be initialized before WorkerManager")

            # Initialize queues and worker tracking
            self._local.worker_queue = mp.Queue()
            self._local.result_queue = mp.Queue()
            self._local.active_workers = {}
            self._local.worker_groups = {}

            # Start health check thread
            self._start_health_check_thread()

            logger.info(f"WorkerManager initialized for process {self._local.pid}")

        except Exception as e:
            logger.error(f"Failed to initialize WorkerManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _monitor_resources(self) -> Dict[str, float]:
        """
        Monitor system resources.

        Returns:
            Dict[str, float]: Dictionary of resource usage metrics
        """
        try:
            import psutil

            cpu_percent = psutil.cpu_percent()
            memory_gb = psutil.Process().memory_info().rss / (1024 ** 3)
            gpu_memory_gb = (
                self._cuda_manager.get_memory_allocated() / (1024 ** 3)
                if self._cuda_manager.is_available() else 0
            )

            return {
                'cpu_percent': cpu_percent,
                'memory_gb': memory_gb,
                'gpu_memory_gb': gpu_memory_gb
            }

        except ImportError:
            logger.warning("psutil not installed - resource monitoring disabled")
            return {}
        except Exception as e:
            logger.error(f"Error monitoring resources: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _check_resource_limits(self, resources: Dict[str, float]) -> bool:
        """
        Check if resource usage is within limits.

        Args:
            resources: Dictionary of current resource usage

        Returns:
            bool: True if within limits, False otherwise
        """
        if not resources:
            return True

        return (
            resources.get('memory_gb', 0) < self._resource_limits['memory_gb'] and
            resources.get('gpu_memory_gb', 0) < self._resource_limits['gpu_memory_gb'] and
            resources.get('cpu_percent', 0) < self._resource_limits['max_cpu_percent']
        )

    def _check_worker_health(self) -> None:
        """Check health of workers and restart if needed."""
        try:
            for worker_id, process in list(self._local.active_workers.items()):
                if not process.is_alive():
                    logger.warning(f"Worker {worker_id} died - restarting")

                    # Find worker group
                    group = next(
                        g for g, workers in self._local.worker_groups.items()
                        if worker_id in workers
                    )

                    # Remove dead worker
                    self._local.active_workers.pop(worker_id)
                    self._local.worker_groups[group].pop(worker_id)

                    # Restart worker
                    self._n_jobs = 1
                    self.start_workers(group)

        except Exception as e:
            logger.error(f"Error checking worker health: {str(e)}")
            logger.error(traceback.format_exc())

    def _start_health_check_thread(self) -> None:
        """Start background thread for periodic health checks."""
        def health_check_loop():
            while True:
                try:
                    self._check_worker_health()
                    resources = self._monitor_resources()
                    if not self._check_resource_limits(resources):
                        logger.warning(
                            f"System resources exceeding limits:\n"
                            f"CPU: {resources.get('cpu_percent', 0)}%\n"
                            f"Memory: {resources.get('memory_gb', 0):.1f}GB\n"
                            f"GPU Memory: {resources.get('gpu_memory_gb', 0):.1f}GB"
                        )
                except Exception as e:
                    logger.error(f"Health check failed: {str(e)}")
                    logger.error(traceback.format_exc())
                time.sleep(self._health_check_interval)

        self._health_check_thread = threading.Thread(
            target=health_check_loop,
            daemon=True,
            name="HealthCheckThread"
        )
        self._health_check_thread.start()
        logger.info("Started health check thread")

    def start_workers(self, group: str = "default", check_resources: bool = True) -> None:
        """
        Start worker processes for parallel optimization.

        Args:
            group: Worker group name for organization
            check_resources: Whether to check system resources before starting

        Raises:
            RuntimeError: If insufficient resources or max workers exceeded
        """
        self.ensure_initialized()

        try:
            # Check resources if requested
            if check_resources:
                resources = self._monitor_resources()
                if not self._check_resource_limits(resources):
                    raise RuntimeError(
                        f"Insufficient resources to start workers:\n"
                        f"CPU: {resources.get('cpu_percent', 0)}%\n"
                        f"Memory: {resources.get('memory_gb', 0):.1f}GB\n"
                        f"GPU Memory: {resources.get('gpu_memory_gb', 0):.1f}GB"
                    )

            # Check worker limit
            if len(self._local.active_workers) + self._n_jobs > self._max_workers:
                raise RuntimeError(
                    f"Cannot start {self._n_jobs} workers - would exceed max_workers "
                    f"({self._max_workers})\n"
                    f"Active workers: {len(self._local.active_workers)}"
                )

            logger.info(f"Starting {self._n_jobs} worker processes in group '{group}'")

            # Initialize group if needed
            if group not in self._local.worker_groups:
                self._local.worker_groups[group] = {}

            # Calculate starting worker ID
            start_id = (
                max(self._local.active_workers.keys()) + 1
                if self._local.active_workers else 0
            )

            # Start workers
            for i in range(self._n_jobs):
                worker_id = start_id + i
                process = mp.Process(
                    target=self._worker_process,
                    args=(worker_id, group),
                    daemon=True
                )
                process.start()

                self._local.active_workers[worker_id] = process
                self._local.worker_groups[group][worker_id] = process

                logger.info(
                    f"Started worker {worker_id} in group '{group}' "
                    f"with PID {process.pid}"
                )

        except Exception as e:
            logger.error(f"Error starting workers: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _worker_process(self, worker_id: int, group: str) -> None:
        """
        Worker process implementation.

        Args:
            worker_id: Worker process ID
            group: Worker group name
        """
        current_pid = os.getpid()
        logger.info(
            f"\nWorker {worker_id} Starting:\n"
            f"- Process ID: {current_pid}\n"
            f"- Parent Process ID: {os.getppid()}"
        )

        try:
            # Verify multiprocessing setup
            verify_spawn_method()

            # Load study
            study = optuna.load_study(
                study_name=self._study_name,
                storage=self._storage_url
            )
            logger.info(f"Worker {worker_id} initialized and connected to study")

            # Start health check thread
            def health_check():
                while True:
                    try:
                        resources = self._monitor_resources()
                        if not self._check_resource_limits(resources):
                            logger.warning(
                                f"Worker {worker_id} exceeding resource limits:\n"
                                f"CPU: {resources.get('cpu_percent', 0)}%\n"
                                f"Memory: {resources.get('memory_gb', 0):.1f}GB\n"
                                f"GPU Memory: {resources.get('gpu_memory_gb', 0):.1f}GB"
                            )
                    except Exception as e:
                        logger.error(f"Health check failed in worker {worker_id}: {str(e)}")
                        logger.error(traceback.format_exc())
                    time.sleep(60)

            health_thread = threading.Thread(target=health_check, daemon=True)
            health_thread.start()

            # Main worker loop
            while True:
                try:
                    # Get trial data with timeout
                    trial_data = self._local.worker_queue.get(timeout=300)
                    if trial_data is None:
                        logger.info(f"Worker {worker_id} received exit signal")
                        break

                    # Process trial
                    trial_number = trial_data['trial_number']
                    logger.info(
                        f"\nTrial {trial_number} Starting in Worker {worker_id}:\n"
                        f"- Process ID: {current_pid}"
                    )

                    # Initialize resources
                    config = trial_data['config']
                    output_path = Path(trial_data['output_path'])
                    ResourceInitializer.initialize_process(config)

                    # Create trial
                    trial = optuna.trial.FixedTrial(trial_data['trial_params'])

                    try:
                        # Execute trial
                        factory = ObjectiveFactory(config, output_path)
                        result = factory.objective(trial)
                        logger.info(f"Trial {trial_number} completed with result: {result}")
                        self._local.result_queue.put((trial_number, result, None))

                    except Exception as e:
                        logger.error(f"Trial {trial_number} failed: {str(e)}")
                        logger.error(traceback.format_exc())
                        self._local.result_queue.put((trial_number, None, str(e)))

                    finally:
                        # Clean up trial resources
                        logger.info(f"Cleaning up resources for trial {trial_number}")
                        self._model_manager.cleanup_worker(worker_id)
                        self._tokenizer_manager.cleanup_worker(worker_id)
                        ResourceInitializer.cleanup_process()

                except Exception as e:
                    logger.error(f"Error in worker loop: {str(e)}")
                    logger.error(traceback.format_exc())
                    if 'trial_number' in locals():
                        self._local.result_queue.put((trial_number, None, str(e)))

        except Exception as e:
            logger.error(f"Worker {worker_id} failed: {str(e)}")
            logger.error(traceback.format_exc())

        finally:
            # Clean up worker resources
            logger.info(f"\nWorker {worker_id} Shutting Down")
            self._model_manager.cleanup_worker(worker_id)
            self._tokenizer_manager.cleanup_worker(worker_id)
            ResourceInitializer.cleanup_process()
            logger.info(f"Worker {worker_id} cleanup complete")

    def cleanup_workers(
        self,
        group: Optional[str] = None,
        timeout: int = 30
    ) -> None:
        """
        Clean up worker processes.

        Args:
            group: Optional group name to clean up specific workers
            timeout: Timeout in seconds for worker shutdown
        """
        self.ensure_initialized()

        try:
            # Determine workers to clean up
            if group:
                if group not in self._local.worker_groups:
                    logger.warning(f"No workers found in group '{group}'")
                    return
                logger.info(f"Cleaning up workers in group '{group}'")
                workers_to_cleanup = self._local.worker_groups[group]
            else:
                logger.info("Cleaning up all worker processes")
                workers_to_cleanup = self._local.active_workers

            # Clean up workers
            for worker_id, process in workers_to_cleanup.items():
                self._local.worker_queue.put(None)  # Send exit signal
                process.join(timeout=timeout)
                if process.is_alive():
                    process.terminate()

                self._local.active_workers.pop(worker_id, None)
                if group:
                    self._local.worker_groups[group].pop(worker_id, None)
                    if not self._local.worker_groups[group]:
                        del self._local.worker_groups[group]

            # Clean up groups if needed
            if not group:
                self._local.worker_groups.clear()
                if self._health_check_thread and self._health_check_thread.is_alive():
                    self._health_check_thread = None

        except Exception as e:
            logger.error(f"Error cleaning up workers: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def queue_trial(self, trial_data: Dict[str, Any]) -> None:
        """
        Queue a trial for execution.

        Args:
            trial_data: Trial data dictionary
        """
        self.ensure_initialized()

        try:
            # Verify trial data is picklable
            pickle.dumps(trial_data)
            self._local.worker_queue.put(trial_data)
            logger.info(f"Queued trial {trial_data['trial_number']}")

        except Exception as e:
            logger.error(f"Failed to queue trial: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_result(self) -> Tuple[int, Optional[float], Optional[str]]:
        """
        Get result from a completed trial.

        Returns:
            Tuple containing:
            - Trial number
            - Optional result value
            - Optional error message
        """
        self.ensure_initialized()
        return self._local.result_queue.get()

    def scale_workers(
        self,
        n_jobs: int,
        group: str = "default",
        check_resources: bool = True
    ) -> None:
        """
        Scale number of workers up or down.

        Args:
            n_jobs: Target number of workers
            group: Worker group name
            check_resources: Whether to check resources when scaling up

        Raises:
            RuntimeError: If insufficient resources
            ValueError: If n_jobs exceeds max_workers
        """
        self.ensure_initialized()

        try:
            # Check resources when scaling up
            if check_resources and n_jobs > len(self._local.worker_groups.get(group, {})):
                resources = self._monitor_resources()
                if not self._check_resource_limits(resources):
                    raise RuntimeError(
                        f"Insufficient resources to scale to {n_jobs} workers:\n"
                        f"CPU: {resources.get('cpu_percent', 0)}%\n"
                        f"Memory: {resources.get('memory_gb', 0):.1f}GB\n"
                        f"GPU Memory: {resources.get('gpu_memory_gb', 0):.1f}GB"
                    )

            # Check worker limit
            if n_jobs > self._max_workers:
                raise ValueError(
                    f"Cannot scale to {n_jobs} workers - exceeds max_workers "
                    f"({self._max_workers})"
                )

            current_workers = len(self._local.worker_groups.get(group, {}))

            # Scale up
            if n_jobs > current_workers:
                self._n_jobs = n_jobs - current_workers
                self.start_workers(group)

            # Scale down
            elif n_jobs < current_workers:
                workers_to_remove = current_workers - n_jobs
                workers = list(self._local.worker_groups[group].items())[-workers_to_remove:]

                for worker_id, _ in workers:
                    self._local.worker_queue.put(None)  # Send exit signal
                    self._local.active_workers.pop(worker_id, None)
                    self._local.worker_groups[group].pop(worker_id, None)

            self._n_jobs = n_jobs

        except Exception as e:
            logger.error(f"Error scaling workers: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def cleanup(self) -> None:
        """Clean up worker manager resources."""
        try:
            self.cleanup_workers()
            logger.info(f"Cleaned up WorkerManager for process {self._local.pid}")
            super().cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up WorkerManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise


__all__ = ['WorkerManager']
