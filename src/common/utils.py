# src/common/utils.py
from __future__ import annotations

import logging
import os
import gc
import random
import psutil
import numpy as np
import torch
import torch.multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, TypeVar, Iterator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps
import yaml

logger = logging.getLogger(__name__)

T = TypeVar('T')

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Sets up logging based on the provided configuration.

    Args:
        config: A dictionary containing logging configuration, or None to use defaults.
                 Expected keys within the 'logging' section (or 'training' if 'logging' is absent):
                     'level': (str) The desired logging level (e.g., "DEBUG", "INFO", "WARNING").
                     'file': (str, optional) The file to write logs to. If None, logs only to console.
                     'format': (str, optional) Custom logging format string.
    """
    import coloredlogs

    log_config = config.get('logging', config.get('training', {}))
    level_str = log_config.get('level', 'INFO').upper()
    log_file = log_config.get('file')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        numeric_level = getattr(logging, level_str)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: "{level_str}"')
    except (AttributeError, ValueError):
        print(f"Invalid log level: {level_str}.  Defaulting to INFO.")
        numeric_level = logging.INFO

    logging.basicConfig(
        level=logging.DEBUG,  # ALWAYS log everything internally
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Always log to console
            *([] if log_file is None else [logging.FileHandler(log_file)])  # Add file handler if specified
        ]
    )

    logging.getLogger().setLevel(numeric_level)

    coloredlogs.install(level=numeric_level)

    logger.info(f"Logging level set to: {logging.getLevelName(numeric_level)}")

def seed_everything(seed: int) -> None:
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_memory_usage() -> Dict[str, float]:
    """Gets the current memory usage (both CPU and GPU, if available)."""
    process = psutil.Process()
    memory_info = process.memory_info()

    usage = {
        'rss': memory_info.rss / (1024 ** 3), #Residen Set Size
        'vms': memory_info.vms / (1024 ** 3), #Virtual Memory Size
    }

    if torch.cuda.is_available():
        usage.update({
            'cuda_allocated': torch.cuda.memory_allocated() / (1024 ** 3),
            'cuda_reserved': torch.cuda.memory_reserved() / (1024 ** 3)
        })

    return usage

def clear_memory() -> None:
    """Clears GPU memory and runs garbage collection."""
    from src.common.managers import get_tokenizer_manager
    tokenizer_manager = get_tokenizer_manager()
    tokenizer_manager.cleanup_worker(os.getpid())

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def parallel_map(
    func: Callable[[T], Any],
    items: List[T],
    max_workers: Optional[int] = None,
    use_processes: bool = False,
    desc: Optional[str] = None,
    chunk_size: Optional[int] = None,
    wandb_manager: Optional[Any] = None
) -> List[Any]:
    """
    Applies a function to a list of items in parallel, using threads or processes.

    Args:
        func: The function to apply.
        items: The list of items to process.
        max_workers: The maximum number of worker threads/processes. Defaults to the number of CPUs.
        use_processes: Whether to use processes (True) or threads (False).
        desc: A description for logging/progress tracking.
        chunk_size: The chunk size for processing.
        wandb_manager: Optional WandB manager for logging progress.

    Returns:
        A list of results, in the same order as the input items.
    """

    if max_workers is None:
        max_workers = os.cpu_count() or 4

    if chunk_size is None:
        chunk_size = max(1, len(items) // (max_workers * 4))

    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    with executor_class(max_workers=max_workers) as executor:
        results = []
        total = len(items)

        for i, result in enumerate(executor.map(func, items, chunksize=chunk_size)):
            results.append(result)
            if wandb_manager:
                wandb_manager.log_progress(i + 1, total, prefix='parallel_')
                if desc:
                    wandb_manager.log_metrics({'parallel_task': desc})

        return results

def batch_iterator(
    items: List[T],
    batch_size: int,
    drop_last: bool = False
) -> Iterator[List[T]]:
    """Iterates over a list in batches."""
    length = len(items)
    for ndx in range(0, length, batch_size):
        batch = items[ndx:min(ndx + batch_size, length)]
        if drop_last and len(batch) < batch_size:
            break
        yield batch

def create_memmap_array(
    path: Path,
    shape: tuple,
    dtype: np.dtype = np.float32,
    data: Optional[np.ndarray] = None
) -> np.ndarray:
    """Creates a memory-mapped array."""
    if data is not None:
        array = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
        array[:] = data[:]
        array.flush()
    else:
        array = np.memmap(path, dtype=dtype, mode='w+', shape=shape)

    return array

def load_memmap_array(
    path: Path,
    shape: tuple,
    dtype: np.dtype = np.float32,
    mode: str = 'r'
) -> np.ndarray:
    """Loads a memory-mapped array."""
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)

def measure_memory(func: Callable) -> Callable:
    """Decorator to measure memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        before = get_memory_usage()
        try:
            result = func(*args, **kwargs)
            after = get_memory_usage()

            diff = {
                k: after[k] - before[k]
                for k in before.keys()
            }

            logger.debug(
                f"Memory usage for {func.__name__}:\n"
                f"  Before: {before}\n"
                f"  After: {after}\n"
                f"  Difference: {diff}"
            )

            return result
        finally:
            clear_memory()

    return wrapper

def chunk_file(
    file_path: Path,
    chunk_size: str = '64MB'
) -> Iterator[bytes]:
    """Reads a file in chunks."""
    units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    size = int(chunk_size[:-2])
    unit = chunk_size[-2:].upper()
    chunk_bytes = size * units[unit]

    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            yield chunk

def init_worker():
    """Initialize worker for dataloaders"""
    logger.info(f"Initializing worker process: {os.getpid()}")
    if torch.cuda.is_available():
        torch.cuda.init()
        logger.info(f"CUDA initialized in worker process: {os.getpid()}")

def get_worker_init_fn(num_workers: int) -> Optional[callable]:
    """Get worker initialization function if needed."""
    if num_workers > 0:
        return init_worker
    return None

# Import the properly validated version from config_utils
from .config_utils import load_yaml_config
__all__ = [
    'setup_logging',
    'seed_everything',
    'get_memory_usage',
    'clear_memory',
    'load_yaml_config',  # Re-exported from config_utils
    'parallel_map',
    'batch_iterator',
    'create_memmap_array',
    'load_memmap_array',
    'chunk_file',
    'measure_memory',
    'init_worker',
    'get_worker_init_fn'
]
