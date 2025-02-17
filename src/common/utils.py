# src/common/utils.py
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
    import coloredlogs

    log_config = config['logging'] if 'logging' in config else config['training']
    level_str = log_config['level'].upper()
    log_file = log_config['file']
    log_format = log_config['format']

    numeric_level = getattr(logging, level_str, logging.INFO)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            *([] if log_file is None else [logging.FileHandler(log_file)])
        ]
    )

    logging.getLogger().setLevel(numeric_level)

    coloredlogs.install(level=numeric_level)

    logger.info(f"Logging level set to: {logging.getLevelName(numeric_level)}")

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_memory_usage() -> Dict[str, float]:
    process = psutil.Process()
    memory_info = process.memory_info()

    usage = {
        'rss': memory_info.rss / (1024 ** 3),
        'vms': memory_info.vms / (1024 ** 3),
    }

    if torch.cuda.is_available():
        usage.update({
            'cuda_allocated': torch.cuda.memory_allocated() / (1024 ** 3),
            'cuda_reserved': torch.cuda.memory_reserved() / (1024 ** 3)
        })

    return usage

def clear_memory() -> None:
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
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)

def measure_memory(func: Callable) -> Callable:
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
    logger.info(f"Initializing worker process: {os.getpid()}")
    if torch.cuda.is_available():
        torch.cuda.init()
        logger.info(f"CUDA initialized in worker process: {os.getpid()}")

def get_worker_init_fn(num_workers: int) -> Optional[callable]:
    if num_workers > 0:
        return init_worker
    return None

from .config_utils import load_yaml_config
__all__ = [
    'setup_logging',
    'seed_everything',
    'get_memory_usage',
    'clear_memory',
    'load_yaml_config',
    'parallel_map',
    'batch_iterator',
    'create_memmap_array',
    'load_memmap_array',
    'chunk_file',
    'measure_memory',
    'init_worker',
    'get_worker_init_fn'
]