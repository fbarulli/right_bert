"""
Utility classes for the embedding pipeline.
"""
import gc
import logging
import threading
import time
import torch
from typing import Dict, Any, Tuple, Optional, List, Union
import psutil
from collections import OrderedDict

logger = logging.getLogger(__name__)

class LogConfig:
    """Configuration for logging behavior."""
    
    def __init__(self, level: str = 'log'):
        """
        Initialize logging configuration.
        
        Args:
            level: Logging level ('debug', 'log', 'none')
        """
        self.level = level
        self._validate()
    
    def _validate(self) -> None:
        """Validate logging configuration."""
        if self.level not in ['debug', 'log', 'none']:
            logger.warning(f"Invalid log level: {self.level}. Using 'log' instead.")
            self.level = 'log'
    
    def should_log(self) -> bool:
        """Check if logging is enabled."""
        return self.level != 'none'
    
    def should_debug(self) -> bool:
        """Check if debug logging is enabled."""
        return self.level == 'debug'


class MemoryTracker:
    """
    Track memory usage and trigger garbage collection when needed.
    """
    
    def __init__(self, gc_threshold: float = 0.8, log_config: Optional[LogConfig] = None):
        """
        Initialize memory tracker.
        
        Args:
            gc_threshold: Memory usage threshold to trigger garbage collection (0.0-1.0)
            log_config: Logging configuration
        """
        self.gc_threshold = gc_threshold
        self.log_config = log_config or LogConfig(level='none')
        self.last_gc_time = time.time()
        self.last_update_time = time.time()
        self.memory_stats = {
            'peak_allocated': 0,
            'current_allocated': 0,
            'peak_reserved': 0,
            'current_reserved': 0,
            'num_gc_calls': 0
        }
        self.lock = threading.RLock()
    
    def update(self) -> None:
        """
        Update memory stats and trigger garbage collection if needed.
        """
        with self.lock:
            current_time = time.time()
            self.last_update_time = current_time
            
            # Only check memory every second to avoid overhead
            if current_time - self.last_gc_time < 1.0:
                return
            
            # Update memory stats
            if torch.cuda.is_available():
                current_allocated = torch.cuda.memory_allocated()
                current_reserved = torch.cuda.memory_reserved()
                max_memory = torch.cuda.get_device_properties(0).total_memory
                usage = current_reserved / max_memory
                
                self.memory_stats['current_allocated'] = current_allocated
                self.memory_stats['current_reserved'] = current_reserved
                self.memory_stats['peak_allocated'] = max(
                    self.memory_stats['peak_allocated'], 
                    current_allocated
                )
                self.memory_stats['peak_reserved'] = max(
                    self.memory_stats['peak_reserved'], 
                    current_reserved
                )
                
                # Trigger garbage collection if memory usage is high
                if usage > self.gc_threshold:
                    self._trigger_gc()
            else:
                # CPU-only version using psutil
                process = psutil.Process()
                current_memory = process.memory_info().rss
                max_memory = psutil.virtual_memory().total
                usage = current_memory / max_memory
                
                self.memory_stats['current_allocated'] = current_memory
                self.memory_stats['peak_allocated'] = max(
                    self.memory_stats['peak_allocated'], 
                    current_memory
                )
                
                # Trigger garbage collection if memory usage is high
                if usage > self.gc_threshold:
                    self._trigger_gc()
    
    def _trigger_gc(self) -> None:
        """
        Trigger garbage collection.
        """
        if self.log_config.should_debug():
            logger.debug("Triggering garbage collection")
        
        # Trigger Python's garbage collector
        gc.collect()
        
        # Empty CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.last_gc_time = time.time()
        self.memory_stats['num_gc_calls'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dict[str, Any]: Memory statistics
        """
        with self.lock:
            # Create a copy to avoid external modifications
            return self.memory_stats.copy()


class TensorPool:
    """
    Pool of reusable tensors to reduce memory allocations.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize tensor pool.
        
        Args:
            max_size: Maximum number of tensors in the pool
        """
        self.max_size = max_size
        self.pools = {}  # Dict[Tuple[shape, dtype, device], List[Tensor]]
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'pool_size': 0
        }
    
    def get(self, shape: Union[torch.Size, Tuple, List], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Get a tensor from the pool or create a new one if needed.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            device: Tensor device
            
        Returns:
            torch.Tensor: Tensor from the pool or newly created
        """
        with self.lock:
            # Convert shape to tuple for hashability
            if isinstance(shape, torch.Size):
                shape = tuple(shape)
            elif isinstance(shape, list):
                shape = tuple(shape)
            
            key = (shape, dtype, str(device))
            
            # Check if we have a tensor in the pool
            if key in self.pools and self.pools[key]:
                tensor = self.pools[key].pop()
                self.stats['hits'] += 1
                self.stats['pool_size'] -= 1
                return tensor
            
            # Create a new tensor
            self.stats['misses'] += 1
            return torch.zeros(shape, dtype=dtype, device=device)
    
    def put(self, tensor: torch.Tensor) -> None:
        """
        Return a tensor to the pool.
        
        Args:
            tensor: Tensor to return to the pool
        """
        with self.lock:
            if self.stats['pool_size'] >= self.max_size:
                return  # Pool is full, let the tensor be garbage collected
            
            # Reset tensor to save memory (zeros are more compressible)
            tensor.zero_()
            
            key = (tuple(tensor.shape), tensor.dtype, str(tensor.device))
            
            if key not in self.pools:
                self.pools[key] = []
            
            self.pools[key].append(tensor)
            self.stats['pool_size'] += 1
    
    def clear(self) -> None:
        """Clear the tensor pool."""
        with self.lock:
            self.pools = {}
            self.stats['pool_size'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics.
        
        Returns:
            Dict[str, Any]: Pool statistics
        """
        with self.lock:
            # Create a copy to avoid external modifications
            return self.stats.copy()


class CachingDict:
    """
    Dictionary with LRU caching and memory usage tracking.
    """
    
    def __init__(self, maxsize: int = 1000, memory_tracker: Optional[MemoryTracker] = None):
        """
        Initialize caching dictionary.
        
        Args:
            maxsize: Maximum number of items in the cache
            memory_tracker: Memory tracker instance
        """
        self.maxsize = maxsize
        self.memory_tracker = memory_tracker
        self.cache = OrderedDict()  # For LRU behavior
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'current_size': 0
        }
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached item or None if not found
        """
        with self.lock:
            if key in self.cache:
                # Move to the end to mark as recently used
                value = self.cache.pop(key)
                self.cache[key] = value
                self.stats['hits'] += 1
                return value
            
            self.stats['misses'] += 1
            return None
    
    def set(self, key: Any, value: Any) -> None:
        """
        Set an item in the cache.
        
        Args:
            key: Cache key
            value: Item to cache
        """
        with self.lock:
            # If key already exists, remove it first
            if key in self.cache:
                del self.cache[key]
            
            # Check if we need to evict an item
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)  # Remove least recently used
                self.stats['evictions'] += 1
            
            # Add the new item
            self.cache[key] = value
            self.stats['current_size'] = len(self.cache)
            
            # Update memory tracker
            if self.memory_tracker:
                self.memory_tracker.update()
    
    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.stats['current_size'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        with self.lock:
            # Create a copy to avoid external modifications
            return self.stats.copy()


__all__ = [
    'LogConfig',
    'MemoryTracker',
    'TensorPool',
    'CachingDict'
]
