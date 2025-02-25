"""
Memory management utilities for efficient embedding training.
"""
from src.embedding.imports import (
    torch,
    Dict,
    Optional,
    Tuple,
    List,
    Set,
    Any,
    Tensor,
    logger,
    log_function,
    LogConfig,
    gc,
)

class GPUMemoryManager:
    """Manages GPU memory allocation and cleanup."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        trial_id: Optional[str] = None
    ):
        self.device = device or (
            torch.device('cuda')
            if torch.cuda.is_available()
            else torch.device('cpu')
        )
        self.trial_id = trial_id
        self.tracked_tensors: Set[int] = set()
        
    def register_tensor(self, tensor: Tensor) -> None:
        """Register a tensor for tracking."""
        if isinstance(tensor, torch.Tensor):
            self.tracked_tensors.add(id(tensor))
            
    def unregister_tensor(self, tensor: Tensor) -> None:
        """Unregister a tensor from tracking."""
        tensor_id = id(tensor)
        if tensor_id in self.tracked_tensors:
            self.tracked_tensors.remove(tensor_id)
            
    def clear_trial_memory(self) -> None:
        """Clear memory for current trial only."""
        if not torch.cuda.is_available():
            return
            
        # Clear only tensors registered for this trial
        for tensor_id in self.tracked_tensors.copy():
            try:
                tensor = next(
                    obj for obj in gc.get_objects()
                    if isinstance(obj, torch.Tensor) and id(obj) == tensor_id
                )
                if tensor.is_cuda:
                    del tensor
            except StopIteration:
                self.tracked_tensors.remove(tensor_id)
                
        # Clear CUDA cache for this device
        torch.cuda.empty_cache()
        if self.trial_id:
            logger.info(f"Cleared memory for trial {self.trial_id}")

    def clear_tensor_cache(self) -> None:
        """Clear PyTorch's tensor cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_memory_snapshot(self) -> Dict[str, float]:
        """Get current GPU memory usage snapshot."""
        if not torch.cuda.is_available():
            return {}
            
        return {
            'allocated_gb': torch.cuda.memory_allocated(self.device) / 1e9,
            'reserved_gb': torch.cuda.memory_reserved(self.device) / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated(self.device) / 1e9,
            'max_reserved_gb': torch.cuda.max_memory_reserved(self.device) / 1e9,
            'num_tracked_tensors': len(self.tracked_tensors)
        }

class TensorPool:
    """Memory pool for reusing tensors to reduce allocations."""
    
    def __init__(
        self,
        max_size: int = 1000,
        trial_id: Optional[str] = None,
        gpu_manager: Optional[GPUMemoryManager] = None
    ):
        self.pools: Dict[Tuple[torch.dtype, tuple], List[Tensor]] = {}
        self.max_size = max_size
        self.trial_id = trial_id
        self.gpu_manager = gpu_manager or GPUMemoryManager(trial_id=trial_id)
        
    def get(self, shape: tuple, dtype: torch.dtype, device: torch.device) -> Tensor:
        """Get a tensor from the pool or create a new one."""
        key = (dtype, shape)
        if key in self.pools and self.pools[key]:
            tensor = self.pools[key].pop()
            tensor.zero_()  # Ensure clean state
            return tensor
            
        # Create new tensor and register it
        tensor = torch.zeros(shape, dtype=dtype, device=device)
        self.gpu_manager.register_tensor(tensor)
        return tensor
    
    def put(self, tensor: Tensor) -> None:
        """Return a tensor to the pool."""
        key = (tensor.dtype, tuple(tensor.shape))
        if key not in self.pools:
            self.pools[key] = []
        if len(self.pools[key]) < self.max_size:
            self.pools[key].append(tensor.detach())
            self.gpu_manager.register_tensor(tensor)

    def clear(self) -> None:
        """Clear all pooled tensors."""
        self.pools.clear()
        self.gpu_manager.clear_trial_memory()

class MemoryTracker:
    """Track and optimize memory usage during training."""
    
    def __init__(
        self,
        gc_threshold: float = 0.8,
        log_config: Optional[LogConfig] = None,
        trial_id: Optional[str] = None,
        gpu_manager: Optional[GPUMemoryManager] = None
    ):
        self.gc_threshold = gc_threshold
        self.log_config = log_config or LogConfig()
        self.trial_id = trial_id
        self.gpu_manager = gpu_manager or GPUMemoryManager(trial_id=trial_id)
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.last_gc = 0
        
    def update(self) -> None:
        """Update memory statistics and trigger GC if needed."""
        if not torch.cuda.is_available():
            return

        current_allocated = torch.cuda.memory_allocated()
        current_reserved = torch.cuda.memory_reserved()
        
        self.peak_allocated = max(self.peak_allocated, current_allocated)
        self.peak_reserved = max(self.peak_reserved, current_reserved)
        
        # Calculate memory pressure
        if current_reserved > 0:
            pressure = current_allocated / current_reserved
            if pressure > self.gc_threshold:
                self._force_gc()
                
        if self.log_config.level == 'debug':
            logger.debug(
                f"Memory stats for trial {self.trial_id}:\n"
                f"- Current allocated: {current_allocated / 1e9:.2f} GB\n"
                f"- Current reserved: {current_reserved / 1e9:.2f} GB\n"
                f"- Peak allocated: {self.peak_allocated / 1e9:.2f} GB"
            )

    def _force_gc(self) -> None:
        """Force garbage collection for this trial."""
        self.gpu_manager.clear_trial_memory()
        self.last_gc = torch.cuda.memory_allocated()
        
    def get_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        return {
            **self.gpu_manager.get_memory_snapshot(),
            'last_gc_gb': self.last_gc / 1e9
        }

    def reset_peaks(self) -> None:
        """Reset peak memory statistics."""
        self.peak_allocated = 0
        self.peak_reserved = 0

class CachingDict:
    """Dictionary with LRU caching and memory-aware eviction."""
    
    def __init__(
        self,
        maxsize: int = 1000,
        trial_id: Optional[str] = None,
        memory_tracker: Optional[MemoryTracker] = None,
        gpu_manager: Optional[GPUMemoryManager] = None
    ):
        self.maxsize = maxsize
        self.cache: Dict[Any, Tuple[Any, int]] = {}
        self.trial_id = trial_id
        self.memory_tracker = memory_tracker
        self.gpu_manager = gpu_manager or GPUMemoryManager(trial_id=trial_id)
        self.hits = 0
        self.misses = 0
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Get item from cache."""
        if key in self.cache:
            value, count = self.cache[key]
            self.cache[key] = (value, count + 1)
            self.hits += 1
            return value
        self.misses += 1
        return default
    
    def set(self, key: Any, value: Any) -> None:
        """Set item in cache."""
        if len(self.cache) >= self.maxsize:
            # Evict least accessed items
            items = sorted(
                self.cache.items(),
                key=lambda x: x[1][1]
            )
            for old_key, _ in items[:len(items)//4]:
                del self.cache[old_key]
            
            self.gpu_manager.clear_tensor_cache()
                
        self.cache[key] = (value, 0)
        if isinstance(value, torch.Tensor):
            self.gpu_manager.register_tensor(value)
        
        if self.memory_tracker:
            self.memory_tracker.update()
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.gpu_manager.clear_trial_memory()
        if self.memory_tracker:
            self.memory_tracker.update()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_ratio': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            'memory': self.gpu_manager.get_memory_snapshot()
        }

__all__ = [
    'GPUMemoryManager',
    'TensorPool',
    'MemoryTracker',
    'CachingDict',
]