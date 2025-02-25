#common/logging_utils.py
from __future__ import annotations
import logging
import functools
from typing import Any, Callable, TypeVar, Optional, Dict
from dataclasses import dataclass
import pandas as pd
import torch
from tqdm.notebook import tqdm

# Type variable for generic function decorator
F = TypeVar('F', bound=Callable[..., Any])

@dataclass
class LogConfig:
    """Configuration for logging levels."""
    level: str = 'none'  # 'debug', 'log', or 'none'
    tqdm_enabled: bool = True

    def __post_init__(self) -> None:
        if self.level not in ['debug', 'log', 'none']:
            raise ValueError(f"Invalid logging level: {self.level}")

def get_shape_info(obj: Any) -> str:
    """Get shape and type information for various data types."""
    if isinstance(obj, torch.Tensor):
        return f"Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device})"
    elif isinstance(obj, pd.DataFrame):
        return f"DataFrame(shape={obj.shape}, columns={list(obj.columns)})"
    elif isinstance(obj, dict):
        return f"Dict(keys={list(obj.keys())})"
    elif isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__}(len={len(obj)})"
    else:
        return f"{type(obj).__name__}"

def get_preview(obj: Any) -> str:
    """Get a preview of the data content."""
    try:
        if isinstance(obj, torch.Tensor):
            return f"First few elements: {obj.flatten()[:5].tolist()}"
        elif isinstance(obj, pd.DataFrame):
            return f"Head:\n{obj.head().to_string()}"
        elif isinstance(obj, dict):
            preview = {k: obj[k] for k in list(obj.keys())[:3]}
            return f"First few items: {preview}"
        elif isinstance(obj, (list, tuple)):
            return f"First few items: {obj[:5]}"
        return str(obj)[:100]
    except Exception:
        return "Preview not available"

def log_function(level: str = 'log') -> Callable[[F], F]:
    """
    Decorator for function logging.
    
    Args:
        level: Logging level ('debug', 'log', or 'none')
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = logging.getLogger(func.__module__)
            
            # Skip logging if level is 'none'
            if level == 'none':
                return func(*args, **kwargs)
            
            # Log function entry
            logger.info(f"Entering {func.__name__}")
            if level == 'debug':
                # Log argument details
                for i, arg in enumerate(args):
                    logger.debug(f"Arg {i}: {get_shape_info(arg)}")
                    logger.debug(f"Preview: {get_preview(arg)}")
                for k, v in kwargs.items():
                    logger.debug(f"Kwarg {k}: {get_shape_info(v)}")
                    logger.debug(f"Preview: {get_preview(v)}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Log function exit and result
            logger.info(f"Exiting {func.__name__}")
            if level == 'debug':
                logger.debug(f"Return: {get_shape_info(result)}")
                logger.debug(f"Preview: {get_preview(result)}")
            
            return result
        return wrapper  # type: ignore
    return decorator

def create_progress_bar(
    iterable: Any,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    enabled: bool = True
) -> Any:
    """
    Create a progress bar that works in both notebook and terminal environments.
    
    Args:
        iterable: Iterable to wrap with progress bar
        desc: Description for the progress bar
        total: Total number of items (optional)
        enabled: Whether to show the progress bar
    
    Returns:
        Wrapped iterable with progress bar
    """
    if not enabled:
        return iterable
        
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=True,
        position=0,
        dynamic_ncols=True
    )

__all__ = [
    'LogConfig',
    'log_function',
    'create_progress_bar'
]