"""
Fixed autocast implementation for QLLM.

This module provides a consistent implementation of autocast context manager
that addresses issues with PyTorch's native autocast in certain scenarios.
It has been refactored to remove duplicated implementations.
"""

import contextlib
import torch
from typing import Any, Generator, Optional, List, Dict


class fixed_autocast(torch.autocast):
    """
    Fixed version of PyTorch's autocast.
    
    This context manager provides a more consistent behavior for mixed precision
    operations, addressing issues with the native PyTorch implementation in
    certain scenarios, particularly involving homomorphic operations.
    """
    
    def __init__(
        self,
        device_type: str = "cuda",
        dtype: Optional[torch.dtype] = torch.float16,
        enabled: bool = True,
        cache_enabled: Optional[bool] = None
    ):
        """
        Initialize the fixed autocast context manager.
        
        Args:
            device_type: Device type to autocast for ('cuda', 'cpu', or 'xpu')
            dtype: Data type to use for autocast
            enabled: Whether autocast is enabled
            cache_enabled: Whether autocast's weight cache is enabled
        """
        if torch.cuda.is_available() and device_type == "cuda":
            # For CUDA devices, default to float16 for better performance
            dtype = dtype or torch.float16
        else:
            # For CPU, default to bfloat16 which has better numerical properties
            dtype = dtype or torch.bfloat16
            
            # Check if bfloat16 is supported by CPU
            if device_type == "cpu" and not hasattr(torch, 'bfloat16'):
                # Fall back to float32 if bfloat16 is not supported
                dtype = torch.float32
        
        # For newer PyTorch versions, cache_enabled is a valid parameter
        if cache_enabled is not None:
            try:
                super().__init__(device_type=device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
                return
            except TypeError:
                # Fall back to older signature if cache_enabled is not supported
                pass
        
        # Initialize with the basic signature for older PyTorch versions
        super().__init__(device_type=device_type, dtype=dtype, enabled=enabled)


@contextlib.contextmanager
def nullcontext(enter_result: Any = None) -> Generator[Any, None, None]:
    """
    Context manager that does nothing.
    
    This is a fallback for when autocast is not available or not needed.
    
    Args:
        enter_result: Value to yield from the context manager
        
    Yields:
        The enter_result
    """
    yield enter_result


def get_autocast_dtype(device_type: str = "cuda") -> torch.dtype:
    """
    Get the appropriate autocast dtype for the given device type.
    
    Args:
        device_type: Device type ('cuda', 'cpu', or 'xpu')
        
    Returns:
        Appropriate dtype for autocast
    """
    if device_type == "cuda" and torch.cuda.is_available():
        return torch.float16
    elif hasattr(torch, 'bfloat16'):
        return torch.bfloat16
    else:
        return torch.float32


def is_autocast_available() -> bool:
    """
    Check if autocast is available in the current PyTorch version.
    
    Returns:
        True if autocast is available
    """
    return hasattr(torch, 'autocast')


def get_autocast_context(
    device_type: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    enabled: bool = True
) -> Any:
    """
    Get the appropriate autocast context for the given parameters.
    
    Args:
        device_type: Device type to autocast for
        dtype: Data type to use for autocast
        enabled: Whether autocast is enabled
        
    Returns:
        Autocast context manager or nullcontext if not available/enabled
    """
    if not is_autocast_available() or not enabled:
        return nullcontext()
    
    # Use fixed_autocast which handles edge cases better
    return fixed_autocast(device_type=device_type, dtype=dtype, enabled=enabled)