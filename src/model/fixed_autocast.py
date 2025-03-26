"""
Device-aware autocast utilities.

Provides consistent mixed precision across different device backends.
"""

import torch
from typing import Optional, Dict, Any


def get_autocast_dtype(device_type: str) -> torch.dtype:
    """
    Get the appropriate dtype for autocast based on device type.
    
    Args:
        device_type: Device type ('cuda', 'mps', 'cpu')
    
    Returns:
        torch.dtype: Appropriate dtype for the device
    """
    if device_type == 'cuda':
        return torch.float16
    elif device_type == 'mps':
        return torch.float16  # MPS supports float16
    else:
        return torch.bfloat16  # Better for CPU


def get_autocast_device_type(device: Optional[torch.device] = None) -> str:
    """
    Get the appropriate device type string for autocast.
    
    Args:
        device: Optional device to determine type from
        
    Returns:
        str: Device type for autocast
    """
    if device is None:
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    # Get device type from provided device
    if hasattr(device, 'type'):
        device_type = device.type
        
        # Map to supported autocast device types
        if device_type.startswith('cuda'):
            return 'cuda'
        elif device_type == 'mps':
            return 'mps'
    
    # Default to CPU
    return 'cpu'


def device_aware_autocast(device: Optional[torch.device] = None, 
                          enabled: bool = True, 
                          **kwargs):
    """
    Use device-aware autocast that works with CUDA, MPS (Apple Silicon), or CPU.
    
    Args:
        device: Device to determine autocast type
        enabled: Whether autocast is enabled
        **kwargs: Additional args to pass to autocast
        
    Returns:
        Autocast context manager
    """
    device_type = get_autocast_device_type(device)
    
    # Use the newer API signature
    return torch.amp.autocast(device_type=device_type, enabled=enabled, **kwargs)