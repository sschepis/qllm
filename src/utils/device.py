"""
Device handling utilities for the Quantum Resonance Language Model.
Provides consistent device setup and management across training and inference.
"""

import os
import torch
from typing import Optional, Union, Tuple


def get_default_device() -> torch.device:
    """
    Get the default device (CUDA if available, otherwise CPU).
    
    Returns:
        torch.device: The default device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_device_str(device_str: Optional[str] = None) -> torch.device:
    """
    Parse a device string into a torch.device.
    
    Args:
        device_str: Device string specification (e.g., "cuda", "cuda:0", "cpu")
        
    Returns:
        torch.device: The specified device or default if None
    """
    if not device_str:
        return get_default_device()
    
    # Handle the case with specific GPU index
    if device_str.startswith("cuda:"):
        try:
            index = int(device_str.split(":")[1])
            if index >= torch.cuda.device_count():
                print(f"Warning: Specified GPU index {index} is out of range. "
                      f"Using default device instead.")
                return get_default_device()
        except (ValueError, IndexError):
            print(f"Warning: Invalid CUDA device format '{device_str}'. "
                  f"Using default device instead.")
            return get_default_device()
    
    # Handle CPU/CUDA basic case
    return torch.device(device_str)


def get_device(
    device_str: Optional[str] = None,
    fallback_to_cpu: bool = True
) -> torch.device:
    """
    Get a device from string specification, with fallback logic.
    
    Args:
        device_str: Device string specification
        fallback_to_cpu: Whether to fallback to CPU if CUDA is not available
        
    Returns:
        torch.device: The specified device or fallback
    """
    if not device_str:
        device = get_default_device()
    else:
        device = parse_device_str(device_str)
    
    # Ensure CUDA is available if requested
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        if fallback_to_cpu:
            print("Warning: CUDA requested but not available. Using CPU instead.")
            return torch.device("cpu")
        else:
            raise RuntimeError("CUDA requested but not available, and fallback disabled.")
    
    return device


def get_device_info(device: Optional[torch.device] = None) -> dict:
    """
    Get detailed information about a device.
    
    Args:
        device: The device to get info about (uses default if None)
        
    Returns:
        dict: Device information including type, name, memory, etc.
    """
    if device is None:
        device = get_default_device()
    
    info = {"type": device.type}
    
    if device.type == "cuda":
        info.update({
            "name": torch.cuda.get_device_name(device),
            "index": device.index if hasattr(device, "index") else 0,
            "total_memory_gb": torch.cuda.get_device_properties(device).total_memory / (1024**3),
            "device_count": torch.cuda.device_count(),
        })
    
    return info


def print_device_info(device: Optional[torch.device] = None) -> None:
    """
    Print formatted information about a device.
    
    Args:
        device: The device to get info about (uses default if None)
    """
    if device is None:
        device = get_default_device()
    
    info = get_device_info(device)
    
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"  Device name: {info['name']}")
        print(f"  Memory: {info['total_memory_gb']:.2f} GB")
        if torch.cuda.device_count() > 1:
            print(f"  Multi-GPU: {torch.cuda.device_count()} devices available")


def move_to_device(
    data: Union[torch.Tensor, dict, list, tuple],
    device: Optional[torch.device] = None,
    non_blocking: bool = False
) -> Union[torch.Tensor, dict, list, tuple]:
    """
    Recursively move data to the specified device.
    Handles nested dictionaries, lists, and tuples containing tensors.
    
    Args:
        data: The data to move (tensor, dict, list, or tuple)
        device: Target device (uses default if None)
        non_blocking: Whether to use non_blocking transfer
        
    Returns:
        Same type as data, but with tensors moved to the device
    """
    if device is None:
        device = get_default_device()
    
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(x, device, non_blocking) for x in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(x, device, non_blocking) for x in data)
    else:
        return data


def set_cuda_device_environment() -> None:
    """
    Set CUDA device environment variables for better performance.
    Call this once at the beginning of training.
    """
    if torch.cuda.is_available():
        # Allow TensorFloat32 (TF32) on Ampere GPUs for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set to benchmark mode for faster performance with fixed input sizes
        torch.backends.cudnn.benchmark = True
        
        # Set memory allocation strategy for CUDA
        # "cudnn_deterministic" might be slower but gives deterministic results
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"