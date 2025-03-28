"""
Device handling utilities for QLLM.

This module provides consistent device setup and management across training and inference,
including detection and configuration of various hardware accelerators.
"""

import os
import torch
import logging
from typing import Optional, Union, Tuple

# Get logger
logger = logging.getLogger("qllm.utils.device")


def get_default_device() -> torch.device:
    """
    Get the default device (CUDA if available, MPS if available on Apple Silicon, otherwise CPU).
    
    Returns:
        torch.device: The default device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


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
                logger.warning(f"Specified GPU index {index} is out of range. "
                      f"Using default device instead.")
                return get_default_device()
        except (ValueError, IndexError):
            logger.warning(f"Invalid CUDA device format '{device_str}'. "
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
    
    # Ensure requested device is available
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        if fallback_to_cpu:
            logger.warning("CUDA requested but not available. Using fallback device.")
            return get_default_device()
        else:
            raise RuntimeError("CUDA requested but not available, and fallback disabled.")
    elif device.type == "mps" and (not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available()):
        if fallback_to_cpu:
            logger.warning("MPS (Apple Silicon) requested but not available. Using CPU instead.")
            return torch.device("cpu")
        else:
            raise RuntimeError("MPS requested but not available, and fallback disabled.")
    
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
    elif device.type == "mps":
        info.update({
            "name": "Apple Silicon GPU",
            "is_available": torch.backends.mps.is_available(),
            "is_built": torch.backends.mps.is_built(),
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
    
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"  Device name: {info['name']}")
        logger.info(f"  Memory: {info['total_memory_gb']:.2f} GB")
        if torch.cuda.device_count() > 1:
            logger.info(f"  Multi-GPU: {torch.cuda.device_count()} devices available")


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


def get_memory_usage(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Get the current memory usage for a device.
    
    Args:
        device: The device to check memory for (uses default if None)
        
    Returns:
        Dict containing memory usage statistics in GB
    """
    if device is None:
        device = get_default_device()
    
    result = {}
    
    if device.type == "cuda":
        # Get current memory usage
        current = torch.cuda.memory_allocated(device) / (1024**3)
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        
        # Get device properties for total memory
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory / (1024**3)
        
        result = {
            "current_gb": current,
            "peak_gb": peak,
            "reserved_gb": reserved,
            "total_gb": total,
            "utilization_pct": (current / total) * 100
        }
    
    return result


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