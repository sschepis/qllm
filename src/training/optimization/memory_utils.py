"""
Memory optimization utilities for the enhanced training system.

This module provides utilities for optimizing memory usage during training,
including gradient checkpointing, batch size optimization, and handling
out-of-memory conditions.
"""

import os
import logging
import math
from typing import Dict, List, Optional, Union, Any, Callable, Tuple

import torch
import torch.nn as nn


# Get logger
logger = logging.getLogger("quantum_resonance")


def enable_gradient_checkpointing(
    model: nn.Module,
    checkpoint_modules: Optional[List[str]] = None
) -> nn.Module:
    """
    Enable gradient checkpointing for a model to save memory.
    
    Args:
        model: Model to enable gradient checkpointing for
        checkpoint_modules: List of module names to enable checkpointing for (default: all supported modules)
        
    Returns:
        Model with gradient checkpointing enabled
    """
    # Default modules to checkpoint
    if checkpoint_modules is None:
        checkpoint_modules = [
            "transformer", "encoder", "decoder", "layers", 
            "blocks", "resblocks", "attentions", "mlp"
        ]
    
    logger.info("Enabling gradient checkpointing to reduce memory usage")
    
    # Track modules with checkpointing enabled
    checkpointed_modules = []
    
    # Function to recursively enable checkpointing for modules
    def _enable_checkpointing(module: nn.Module, name: str) -> None:
        # Check if module name contains any of the checkpoint modules
        should_checkpoint = any(checkpoint_name in name.lower() for checkpoint_name in checkpoint_modules)
        
        # Check if module has gradient checkpointing
        if should_checkpoint:
            # Handle different model architectures
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()
                checkpointed_modules.append(name)
            elif hasattr(module, "enable_gradient_checkpointing"):
                module.enable_gradient_checkpointing()
                checkpointed_modules.append(name)
            elif hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
                checkpointed_modules.append(name)
            # Check if we can add a custom checkpointing implementation
            elif hasattr(module, "forward") and "transformer" in name.lower():
                original_forward = module.forward
                
                # Create checkpointed forward
                def checkpointed_forward(*args, **kwargs):
                    return torch.utils.checkpoint.checkpoint(
                        original_forward, *args, **kwargs
                    )
                
                # Replace forward method
                module.forward = checkpointed_forward
                checkpointed_modules.append(name)
        
        # Recurse through child modules
        for child_name, child_module in module.named_children():
            _enable_checkpointing(child_module, f"{name}.{child_name}")
    
    # Enable checkpointing for model
    for name, module in model.named_children():
        _enable_checkpointing(module, name)
    
    if checkpointed_modules:
        logger.info(f"Gradient checkpointing enabled for {len(checkpointed_modules)} modules")
    else:
        logger.warning("No modules found for gradient checkpointing")
    
    return model


def optimize_memory_for_evaluation(
    model: nn.Module,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    max_memory_gb: float = 0.8
) -> Tuple[nn.Module, int]:
    """
    Optimize memory usage for model evaluation.
    
    Args:
        model: Model to optimize
        batch_size: Current batch size
        sequence_length: Sequence length for inputs
        vocab_size: Vocabulary size
        max_memory_gb: Maximum GPU memory to use (in GB, as a fraction of total)
        
    Returns:
        Tuple of (optimized model, optimized batch size)
    """
    logger.info("Optimizing memory usage for evaluation")
    
    # Enable eval mode to reduce memory usage
    model.eval()
    
    # Calculate available GPU memory
    available_memory = get_available_gpu_memory()
    max_memory = int(get_total_gpu_memory() * max_memory_gb)
    
    logger.info(f"GPU memory - Available: {available_memory:.2f} GB, Target max: {max_memory:.2f} GB")
    
    # Enable gradient checkpointing for memory efficiency (even in eval mode)
    model = enable_gradient_checkpointing(model)
    
    # Disable model parts that aren't needed for evaluation
    model = optimize_model_memory(model)
    
    # Calculate memory needed per sample (rough estimate)
    # Memory for activations: batch_size * sequence_length * hidden_size * 4 bytes (float32)
    # Memory for logits: batch_size * sequence_length * vocab_size * 4 bytes (float32)
    hidden_size = estimate_model_hidden_size(model)
    
    # Memory per sample in GB
    memory_per_sample = (
        (sequence_length * hidden_size * 4) +  # Activations
        (sequence_length * vocab_size * 4)     # Logits
    ) / (1024 ** 3)  # Convert to GB
    
    # Calculate optimal batch size
    target_memory = min(available_memory * 0.8, max_memory)
    optimal_batch_size = max(1, int(target_memory / memory_per_sample))
    
    # Limit batch size change to prevent extreme reductions
    optimized_batch_size = max(1, min(batch_size, optimal_batch_size))
    
    if optimized_batch_size < batch_size:
        logger.info(f"Reduced batch size for evaluation: {batch_size} -> {optimized_batch_size}")
    
    return model, optimized_batch_size


def optimize_model_memory(model: nn.Module) -> nn.Module:
    """
    Optimize model memory usage by disabling unnecessary components.
    
    Args:
        model: Model to optimize
        
    Returns:
        Memory-optimized model
    """
    # Disable dropout during evaluation
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0
    
    # Set eval mode to disable batch normalization stats
    model.eval()
    
    # Force empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model


def estimate_model_hidden_size(model: nn.Module) -> int:
    """
    Estimate the hidden size of a model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Estimated hidden size
    """
    hidden_size = 768  # Default for base models
    
    # Try to get hidden size from model config
    if hasattr(model, "config"):
        config = getattr(model, "config")
        for attr in ["hidden_size", "d_model", "n_embd", "dim"]:
            if hasattr(config, attr):
                hidden_size = getattr(config, attr)
                break
    
    # Try to infer from model parameters
    else:
        # Look for embedding or attention modules with dimension attributes
        for name, module in model.named_modules():
            if "embed" in name or "attention" in name:
                for attr in ["embedding_dim", "hidden_size", "d_model", "embed_dim"]:
                    if hasattr(module, attr):
                        hidden_size = getattr(module, attr)
                        break
    
    return hidden_size


def get_available_gpu_memory() -> float:
    """
    Get available GPU memory in GB.
    
    Returns:
        Available GPU memory in GB
    """
    if torch.cuda.is_available():
        # Get GPU device
        device = torch.cuda.current_device()
        
        # Get memory information
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # GB
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
        total = get_total_gpu_memory()
        
        # Calculate available memory
        available = total - allocated
        
        return available
    else:
        return 0.0


def get_total_gpu_memory() -> float:
    """
    Get total GPU memory in GB.
    
    Returns:
        Total GPU memory in GB
    """
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory / (1024 ** 3)  # GB
        return total
    else:
        return 0.0


def print_memory_stats() -> Dict[str, float]:
    """
    Print memory usage statistics.
    
    Returns:
        Dictionary of memory statistics
    """
    stats = {}
    
    if torch.cuda.is_available():
        # Get GPU stats
        device = torch.cuda.current_device()
        stats["gpu_total_gb"] = get_total_gpu_memory()
        stats["gpu_reserved_gb"] = torch.cuda.memory_reserved(device) / (1024 ** 3)
        stats["gpu_allocated_gb"] = torch.cuda.memory_allocated(device) / (1024 ** 3)
        stats["gpu_available_gb"] = stats["gpu_total_gb"] - stats["gpu_allocated_gb"]
        
        # Log GPU stats
        logger.info(f"GPU Memory: {stats['gpu_allocated_gb']:.2f}GB allocated / "
                   f"{stats['gpu_total_gb']:.2f}GB total "
                   f"({stats['gpu_available_gb']:.2f}GB available)")
    
    # Get CPU stats
    try:
        import psutil
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss / (1024 ** 3)  # GB
        stats["cpu_memory_gb"] = cpu_memory
        logger.info(f"CPU Memory: {cpu_memory:.2f}GB")
    except ImportError:
        logger.warning("psutil not installed, CPU memory stats not available")
    
    return stats


def try_batch_optimization(func: Callable) -> Callable:
    """
    Decorator to attempt batch size optimization on OOM errors.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function with batch optimization
    """
    def wrapper(*args, **kwargs):
        # Extract batch if available
        batch = None
        if "batch" in kwargs:
            batch = kwargs["batch"]
        elif len(args) > 1 and isinstance(args[1], (dict, torch.Tensor)):
            batch = args[1]
        
        # Original batch size
        batch_size = None
        if isinstance(batch, dict) and "input_ids" in batch:
            batch_size = batch["input_ids"].size(0)
        elif isinstance(batch, torch.Tensor):
            batch_size = batch.size(0)
        
        # Try with original batch size
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            if batch_size is None or batch_size <= 1:
                # Can't reduce batch size further
                logger.error("CUDA out of memory with minimum batch size, can't optimize further")
                torch.cuda.empty_cache()
                raise e
            
            # Log the error
            logger.warning(f"CUDA out of memory with batch size {batch_size}, attempting optimization")
            torch.cuda.empty_cache()
            
            # Reduce batch size by half
            new_batch_size = max(1, batch_size // 2)
            logger.info(f"Reducing batch size: {batch_size} -> {new_batch_size}")
            
            # Create reduced batch
            reduced_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.size(0) == batch_size:
                    reduced_batch[key] = value[:new_batch_size]
                else:
                    reduced_batch[key] = value
            
            # Update args or kwargs
            if "batch" in kwargs:
                kwargs["batch"] = reduced_batch
            elif len(args) > 1 and isinstance(args[1], (dict, torch.Tensor)):
                args_list = list(args)
                args_list[1] = reduced_batch
                args = tuple(args_list)
            
            # Try again with reduced batch
            return func(*args, **kwargs)
    
    return wrapper


def handle_cuda_oom(
    model: nn.Module,
    error: Exception,
    reduce_batch: bool = True,
    enable_checkpointing: bool = True,
    offload_to_cpu: bool = False
) -> None:
    """
    Handle CUDA out of memory errors with various strategies.
    
    Args:
        model: Model that caused the error
        error: The error that occurred
        reduce_batch: Whether to suggest batch size reduction
        enable_checkpointing: Whether to suggest gradient checkpointing
        offload_to_cpu: Whether to suggest CPU offloading
    """
    # Get memory stats
    memory_stats = print_memory_stats()
    
    # Log error with detailed memory information
    logger.error(f"CUDA out of memory error: {str(error)}")
    logger.error(f"Memory stats at time of error: {memory_stats}")
    
    # Suggest specific optimizations
    suggestions = []
    
    if reduce_batch:
        suggestions.append(
            "Reduce batch size or sequence length to decrease memory usage"
        )
    
    if enable_checkpointing:
        suggestions.append(
            "Enable gradient checkpointing: model.gradient_checkpointing_enable()"
        )
    
    if offload_to_cpu:
        suggestions.append(
            "Offload optimizer states to CPU with optimizer options: 'capturable=True, offload_to_cpu=True'"
        )
    
    # GPU-specific suggestions
    if torch.cuda.is_available():
        suggestions.append(
            "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True environment variable to avoid fragmentation"
        )
        
        # Check if mixed precision is an option
        if torch.cuda.get_device_capability()[0] >= 7:  # Volta or newer
            suggestions.append(
                "Enable mixed precision training with torch.cuda.amp.autocast"
            )
    
    # Log suggestions
    if suggestions:
        logger.info("Suggestions to address memory issue:")
        for i, suggestion in enumerate(suggestions, 1):
            logger.info(f"  {i}. {suggestion}")
    
    # Try to free memory
    torch.cuda.empty_cache()
    
    # Throw the error again
    raise error


def calculate_max_batch_size(
    model: nn.Module,
    sample_input: Dict[str, torch.Tensor],
    max_memory_fraction: float = 0.8,
    start_batch_size: int = 1,
    growth_factor: float = 1.5,
    max_iterations: int = 10
) -> int:
    """
    Calculate maximum batch size that fits in memory.
    
    Args:
        model: Model to test
        sample_input: Sample input dictionary with batch_size=1
        max_memory_fraction: Maximum fraction of GPU memory to use
        start_batch_size: Starting batch size for testing
        growth_factor: Factor to grow batch size by in each iteration
        max_iterations: Maximum number of iterations to try
        
    Returns:
        Maximum batch size that fits in memory
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, returning default batch size")
        return start_batch_size
    
    # Put model in eval mode
    model.eval()
    
    # Move model to GPU
    device = torch.cuda.current_device()
    model.to(device)
    
    # Get total GPU memory
    total_memory = get_total_gpu_memory()
    max_memory = total_memory * max_memory_fraction
    
    # Start with small batch size
    batch_size = start_batch_size
    max_batch_size = start_batch_size
    
    # Free memory
    torch.cuda.empty_cache()
    
    # Try progressively larger batch sizes
    for _ in range(max_iterations):
        try:
            # Clone and resize sample input
            batch = {
                k: torch.cat([v] * batch_size, dim=0) if isinstance(v, torch.Tensor) and v.dim() > 0 else v
                for k, v in sample_input.items()
            }
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Run model with batch
            with torch.no_grad():
                _ = model(**batch)
            
            # If successful, increase batch size
            max_batch_size = batch_size
            prev_batch_size = batch_size
            batch_size = int(batch_size * growth_factor)
            
            # Check if we've reached or exceeded the maximum
            current_usage = torch.cuda.memory_allocated(device) / (1024 ** 3)
            memory_usage_ratio = current_usage / total_memory
            
            logger.info(f"Batch size {prev_batch_size} successful - "
                       f"Memory usage: {current_usage:.2f}GB ({memory_usage_ratio:.1%})")
            
            if memory_usage_ratio > max_memory_fraction * 0.8:
                logger.info(f"Memory usage approaching limit, stopping at batch size {max_batch_size}")
                break
            
            # Free memory before next iteration
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            # If we encounter OOM, we've found the limit
            logger.info(f"Batch size {batch_size} causes OOM, maximum is {max_batch_size}")
            torch.cuda.empty_cache()
            break
    
    return max_batch_size