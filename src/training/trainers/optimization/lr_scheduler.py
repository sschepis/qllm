"""
Learning rate scheduler factory and utilities for the enhanced training system.

This module provides factory functions for creating learning rate schedulers,
including warmup schedulers and custom scheduling functions.
"""

import math
import logging
import inspect
from typing import Dict, List, Optional, Union, Any, Callable, Type

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler, 
    LambdaLR, 
    StepLR, 
    MultiStepLR,
    ExponentialLR, 
    CosineAnnealingLR, 
    ReduceLROnPlateau
)

from src.config.training_config import TrainingConfig


# Get logger
logger = logging.getLogger("quantum_resonance")


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with linear warmup and decay.
    
    Args:
        optimizer: Optimizer to use
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: Last epoch to resume from
        
    Returns:
        LambdaLR: Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with cosine annealing and warmup.
    
    Args:
        optimizer: Optimizer to use
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles
        last_epoch: Last epoch to resume from
        
    Returns:
        LambdaLR: Learning rate scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with a constant learning rate after warmup.
    
    Args:
        optimizer: Optimizer to use
        num_warmup_steps: Number of warmup steps
        last_epoch: Last epoch to resume from
        
    Returns:
        LambdaLR: Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float = 1e-7,
    power: float = 1.0,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with polynomial decay and warmup.
    
    Args:
        optimizer: Optimizer to use
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        lr_end: Minimum learning rate
        power: Power factor for polynomial decay
        last_epoch: Last epoch to resume from
        
    Returns:
        LambdaLR: Learning rate scheduler
    """
    lr_init = optimizer.defaults["lr"]
    assert lr_init > lr_end, f"Initial learning rate ({lr_init}) must be higher than end learning rate ({lr_end})"
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        if current_step > num_training_steps:
            return lr_end / lr_init  # Return minimum rate
        
        # Polynomial decay
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining ** power + lr_end
        return decay / lr_init
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler_from_name(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
) -> LRScheduler:
    """
    Get scheduler by name.
    
    Args:
        name: Name of the scheduler
        optimizer: Optimizer to use
        num_warmup_steps: Number of warmup steps (for schedulers with warmup)
        num_training_steps: Total number of training steps (for some schedulers)
        **kwargs: Additional arguments for the scheduler
        
    Returns:
        Learning rate scheduler
    """
    name = name.lower().replace("_", "").replace("-", "")
    
    # Schedulers with warmup
    if "linear" in name and "warmup" in name:
        if num_warmup_steps is None or num_training_steps is None:
            raise ValueError("num_warmup_steps and num_training_steps must be specified for linear warmup scheduler")
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, **kwargs
        )
    elif "cosine" in name and "warmup" in name:
        if num_warmup_steps is None or num_training_steps is None:
            raise ValueError("num_warmup_steps and num_training_steps must be specified for cosine warmup scheduler")
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, **kwargs
        )
    elif "constant" in name and "warmup" in name:
        if num_warmup_steps is None:
            raise ValueError("num_warmup_steps must be specified for constant warmup scheduler")
        return get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps, **kwargs
        )
    elif "polynomial" in name and "warmup" in name:
        if num_warmup_steps is None or num_training_steps is None:
            raise ValueError("num_warmup_steps and num_training_steps must be specified for polynomial warmup scheduler")
        return get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, **kwargs
        )
    
    # Standard PyTorch schedulers
    if name == "step":
        step_size = kwargs.pop("step_size", 30)
        gamma = kwargs.pop("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma, **kwargs)
    elif name == "multistep":
        milestones = kwargs.pop("milestones", [30, 60, 90])
        gamma = kwargs.pop("gamma", 0.1)
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma, **kwargs)
    elif name == "exponential":
        gamma = kwargs.pop("gamma", 0.95)
        return ExponentialLR(optimizer, gamma=gamma, **kwargs)
    elif name == "cosine" or name == "cosineannealinglr":
        if "T_max" not in kwargs and num_training_steps is not None:
            kwargs["T_max"] = num_training_steps
        elif "T_max" not in kwargs:
            kwargs["T_max"] = 1000  # Default value
        return CosineAnnealingLR(optimizer, **kwargs)
    elif name == "reducelronplateau":
        mode = kwargs.pop("mode", "min")
        factor = kwargs.pop("factor", 0.1)
        patience = kwargs.pop("patience", 10)
        threshold = kwargs.pop("threshold", 1e-4)
        return ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold, **kwargs
        )
    
    # Try to get from PyTorch
    try:
        scheduler_class = getattr(torch.optim.lr_scheduler, name.upper(), None)
        if scheduler_class is None:
            scheduler_class = getattr(torch.optim.lr_scheduler, name, None)
        
        if scheduler_class is not None and inspect.isclass(scheduler_class):
            return scheduler_class(optimizer, **kwargs)
    except Exception:
        pass
    
    # If not found, default to linear warmup
    logger.warning(f"Scheduler {name} not found, defaulting to linear warmup scheduler")
    if num_warmup_steps is None or num_training_steps is None:
        num_warmup_steps = kwargs.get("num_warmup_steps", 0)
        num_training_steps = kwargs.get("num_training_steps", 1000)
        logger.warning(f"Using default values: num_warmup_steps={num_warmup_steps}, num_training_steps={num_training_steps}")
    
    return get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, **kwargs
    )


def create_scheduler(
    optimizer: Optimizer,
    config: Union[dict, TrainingConfig],
    scheduler_name: Optional[str] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
) -> LRScheduler:
    """
    Create a learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to use
        config: Training configuration
        scheduler_name: Name of the scheduler to use (overrides config)
        num_training_steps: Total number of training steps (overrides config)
        **kwargs: Additional scheduler arguments
        
    Returns:
        Configured learning rate scheduler
    """
    # Get parameters from config if not provided explicitly
    if isinstance(config, dict):
        name = scheduler_name or config.get("scheduler", "linear_warmup")
        num_steps = num_training_steps or config.get("num_training_steps", None)
        num_warmup_steps = config.get("warmup_steps", 0)
        if isinstance(num_warmup_steps, float) and 0.0 <= num_warmup_steps < 1.0 and num_steps is not None:
            # Handle warmup as a fraction of total steps
            num_warmup_steps = int(num_steps * num_warmup_steps)
    else:
        name = scheduler_name or getattr(config, "scheduler", "linear_warmup")
        num_steps = num_training_steps or getattr(config, "num_training_steps", None)
        num_warmup_steps = getattr(config, "warmup_steps", 0)
        if isinstance(num_warmup_steps, float) and 0.0 <= num_warmup_steps < 1.0 and num_steps is not None:
            # Handle warmup as a fraction of total steps
            num_warmup_steps = int(num_steps * num_warmup_steps)
    
    # Create scheduler
    scheduler = get_scheduler_from_name(
        name=name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_steps,
        **kwargs
    )
    
    logger.info(f"Created {name} scheduler - warmup steps: {num_warmup_steps}, training steps: {num_steps}")
    
    return scheduler


def get_scheduler_warmup_ratio(scheduler: LRScheduler) -> float:
    """
    Get the warmup ratio from a scheduler.
    
    Args:
        scheduler: Learning rate scheduler
        
    Returns:
        Warmup ratio (0.0 to 1.0) or 0.0 if not applicable
    """
    if not hasattr(scheduler, "lr_lambdas") or len(scheduler.lr_lambdas) == 0:
        return 0.0
    
    # Try to extract from scheduler object
    scheduler_dict = scheduler.__dict__
    if "num_warmup_steps" in scheduler_dict and "num_training_steps" in scheduler_dict:
        warmup_steps = scheduler_dict["num_warmup_steps"]
        total_steps = scheduler_dict["num_training_steps"]
        if total_steps > 0:
            return warmup_steps / total_steps
    
    return 0.0


def get_current_lr(scheduler: LRScheduler) -> Union[float, List[float]]:
    """
    Get the current learning rate from a scheduler.
    
    Args:
        scheduler: Learning rate scheduler
        
    Returns:
        Current learning rate or list of learning rates
    """
    if isinstance(scheduler, LambdaLR):
        return [group["lr"] for group in scheduler.optimizer.param_groups]
    elif hasattr(scheduler, "get_last_lr"):
        return scheduler.get_last_lr()
    else:
        return [group["lr"] for group in scheduler.optimizer.param_groups]


def get_scheduler_info(scheduler: LRScheduler) -> Dict[str, Any]:
    """
    Get information about a scheduler.
    
    Args:
        scheduler: Learning rate scheduler
        
    Returns:
        Dictionary with scheduler information
    """
    info = {
        "type": scheduler.__class__.__name__,
        "current_lr": get_current_lr(scheduler),
    }
    
    # Add warmup info if available
    warmup_ratio = get_scheduler_warmup_ratio(scheduler)
    if warmup_ratio > 0:
        info["warmup_ratio"] = warmup_ratio
    
    return info


def get_warmup_steps(
    num_training_steps: int,
    warmup_ratio: float = 0.1,
    min_warmup_steps: int = 100
) -> int:
    """
    Calculate number of warmup steps based on total training steps.
    
    Args:
        num_training_steps: Total number of training steps
        warmup_ratio: Ratio of warmup steps to total steps
        min_warmup_steps: Minimum number of warmup steps
        
    Returns:
        Number of warmup steps
    """
    warmup_steps = int(num_training_steps * warmup_ratio)
    return max(warmup_steps, min_warmup_steps)