"""
Optimizer factory and utilities for the enhanced training system.

This module provides factory functions for creating optimizers,
parameter group handling, and weight decay management.
"""

import logging
import inspect
import re
from typing import Dict, List, Optional, Union, Any, Callable, Tuple, Type

import torch
from torch.optim import Optimizer, SGD, Adam, AdamW
from torch.nn import Module

from src.config.training_config import TrainingConfig


# Get logger
logger = logging.getLogger("quantum_resonance")


def get_optimizer_from_name(name: str) -> Type[Optimizer]:
    """
    Get optimizer class by name.
    
    Args:
        name: Name of the optimizer
        
    Returns:
        Optimizer class
    """
    name = name.lower()
    
    # Core PyTorch optimizers
    if name == "sgd":
        return SGD
    elif name == "adam":
        return Adam
    elif name == "adamw":
        return AdamW
    
    # Try to import from torch.optim
    try:
        optimizer_class = getattr(torch.optim, name.upper(), None)
        if optimizer_class is None:
            optimizer_class = getattr(torch.optim, name, None)
        
        if optimizer_class is not None and inspect.isclass(optimizer_class) and issubclass(optimizer_class, Optimizer):
            return optimizer_class
    except Exception:
        pass
    
    # Handle additional optimizers
    try:
        if name == "adafactor":
            # Try to import from transformers
            from transformers.optimization import Adafactor
            return Adafactor
        elif name == "adabelief":
            # Try to import AdaBelief
            from adabelief_pytorch import AdaBelief
            return AdaBelief
        elif name == "lamb":
            # Try to import LAMB
            try:
                from apex.optimizers import FusedLAMB
                return FusedLAMB
            except ImportError:
                from torch_optimizer import Lamb
                return Lamb
    except ImportError:
        logger.warning(f"Failed to import optimizer {name}, trying alternatives")
    
    # Fall back to simple optimizer name matching in torch_optimizer
    try:
        import torch_optimizer
        optimizer_class = getattr(torch_optimizer, name, None)
        if optimizer_class is not None and inspect.isclass(optimizer_class) and issubclass(optimizer_class, Optimizer):
            return optimizer_class
    except ImportError:
        pass
    
    # If we can't find it, default to AdamW
    logger.warning(f"Optimizer {name} not found, defaulting to AdamW")
    return AdamW


def create_param_groups(
    model: Module,
    weight_decay: float = 0.01,
    no_decay_norm_and_bias: bool = True,
    layer_decay: Optional[float] = None,
    custom_parameter_rules: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Create parameter groups for an optimizer with appropriate weight decay settings.
    
    Args:
        model: Model to create parameter groups for
        weight_decay: Weight decay factor
        no_decay_norm_and_bias: Whether to disable weight decay for normalization layers and bias
        layer_decay: Layer-wise learning rate decay factor
        custom_parameter_rules: Custom rules for parameter grouping
        
    Returns:
        List of parameter groups
    """
    # Base parameter groups
    if no_decay_norm_and_bias:
        # Exclude normalization layers and bias from weight decay
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "layer_norm.weight", "norm.weight"]
        decay_and_no_decay = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        param_groups = decay_and_no_decay
    else:
        # Simple parameter group with uniform weight decay
        param_groups = [
            {
                "params": [p for n, p in model.named_parameters()],
                "weight_decay": weight_decay,
            }
        ]
    
    # Apply layer-wise learning rate decay if specified
    if layer_decay is not None and layer_decay < 1.0:
        param_groups = _apply_layer_decay(model, weight_decay, no_decay_norm_and_bias, layer_decay)
    
    # Apply custom parameter rules if provided
    if custom_parameter_rules is not None and len(custom_parameter_rules) > 0:
        param_groups = _apply_custom_parameter_rules(model, custom_parameter_rules)
    
    # Verify that all parameters are included in at least one group
    all_params = set(model.parameters())
    params_in_groups = set()
    
    for group in param_groups:
        params_in_groups.update(group["params"])
    
    if all_params != params_in_groups:
        logger.warning("Some parameters are not included in any parameter group")
    
    return param_groups


def _apply_layer_decay(
    model: Module,
    weight_decay: float = 0.01,
    no_decay_norm_and_bias: bool = True,
    layer_decay: float = 0.9
) -> List[Dict[str, Any]]:
    """
    Apply layer-wise learning rate decay.
    
    Args:
        model: Model to create parameter groups for
        weight_decay: Weight decay factor
        no_decay_norm_and_bias: Whether to disable weight decay for normalization layers and bias
        layer_decay: Layer-wise learning rate decay factor
        
    Returns:
        List of parameter groups with layer-wise learning rate decay
    """
    # Heuristic function to get layer depth
    def get_layer_depth(name: str) -> int:
        if "embeddings" in name:
            return 0
        
        if "encoder.layer" in name or "decoder.layer" in name:
            # Extract layer number for transformer models
            try:
                layer_match = re.search(r"layer\.(\d+)", name)
                if layer_match:
                    return int(layer_match.group(1)) + 1
                
                block_match = re.search(r"block\.(\d+)", name)
                if block_match:
                    return int(block_match.group(1)) + 1
            except:
                pass
        
        # Fallback depth calculation based on dots in name
        depth = name.count(".") + 1
        return min(depth, 16)  # Cap at reasonable value
    
    # Get the maximum layer depth
    all_params = list(model.named_parameters())
    layer_depths = [get_layer_depth(name) for name, _ in all_params]
    max_depth = max(layer_depths) if layer_depths else 0
    
    # For no weight decay parameters
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "layer_norm.weight", "norm.weight"]
    
    # Create parameter groups
    param_groups = []
    
    for depth in range(max_depth + 1):
        # Calculate decay factor based on layer depth
        decay_factor = layer_decay ** (max_depth - depth)
        
        if no_decay_norm_and_bias:
            # With weight decay
            param_groups.append({
                "params": [p for n, p in all_params 
                          if get_layer_depth(n) == depth and not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr_scale": decay_factor
            })
            
            # Without weight decay
            param_groups.append({
                "params": [p for n, p in all_params 
                          if get_layer_depth(n) == depth and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr_scale": decay_factor
            })
        else:
            # Single group per layer
            param_groups.append({
                "params": [p for n, p in all_params if get_layer_depth(n) == depth],
                "weight_decay": weight_decay,
                "lr_scale": decay_factor
            })
    
    # Filter out empty groups
    return [g for g in param_groups if len(g["params"]) > 0]


def _apply_custom_parameter_rules(
    model: Module,
    rules: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Apply custom parameter grouping rules.
    
    Args:
        model: Model to create parameter groups for
        rules: List of custom parameter rules
        
    Returns:
        List of parameter groups based on custom rules
    """
    all_params = list(model.named_parameters())
    applied_params = set()
    param_groups = []
    
    # Apply each rule in order
    for rule in rules:
        pattern = rule.get("pattern", "")
        lr_scale = rule.get("lr_scale", 1.0)
        weight_decay = rule.get("weight_decay", 0.01)
        
        # Find matching parameters
        matching_params = []
        for name, param in all_params:
            if param in applied_params:
                continue
                
            if re.search(pattern, name):
                matching_params.append(param)
                applied_params.add(param)
        
        # Create group if any parameters match
        if matching_params:
            param_groups.append({
                "params": matching_params,
                "weight_decay": weight_decay,
                "lr_scale": lr_scale
            })
    
    # Add remaining parameters
    remaining_params = [p for n, p in all_params if p not in applied_params]
    if remaining_params:
        param_groups.append({
            "params": remaining_params,
            "weight_decay": 0.01,  # Default weight decay
            "lr_scale": 1.0        # Default lr scale
        })
    
    return param_groups


def create_optimizer(
    model: Module,
    config: Union[dict, TrainingConfig],
    optimizer_name: Optional[str] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    **kwargs
) -> Optimizer:
    """
    Create an optimizer instance based on configuration.
    
    Args:
        model: Model to optimize
        config: Training configuration
        optimizer_name: Name of the optimizer to use (overrides config)
        learning_rate: Learning rate (overrides config)
        weight_decay: Weight decay factor (overrides config)
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer instance
    """
    # Get parameters from config if not provided explicitly
    if isinstance(config, dict):
        opt_name = optimizer_name or config.get("optimizer", "adamw")
        lr = learning_rate or config.get("learning_rate", 5e-5)
        wd = weight_decay or config.get("weight_decay", 0.01)
        no_decay_norm_bias = config.get("no_decay_norm_bias", True)
        layer_decay = config.get("layer_decay", None)
        custom_parameter_rules = config.get("custom_parameter_rules", None)
    else:
        opt_name = optimizer_name or getattr(config, "optimizer", "adamw")
        lr = learning_rate or getattr(config, "learning_rate", 5e-5)
        wd = weight_decay or getattr(config, "weight_decay", 0.01)
        no_decay_norm_bias = getattr(config, "no_decay_norm_bias", True)
        layer_decay = getattr(config, "layer_decay", None)
        custom_parameter_rules = getattr(config, "custom_parameter_rules", None)
    
    # Get optimizer class
    optimizer_class = get_optimizer_from_name(opt_name)
    
    # Prepare parameter groups
    param_groups = create_param_groups(
        model,
        weight_decay=wd,
        no_decay_norm_and_bias=no_decay_norm_bias,
        layer_decay=layer_decay,
        custom_parameter_rules=custom_parameter_rules
    )
    
    # Create optimizer instance
    optimizer_args = {"lr": lr}
    
    # Add any additional arguments from kwargs
    optimizer_args.update(kwargs)
    
    # Create and return the optimizer
    optimizer = optimizer_class(param_groups, **optimizer_args)
    
    logger.info(f"Created {opt_name} optimizer with lr={lr}, weight_decay={wd}")
    
    return optimizer


def adjust_learning_rate(
    optimizer: Optimizer,
    new_lr: float
) -> None:
    """
    Adjust the learning rate of an optimizer.
    
    Args:
        optimizer: Optimizer to adjust
        new_lr: New learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr * param_group.get('lr_scale', 1.0)
    
    logger.info(f"Adjusted learning rate to {new_lr}")


def get_optimizer_parameters(optimizer: Optimizer) -> Dict[str, Any]:
    """
    Get parameters of an optimizer.
    
    Args:
        optimizer: Optimizer to get parameters for
        
    Returns:
        Dictionary of optimizer parameters
    """
    params = {
        "type": optimizer.__class__.__name__,
        "learning_rates": [group['lr'] for group in optimizer.param_groups],
        "weight_decays": [group.get('weight_decay', 0.0) for group in optimizer.param_groups],
        "parameter_count": sum(len(group['params']) for group in optimizer.param_groups)
    }
    
    return params


def apply_gradients(
    optimizer: Optimizer,
    grad_scaler: Optional[Any] = None,
    max_grad_norm: float = 1.0,
    **kwargs
) -> float:
    """
    Apply gradients to optimizer with optional gradient clipping and scaling.
    
    Args:
        optimizer: Optimizer to apply gradients to
        grad_scaler: Optional gradient scaler for mixed precision training
        max_grad_norm: Maximum gradient norm for clipping
        **kwargs: Additional arguments
        
    Returns:
        Gradient norm
    """
    grad_norm = 0.0
    
    # Clip gradients if max_grad_norm is specified
    if max_grad_norm > 0:
        if grad_scaler is not None and hasattr(grad_scaler, "unscale_"):
            # Unscale gradients for accurate clipping with AMP
            grad_scaler.unscale_(optimizer)
        
        # Compute gradient norm for all parameters
        parameters = []
        for param_group in optimizer.param_groups:
            parameters.extend([p for p in param_group['params'] if p.grad is not None])
        
        if parameters:
            # Clip gradients in-place
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm).item()
    
    # Apply gradients
    if grad_scaler is not None and hasattr(grad_scaler, "step"):
        # Use gradient scaler for mixed precision
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        # Standard optimizer step
        optimizer.step()
    
    return grad_norm