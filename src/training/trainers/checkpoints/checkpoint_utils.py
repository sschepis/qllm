"""
Utilities for checkpoint management in the enhanced training system.

This module provides utility functions for checkpoint naming,
state extraction, and checkpoint resumption.
"""

import os
import re
import glob
import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple

import torch


# Get logger
logger = logging.getLogger("quantum_resonance")


def generate_checkpoint_name(
    prefix: str = "checkpoint",
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    extension: str = "pt"
) -> str:
    """
    Generate a descriptive checkpoint filename.
    
    Args:
        prefix: Prefix for the checkpoint name
        epoch: Current epoch
        step: Current step
        metrics: Dictionary of metrics to include in name
        extension: File extension
        
    Returns:
        Generated checkpoint filename
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    components = [prefix, timestamp]
    
    # Add epoch if provided
    if epoch is not None:
        components.append(f"epoch-{epoch}")
    
    # Add step if provided
    if step is not None:
        components.append(f"step-{step}")
    
    # Add metric if provided (only include one key metric)
    if metrics is not None and metrics:
        # Find best metric to include (prefer validation loss if available)
        metric_key = "val_loss"
        if metric_key not in metrics:
            for key in ["loss", "val_accuracy", "val_ppl", "accuracy"]:
                if key in metrics:
                    metric_key = key
                    break
            else:
                # Use first metric if none of the preferred ones are found
                metric_key = next(iter(metrics))
        
        # Format metric value
        metric_value = metrics[metric_key]
        if "loss" in metric_key or "ppl" in metric_key:
            # Lower is better for loss/perplexity metrics
            components.append(f"{metric_key}-{metric_value:.4f}")
        else:
            # Higher is better for accuracy/other metrics
            components.append(f"{metric_key}-{metric_value:.2f}")
    
    # Join components with underscores and add extension
    return f"{'-'.join(components)}.{extension}"


def find_best_checkpoint(
    checkpoint_dir: str,
    metric_name: str = "val_loss",
    higher_better: bool = False,
    include_epoch: bool = False
) -> Optional[str]:
    """
    Find the best checkpoint based on a metric.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        metric_name: Metric to use for ranking
        higher_better: Whether higher metric values are better
        include_epoch: Whether to only consider checkpoints with epoch in name
        
    Returns:
        Path to best checkpoint or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None
    
    # Pattern to extract metrics from checkpoint names
    metric_pattern = re.compile(f"{metric_name}-([0-9.]+)")
    epoch_pattern = re.compile(r"epoch-(\d+)")
    
    # Get all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    
    # Filter by epoch presence if requested
    if include_epoch:
        checkpoint_files = [f for f in checkpoint_files if epoch_pattern.search(os.path.basename(f))]
    
    # Find checkpoints with the specified metric
    best_checkpoint = None
    best_value = float('-inf') if higher_better else float('inf')
    
    for checkpoint_file in checkpoint_files:
        basename = os.path.basename(checkpoint_file)
        
        # Extract metric value
        match = metric_pattern.search(basename)
        if match:
            try:
                value = float(match.group(1))
                
                # Update best checkpoint if this one is better
                if (higher_better and value > best_value) or (not higher_better and value < best_value):
                    best_value = value
                    best_checkpoint = checkpoint_file
            except ValueError:
                # Invalid metric value
                continue
    
    if best_checkpoint:
        logger.info(f"Found best checkpoint: {os.path.basename(best_checkpoint)} with {metric_name}={best_value}")
    else:
        logger.warning(f"No checkpoints found with metric {metric_name}")
    
    return best_checkpoint


def find_latest_checkpoint(
    checkpoint_dir: str,
    prefix: Optional[str] = None
) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Optional prefix to filter checkpoints
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None
    
    # Get all checkpoint files with the specified prefix
    pattern = f"{prefix}*.pt" if prefix else "*.pt"
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not checkpoint_files:
        logger.warning(f"No checkpoints found in {checkpoint_dir}")
        return None
    
    # Find the latest checkpoint by modification time
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    logger.info(f"Found latest checkpoint: {os.path.basename(latest_checkpoint)}")
    
    return latest_checkpoint


def extract_state_dict(
    checkpoint: Dict[str, Any],
    key: str = "model_state_dict"
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Extract a state dictionary from a checkpoint.
    
    Args:
        checkpoint: Checkpoint dictionary
        key: Key for the state dictionary
        
    Returns:
        State dictionary or None if not found
    """
    if not isinstance(checkpoint, dict):
        logger.error(f"Checkpoint is not a dictionary: {type(checkpoint)}")
        return None
    
    # Handle different checkpoint formats
    if key in checkpoint:
        return checkpoint[key]
    elif "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    elif all(k.startswith("module.") for k in checkpoint.keys() if isinstance(k, str)):
        # DataParallel state dict
        return {k[7:]: v for k, v in checkpoint.items() if k.startswith("module.")}
    elif all(isinstance(k, str) and "." in k for k in checkpoint.keys()):
        # Direct model state dict
        return checkpoint
    else:
        logger.warning(f"Could not find state dict in checkpoint with keys: {list(checkpoint.keys())}")
        return None


def fix_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
    target_model: torch.nn.Module,
    prefix_map: Optional[Dict[str, str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Fix keys in a state dictionary to match a target model.
    
    Args:
        state_dict: State dictionary to fix
        target_model: Model to match keys with
        prefix_map: Mapping of key prefixes to their replacements
        
    Returns:
        Fixed state dictionary
    """
    # Create a map of model keys
    target_keys = set(target_model.state_dict().keys())
    
    # Default prefix map
    if prefix_map is None:
        prefix_map = {
            "module.": "",        # Remove DataParallel prefix
            "model.": "",         # Remove model wrapper prefix
            "encoder.": "",       # Handle encoder-only format
            "transformer.": "",   # Handle transformer prefix
        }
    
    # Apply prefix mapping
    fixed_dict = {}
    unmapped_keys = []
    
    for key, value in state_dict.items():
        # Try different prefix mappings
        mapped_key = key
        for prefix, replacement in prefix_map.items():
            if key.startswith(prefix):
                mapped_key = replacement + key[len(prefix):]
                break
        
        # Check if mapped key exists in target model
        if mapped_key in target_keys:
            fixed_dict[mapped_key] = value
        else:
            unmapped_keys.append(key)
    
    # Log unmapped keys
    if unmapped_keys:
        logger.warning(f"Could not map {len(unmapped_keys)}/{len(state_dict)} keys in state dict")
        if len(unmapped_keys) < 10:
            logger.warning(f"Unmapped keys: {unmapped_keys}")
    
    return fixed_dict


def get_checkpoint_info(
    checkpoint_path: str,
    load_weights: bool = False,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Get information about a checkpoint without loading the full model.
    
    Args:
        checkpoint_path: Path to checkpoint file
        load_weights: Whether to load model weights
        device: Device to load weights to
        
    Returns:
        Dictionary of checkpoint information
    """
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return {}
    
    try:
        # Load checkpoint file
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract metadata
        info = {}
        
        # Basic info
        info["filename"] = os.path.basename(checkpoint_path)
        info["last_modified"] = time.ctime(os.path.getmtime(checkpoint_path))
        info["size_mb"] = os.path.getsize(checkpoint_path) / (1024 * 1024)
        
        # Check if it's a dictionary
        if not isinstance(checkpoint, dict):
            logger.warning(f"Checkpoint is not a dictionary: {type(checkpoint)}")
            if load_weights:
                info["state_dict"] = checkpoint
            return info
        
        # Extract common metadata fields
        for key in ["epoch", "global_step", "timestamp", "metrics", "metadata"]:
            if key in checkpoint:
                info[key] = checkpoint[key]
        
        # Get optimizer info
        if "optimizer_state_dict" in checkpoint:
            optim_dict = checkpoint["optimizer_state_dict"]
            info["optimizer"] = {
                "type": optim_dict.get("name", "unknown"),
                "param_groups": len(optim_dict.get("param_groups", [])),
                "has_state": bool(optim_dict.get("state", {}))
            }
        
        # Get scheduler info
        if "scheduler_state_dict" in checkpoint:
            scheduler_dict = checkpoint["scheduler_state_dict"]
            info["scheduler"] = {
                "type": scheduler_dict.get("name", "unknown"),
                "last_epoch": scheduler_dict.get("last_epoch", -1)
            }
        
        # Get model info
        if "model_state_dict" in checkpoint or "state_dict" in checkpoint:
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
            info["model"] = {
                "num_parameters": len(state_dict),
                "parameter_names": list(state_dict.keys())[:5] + ['...']
            }
            
            # Load weights if requested
            if load_weights:
                info["state_dict"] = state_dict
        
        return info
    
    except Exception as e:
        logger.error(f"Error getting checkpoint info: {e}")
        return {"error": str(e)}


def resume_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    checkpoint_path: str = None,
    checkpoint_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Resume training from a checkpoint.
    
    Args:
        model: Model to load checkpoint into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        checkpoint_path: Path to specific checkpoint file
        checkpoint_dir: Directory to search for latest checkpoint
        device: Device to load tensors to
        strict: Whether to strictly enforce parameter names
        
    Returns:
        Dictionary of checkpoint information
    """
    # Determine checkpoint path
    if checkpoint_path is None and checkpoint_dir is not None:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        logger.warning(f"No checkpoint found to resume from")
        return {}
    
    try:
        # Set device
        if device is None:
            device = next(model.parameters()).device
        
        # Load checkpoint
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract state dict
        state_dict = extract_state_dict(checkpoint)
        
        if state_dict is None:
            logger.error(f"Could not extract state dict from checkpoint")
            return {}
        
        # Fix state dict keys if needed
        fixed_state_dict = fix_state_dict_keys(state_dict, model)
        
        # Load state dict
        try:
            model.load_state_dict(fixed_state_dict, strict=strict)
            logger.info(f"Loaded model state from checkpoint")
        except Exception as e:
            logger.error(f"Error loading model state: {e}")
            # Try loading with strict=False as fallback
            if strict:
                logger.info(f"Attempting to load with strict=False")
                model.load_state_dict(fixed_state_dict, strict=False)
                logger.info(f"Loaded model state with strict=False")
        
        # Load optimizer state if available
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info(f"Loaded optimizer state from checkpoint")
            except Exception as e:
                logger.warning(f"Error loading optimizer state: {e}")
        
        # Load scheduler state if available
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info(f"Loaded scheduler state from checkpoint")
            except Exception as e:
                logger.warning(f"Error loading scheduler state: {e}")
        
        # Extract metadata
        metadata = {}
        for key in ["epoch", "global_step", "metrics", "metadata"]:
            if key in checkpoint:
                metadata[key] = checkpoint[key]
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error resuming from checkpoint: {e}")
        return {"error": str(e)}


def get_checkpoint_metrics(
    checkpoint_dir: str,
    sort_by: str = "step"
) -> List[Dict[str, Any]]:
    """
    Get metrics from all checkpoints in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        sort_by: Field to sort results by
        
    Returns:
        List of checkpoint metrics
    """
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return []
    
    # Get all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    
    # Extract metrics from each checkpoint
    metrics_list = []
    
    for checkpoint_file in checkpoint_files:
        try:
            info = get_checkpoint_info(checkpoint_file)
            
            # Add filename and basic info
            metrics_entry = {
                "filename": os.path.basename(checkpoint_file),
                "last_modified": info.get("last_modified", ""),
                "size_mb": info.get("size_mb", 0)
            }
            
            # Add training progress
            metrics_entry["epoch"] = info.get("epoch", 0)
            metrics_entry["step"] = info.get("global_step", 0)
            
            # Add metrics if available
            if "metrics" in info:
                for key, value in info["metrics"].items():
                    metrics_entry[key] = value
            
            metrics_list.append(metrics_entry)
        
        except Exception as e:
            logger.warning(f"Error extracting metrics from {os.path.basename(checkpoint_file)}: {e}")
    
    # Sort by the specified field
    if metrics_list and sort_by in metrics_list[0]:
        metrics_list.sort(key=lambda x: x.get(sort_by, 0))
    
    return metrics_list