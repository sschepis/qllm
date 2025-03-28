"""
Checkpoint utilities for QLLM.

This module provides utilities for saving and loading model checkpoints with
error handling, different formats, and recovery options.
"""

import os
import re
import shutil
import pickle
import logging
from typing import Optional, Dict, Tuple, Any, Union, List

import torch
from torch.nn import Module
from torch.optim import Optimizer

# Get logger
logger = logging.getLogger("qllm.utils.checkpoint")


def get_disk_space(path: str) -> Tuple[float, float, float]:
    """
    Get available disk space for a given path.
    
    Args:
        path: Directory path to check
        
    Returns:
        Tuple of (total_space_mb, used_space_mb, free_space_mb)
    """
    try:
        disk_stats = shutil.disk_usage(path)
        total_mb = disk_stats.total / (1024 * 1024)
        free_mb = disk_stats.free / (1024 * 1024)
        used_mb = (disk_stats.total - disk_stats.free) / (1024 * 1024)
        return (total_mb, used_mb, free_mb)
    except Exception as e:
        logger.warning(f"Could not check disk space for {path}: {e}")
        return (0, 0, 0)


def is_likely_state_dict(data: Dict[str, Any]) -> bool:
    """
    Check if a dictionary is likely to be a state dict.
    
    Args:
        data: Dictionary to check
        
    Returns:
        bool: True if it looks like a state dict
    """
    if not isinstance(data, dict):
        return False
    
    # Count tensor items (state dicts typically have many tensor parameters)
    tensor_count = sum(1 for v in data.values() if isinstance(v, torch.Tensor))
    
    # If it has more than 10 tensor parameters, it's likely a state dict
    return tensor_count > 10


def find_latest_checkpoint(
    directory: str,
    patterns: Optional[List[str]] = None,
    epoch_regex: str = r'epoch_(\d+)'
) -> Optional[str]:
    """
    Find the latest checkpoint in a directory based on filename patterns.
    
    Args:
        directory: Directory to search in
        patterns: List of filename patterns to match (glob-style)
        epoch_regex: Regex pattern to extract epoch numbers
        
    Returns:
        str: Path to the latest checkpoint, or None if not found
    """
    import glob
    
    if not os.path.exists(directory):
        logger.warning(f"Checkpoint directory {directory} does not exist")
        return None
    
    # Default patterns for different checkpoint types
    if patterns is None:
        patterns = [
            "checkpoint_epoch_*.pt",      # Standard checkpoints
            "model_only_epoch_*.pt",      # Model-only fallbacks
            "emergency_save_epoch_*.pt"   # Emergency saves
        ]
    
    # Find all matching files
    checkpoint_files = []
    for pattern in patterns:
        full_pattern = os.path.join(directory, pattern)
        checkpoint_files.extend(glob.glob(full_pattern))
    
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers using regex
    with_epochs = []
    for filepath in checkpoint_files:
        match = re.search(epoch_regex, filepath)
        if match:
            try:
                epoch = int(match.group(1))
                with_epochs.append((filepath, epoch))
            except (ValueError, IndexError):
                # If epoch extraction fails, still include file
                with_epochs.append((filepath, -1))
        else:
            with_epochs.append((filepath, -1))
    
    # Sort by epoch number (descending)
    with_epochs.sort(key=lambda x: x[1], reverse=True)
    
    # Return the file with the highest epoch
    return with_epochs[0][0] if with_epochs else None


def try_load_checkpoint(
    path: str,
    map_location: Optional[Union[torch.device, str]] = None,
    strict: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Try multiple methods to load a checkpoint with robust error handling.
    
    Args:
        path: Path to the checkpoint file
        map_location: Device to map tensors to
        strict: Whether to raise errors or return None on failure
        
    Returns:
        dict: Loaded checkpoint data or None if failed and strict=False
    """
    if not os.path.exists(path):
        if strict:
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        logger.warning(f"Checkpoint file not found: {path}")
        return None
    
    # Try standard PyTorch load first
    try:
        logger.info(f"Loading checkpoint from {path}")
        return torch.load(path, map_location=map_location)
    except Exception as e:
        logger.warning(f"Standard loading failed for {path}: {e}")
        
        # Try with CPU map location as fallback
        try:
            if map_location is None or str(map_location) != "cpu":
                logger.info("Retrying with CPU map location")
                return torch.load(path, map_location="cpu")
        except Exception as e:
            logger.warning(f"CPU map location loading failed: {e}")
        
        # Try pickle loading with higher protocol version tolerance
        try:
            logger.info("Trying pickle-based loading")
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Pickle loading failed: {e}")
            
        # Try with torch.jit._load_for_lite_interpreter
        try:
            logger.info("Trying light interpreter loading")
            import torch.jit
            return torch.jit._load_for_lite_interpreter(path)
        except Exception as e:
            logger.warning(f"Light interpreter loading failed: {e}")
        
        # All methods failed
        if strict:
            raise RuntimeError(f"All loading methods failed for {path}")
        
        logger.error(f"All loading methods failed for {path}")
        return None


def get_checkpoint_type(checkpoint_path: str) -> str:
    """
    Determine checkpoint type based on filename.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        str: One of "full", "model_only", "emergency", or "unknown"
    """
    basename = os.path.basename(checkpoint_path)
    
    if "checkpoint_epoch_" in basename:
        return "full"
    elif "model_only_epoch_" in basename:
        return "model_only"
    elif "emergency_save_epoch_" in basename:
        return "emergency"
    else:
        return "unknown"


def extract_model_state(
    checkpoint: Any,
    checkpoint_type: str = "unknown"
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Extract model state dict from a checkpoint.
    
    Args:
        checkpoint: Loaded checkpoint data
        checkpoint_type: Type of checkpoint ("full", "model_only", etc.)
        
    Returns:
        dict: Model state dict or None if not found
    """
    if checkpoint is None:
        return None
    
    # For full checkpoints, model is in model_state_dict
    if checkpoint_type == "full":
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
    
    # For model_only and emergency, the checkpoint might be the state dict directly
    if checkpoint_type in ["model_only", "emergency"]:
        if is_likely_state_dict(checkpoint):
            return checkpoint
    
    # Unknown format - try different approaches
    if isinstance(checkpoint, dict):
        # Try common keys
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint:
                return checkpoint[key]
        
        # Check if the dict itself looks like a state dict
        if is_likely_state_dict(checkpoint):
            return checkpoint
    
    # If it's a model object that has state_dict method
    if hasattr(checkpoint, "state_dict") and callable(checkpoint.state_dict):
        return checkpoint.state_dict()
    
    return None


def get_checkpoint_epoch(
    checkpoint_path: str,
    checkpoint_data: Optional[Dict[str, Any]] = None
) -> int:
    """
    Get epoch number from checkpoint path or data.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        checkpoint_data: Loaded checkpoint data (optional)
        
    Returns:
        int: Epoch number or 0 if not found
    """
    # Try to get from filename
    match = re.search(r'epoch_(\d+)', checkpoint_path)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, IndexError):
            pass
    
    # Try to get from checkpoint data
    if checkpoint_data is not None and isinstance(checkpoint_data, dict):
        if "epoch" in checkpoint_data:
            try:
                return int(checkpoint_data["epoch"])
            except (ValueError, TypeError):
                pass
    
    # Default to 0
    return 0


def save_checkpoint(
    model: Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    loss: float = 0.0,
    metrics: Optional[Dict[str, Any]] = None,
    output_dir: str = "checkpoints",
    filename: Optional[str] = None,
    save_optimizer: bool = True,
    save_scheduler: bool = True,
    add_metadata: bool = True,
    backup_existing: bool = True,
    min_free_space_mb: float = 500.0,
    emergency_save_dir: Optional[str] = None
) -> str:
    """
    Save checkpoint with robust error handling and fallbacks.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save
        epoch: Current epoch number
        loss: Current loss value
        metrics: Additional metrics to save
        output_dir: Directory to save checkpoint to
        filename: Specific filename (default: checkpoint_epoch_{epoch}.pt)
        save_optimizer: Whether to save optimizer state
        save_scheduler: Whether to save scheduler state
        add_metadata: Whether to add metadata to checkpoint
        backup_existing: Whether to backup existing checkpoint
        min_free_space_mb: Minimum free space required in MB
        emergency_save_dir: Alternative directory for emergency saving

    Returns:
        str: Path to saved checkpoint
    
    Raises:
        RuntimeError: If all save methods fail
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default filename if not provided
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pt"
    
    # Full path to checkpoint
    checkpoint_path = os.path.join(output_dir, filename)
    
    # Check disk space
    total_mb, used_mb, free_mb = get_disk_space(output_dir)
    logger.info(f"Available disk space: {free_mb:.2f} MB of {total_mb:.2f} MB")
    
    if free_mb < min_free_space_mb:
        logger.warning(f"Low disk space warning: {free_mb:.2f} MB available")
    
    # Backup existing checkpoint if it exists
    if os.path.exists(checkpoint_path) and backup_existing:
        try:
            backup_path = f"{checkpoint_path}.bak"
            logger.info(f"Backing up existing checkpoint to {backup_path}")
            shutil.copy2(checkpoint_path, backup_path)
        except Exception as e:
            logger.warning(f"Failed to backup existing checkpoint: {e}")
    
    # Create checkpoint data
    checkpoint_data = {}
    
    # Always save model state
    try:
        logger.info("Saving model state...")
        model_state = model.state_dict()
        checkpoint_data["model_state_dict"] = model_state
    except Exception as e:
        logger.error(f"Error saving model state: {e}")
        raise RuntimeError(f"Could not save model state: {e}")
    
    # Save optimizer if requested
    if optimizer is not None and save_optimizer:
        try:
            logger.info("Saving optimizer state...")
            checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
        except Exception as e:
            logger.warning(f"Error saving optimizer state: {e}")
    
    # Save scheduler if requested
    if scheduler is not None and save_scheduler:
        try:
            logger.info("Saving scheduler state...")
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
        except Exception as e:
            logger.warning(f"Error saving scheduler state: {e}")
    
    # Add metadata
    if add_metadata:
        checkpoint_data.update({
            "epoch": epoch,
            "loss": loss
        })
        
        # Add additional metrics
        if metrics is not None:
            checkpoint_data["metrics"] = metrics
    
    # Try saving the checkpoint
    try:
        logger.info(f"Saving full checkpoint to {checkpoint_path}")
        torch.save(checkpoint_data, checkpoint_path)
        logger.info("Checkpoint saved successfully!")
        return checkpoint_path
    except Exception as primary_error:
        logger.error(f"Error saving full checkpoint: {primary_error}")
        
        # First fallback: Try to save just the model state dict
        try:
            model_only_path = os.path.join(output_dir, f"model_only_epoch_{epoch}.pt")
            logger.info(f"Attempting to save model-only checkpoint to {model_only_path}")
            torch.save(model_state, model_only_path)
            logger.info("Model-only checkpoint saved successfully")
            return model_only_path
        except Exception as model_error:
            logger.error(f"Could not save model-only checkpoint: {model_error}")
            
            # Second fallback: Try to save to an alternative location
            if emergency_save_dir is not None:
                try:
                    os.makedirs(emergency_save_dir, exist_ok=True)
                    emergency_path = os.path.join(emergency_save_dir, f"emergency_save_epoch_{epoch}.pt")
                    logger.info(f"Final attempt: Saving model to alternative location: {emergency_path}")
                    torch.save(model_state, emergency_path)
                    logger.info(f"Emergency save successful at {emergency_path}")
                    return emergency_path
                except Exception as emergency_error:
                    logger.error(f"Emergency save failed: {emergency_error}")
            
        # All methods failed
        raise RuntimeError(f"All checkpoint saving methods failed. Primary error: {primary_error}")


def load_checkpoint(
    model: Module,
    checkpoint_path: str,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Load checkpoint with robust error handling.
    
    Args:
        model: Model to load state into
        checkpoint_path: Path to the checkpoint file
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        map_location: Device to map tensors to
        strict: Whether to use strict state dict loading
        
    Returns:
        dict: Checkpoint metadata (epoch, loss, etc.)
    
    Raises:
        RuntimeError: If loading fails and strict=True
    """
    # Determine checkpoint type from filename
    checkpoint_type = get_checkpoint_type(checkpoint_path)
    logger.info(f"Loading {checkpoint_type} checkpoint from {checkpoint_path}")
    
    # Try to load the checkpoint
    checkpoint = try_load_checkpoint(checkpoint_path, map_location=map_location)
    if checkpoint is None:
        if strict:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}")
        logger.warning("Checkpoint loading failed, returning empty metadata")
        return {}
    
    # Extract model state dict
    model_state = extract_model_state(checkpoint, checkpoint_type)
    if model_state is None:
        if strict:
            raise RuntimeError(f"Could not extract model state from checkpoint {checkpoint_path}")
        logger.warning("Could not extract model state from checkpoint")
        return {}
    
    # Load model state dict
    try:
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        
        # Log missing and unexpected keys
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
            # Save for later reference
            model._load_missing_keys = missing_keys
            
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
            # Save for later reference
            model._load_unexpected_keys = unexpected_keys
            
        logger.info("Successfully loaded model weights")
    except Exception as e:
        if strict:
            raise RuntimeError(f"Error loading model state: {e}")
        logger.error(f"Error loading model state: {e}")
    
    # Determine if model architecture has changed significantly
    architecture_changed = (
        hasattr(model, "_load_missing_keys") and 
        bool(getattr(model, "_load_missing_keys", []))
    )
    
    # Load optimizer state if available and requested
    if optimizer is not None and checkpoint_type == "full":
        try:
            if "optimizer_state_dict" in checkpoint and not architecture_changed:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Loaded optimizer state from checkpoint")
            else:
                if architecture_changed:
                    logger.warning("Model architecture changed - using fresh optimizer state")
                else:
                    logger.warning("Optimizer state not found in checkpoint")
        except Exception as e:
            logger.warning(f"Error loading optimizer state: {e}")
    
    # Load scheduler state if available and requested
    if scheduler is not None and checkpoint_type == "full":
        try:
            if "scheduler_state_dict" in checkpoint and not architecture_changed:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("Loaded scheduler state from checkpoint")
            else:
                if architecture_changed:
                    logger.warning("Model architecture changed - using fresh scheduler state")
                else:
                    logger.warning("Scheduler state not found in checkpoint")
        except Exception as e:
            logger.warning(f"Error loading scheduler state: {e}")
    
    # Extract metadata
    metadata = {}
    
    # Get epoch number
    metadata["epoch"] = get_checkpoint_epoch(checkpoint_path, checkpoint)
    
    # Extract loss if available
    if isinstance(checkpoint, dict) and "loss" in checkpoint:
        try:
            metadata["loss"] = float(checkpoint["loss"])
            logger.info(f"Previous loss: {metadata['loss']:.4f}")
        except (ValueError, TypeError):
            logger.warning("Could not extract loss from checkpoint")
    
    # Extract other metrics if available
    if isinstance(checkpoint, dict) and "metrics" in checkpoint:
        metadata["metrics"] = checkpoint["metrics"]
    
    return metadata


def create_model_archive(
    checkpoint_path: str,
    output_path: Optional[str] = None,
    include_optimizer: bool = False,
    include_metadata: bool = True
) -> str:
    """
    Create a model archive from a checkpoint.
    
    This utility creates a standalone model checkpoint that can be
    easily shared or used for inference, optionally stripping optimizer
    state and other training data.
    
    Args:
        checkpoint_path: Path to the source checkpoint
        output_path: Path for the output archive
        include_optimizer: Whether to include optimizer state
        include_metadata: Whether to include training metadata
        
    Returns:
        str: Path to the created archive
    """
    # Set default output path if not provided
    if output_path is None:
        dirname = os.path.dirname(checkpoint_path)
        basename = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_path = os.path.join(dirname, f"{basename}_archive.pt")
    
    # Load the checkpoint
    checkpoint = try_load_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}")
    
    # Extract model state dict
    model_state = extract_model_state(checkpoint, get_checkpoint_type(checkpoint_path))
    if model_state is None:
        raise RuntimeError(f"Could not extract model state from checkpoint {checkpoint_path}")
    
    # Create archive data
    archive_data = {"model_state_dict": model_state}
    
    # Add optimizer state if requested
    if include_optimizer and isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
        archive_data["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]
    
    # Add metadata if requested
    if include_metadata and isinstance(checkpoint, dict):
        for key in ["epoch", "loss", "metrics"]:
            if key in checkpoint:
                archive_data[key] = checkpoint[key]
    
    # Save the archive
    try:
        logger.info(f"Creating model archive at {output_path}")
        torch.save(archive_data, output_path)
        logger.info("Model archive created successfully")
        return output_path
    except Exception as e:
        logger.error(f"Error creating model archive: {e}")
        raise RuntimeError(f"Failed to create model archive: {e}")