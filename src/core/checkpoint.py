"""
Unified checkpoint handling for QLLM.

This module provides a consolidated implementation of checkpoint saving and loading
functionality, which was previously duplicated across multiple files in the codebase.
It includes robust error handling, disk space checks, and support for different
checkpoint types.
"""

import os
import re
import shutil
import pickle
import logging
import warnings
from datetime import datetime
from typing import Dict, Any, Optional, Union, List, Tuple, Set, Type

import torch
from torch.nn import Module
from torch.optim import Optimizer


# Setup logger
logger = logging.getLogger("qllm.checkpoint")


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
    epoch_regex: str = r'epoch[-_]?(\d+)',
    step_regex: str = r'step[-_]?(\d+)',
    sort_by: str = 'epoch'
) -> Optional[str]:
    """
    Find the latest checkpoint in a directory based on filename patterns.
    
    This function consolidates code duplicated between core and examples.
    
    Args:
        directory: Directory to search in
        patterns: List of filename patterns to match (glob-style)
        epoch_regex: Regex pattern to extract epoch numbers
        step_regex: Regex pattern to extract step numbers
        sort_by: Sort criterion ('epoch', 'step', or 'time')
        
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
            "checkpoint_epoch*.pt",      # Standard checkpoints
            "checkpoint_step*.pt",       # Step-based checkpoints
            "model_epoch*.pt",           # Model-only checkpoints
            "model_step*.pt",            # Model-only step checkpoints
            "*.pt"                       # Any other PT files
        ]
    
    # Find all matching files
    checkpoint_files = []
    for pattern in patterns:
        full_pattern = os.path.join(directory, pattern)
        checkpoint_files.extend(glob.glob(full_pattern))
    
    if not checkpoint_files:
        return None
    
    # Extract info based on sort criterion
    with_info = []
    for filepath in checkpoint_files:
        file_info = {"path": filepath, "epoch": -1, "step": -1, "mtime": os.path.getmtime(filepath)}
        
        # Extract epoch number
        epoch_match = re.search(epoch_regex, filepath)
        if epoch_match:
            try:
                file_info["epoch"] = int(epoch_match.group(1))
            except (ValueError, IndexError):
                pass
                
        # Extract step number
        step_match = re.search(step_regex, filepath)
        if step_match:
            try:
                file_info["step"] = int(step_match.group(1))
            except (ValueError, IndexError):
                pass
        
        with_info.append(file_info)
    
    # Sort by requested criterion
    if sort_by == 'epoch':
        with_info.sort(key=lambda x: x["epoch"], reverse=True)
    elif sort_by == 'step':
        with_info.sort(key=lambda x: x["step"], reverse=True)
    else:  # Default to time
        with_info.sort(key=lambda x: x["mtime"], reverse=True)
    
    # Return the latest file
    return with_info[0]["path"] if with_info else None


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
        str: One of "full", "model_only", "optimizer_only", "emergency", or "unknown"
    """
    basename = os.path.basename(checkpoint_path).lower()
    
    if "full" in basename or "checkpoint" in basename:
        return "full"
    elif "model" in basename and "only" in basename:
        return "model_only"
    elif "optim" in basename:
        return "optimizer_only"
    elif "emergency" in basename:
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
    
    # For model_only, the checkpoint might be the state dict directly
    if checkpoint_type in ["model_only", "emergency"]:
        if is_likely_state_dict(checkpoint):
            return checkpoint
    
    # Unknown format - try different approaches
    if isinstance(checkpoint, dict):
        # Try common keys
        for key in ["model_state_dict", "state_dict", "model", "net", "network"]:
            if key in checkpoint:
                return checkpoint[key]
        
        # Check if the dict itself looks like a state dict
        if is_likely_state_dict(checkpoint):
            return checkpoint
    
    # If it's a model object that has state_dict method
    if hasattr(checkpoint, "state_dict") and callable(checkpoint.state_dict):
        return checkpoint.state_dict()
    
    return None


def get_checkpoint_metadata(
    checkpoint_path: str,
    checkpoint_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get metadata from checkpoint path or data.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        checkpoint_data: Loaded checkpoint data (optional)
        
    Returns:
        dict: Metadata including epoch, step, etc.
    """
    metadata = {
        "epoch": 0,
        "step": 0,
        "timestamp": os.path.getmtime(checkpoint_path) if os.path.exists(checkpoint_path) else 0,
        "type": get_checkpoint_type(checkpoint_path),
        "path": checkpoint_path
    }
    
    # Try to extract epoch and step from filename
    epoch_match = re.search(r'epoch[-_]?(\d+)', checkpoint_path)
    if epoch_match:
        try:
            metadata["epoch"] = int(epoch_match.group(1))
        except (ValueError, IndexError):
            pass
            
    step_match = re.search(r'step[-_]?(\d+)', checkpoint_path)
    if step_match:
        try:
            metadata["step"] = int(step_match.group(1))
        except (ValueError, IndexError):
            pass
    
    # Try to get from checkpoint data
    if checkpoint_data is not None and isinstance(checkpoint_data, dict):
        for key in ["epoch", "step", "global_step", "iteration"]:
            if key in checkpoint_data:
                try:
                    metadata[key.replace("global_step", "step").replace("iteration", "step")] = int(checkpoint_data[key])
                except (ValueError, TypeError):
                    pass
        
        # Extract additional metadata if available
        for key in ["loss", "metrics", "config"]:
            if key in checkpoint_data:
                metadata[key] = checkpoint_data[key]
    
    return metadata


def save_checkpoint(
    model: Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    loss: float = 0.0,
    metrics: Optional[Dict[str, Any]] = None,
    output_dir: str = "checkpoints",
    filename: Optional[str] = None,
    save_optimizer: bool = True,
    save_scheduler: bool = True,
    add_metadata: bool = True,
    backup_existing: bool = True,
    min_free_space_mb: float = 500.0,
    emergency_save_dir: Optional[str] = None,
    **extra_data
) -> str:
    """
    Save checkpoint with robust error handling and fallbacks.
    
    This function consolidates checkpoint saving logic that was duplicated in
    multiple places in the codebase.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save
        epoch: Current epoch number
        step: Current step number
        loss: Current loss value
        metrics: Additional metrics to save
        output_dir: Directory to save checkpoint to
        filename: Specific filename (default: checkpoint_epoch_{epoch}_step_{step}.pt)
        save_optimizer: Whether to save optimizer state
        save_scheduler: Whether to save scheduler state
        add_metadata: Whether to add metadata to checkpoint
        backup_existing: Whether to backup existing checkpoint
        min_free_space_mb: Minimum free space required in MB
        emergency_save_dir: Alternative directory for emergency saving
        **extra_data: Additional data to include in the checkpoint
    
    Returns:
        str: Path to saved checkpoint
    
    Raises:
        RuntimeError: If all save methods fail
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch_{epoch}_step_{step}_{timestamp}.pt"
    
    # Full path to checkpoint
    checkpoint_path = os.path.join(output_dir, filename)
    
    # Check disk space
    total_mb, used_mb, free_mb = get_disk_space(output_dir)
    logger.info(f"Available disk space: {free_mb:.2f} MB of {total_mb:.2f} MB")
    
    if free_mb < min_free_space_mb:
        logger.warning(f"Low disk space warning: {free_mb:.2f} MB available")
        if emergency_save_dir:
            # Check if emergency dir has more space
            _, _, emergency_free_mb = get_disk_space(emergency_save_dir)
            if emergency_free_mb > free_mb:
                os.makedirs(emergency_save_dir, exist_ok=True)
                output_dir = emergency_save_dir
                checkpoint_path = os.path.join(output_dir, filename)
                logger.info(f"Using emergency save directory: {emergency_save_dir}")
    
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
            "step": step,
            "loss": loss,
        })
        
        # Add additional metrics
        if metrics is not None:
            checkpoint_data["metrics"] = metrics
    
    # Add any extra data
    for key, value in extra_data.items():
        checkpoint_data[key] = value
    
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
            model_only_path = os.path.join(output_dir, f"model_only_epoch_{epoch}_step_{step}.pt")
            logger.info(f"Attempting to save model-only checkpoint to {model_only_path}")
            torch.save(model_state, model_only_path)
            logger.info("Model-only checkpoint saved successfully")
            return model_only_path
        except Exception as model_error:
            logger.error(f"Could not save model-only checkpoint: {model_error}")
            
            # Second fallback: Try to save to an alternative location
            if emergency_save_dir is not None and emergency_save_dir != output_dir:
                try:
                    os.makedirs(emergency_save_dir, exist_ok=True)
                    emergency_path = os.path.join(emergency_save_dir, f"emergency_save_epoch_{epoch}_step_{step}.pt")
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
    strict: bool = False,
    load_optimizer: bool = True,
    load_scheduler: bool = True
) -> Dict[str, Any]:
    """
    Load checkpoint with robust error handling.
    
    This function consolidates checkpoint loading logic that was duplicated in
    multiple places in the codebase.
    
    Args:
        model: Model to load state into
        checkpoint_path: Path to the checkpoint file
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        map_location: Device to map tensors to
        strict: Whether to use strict state dict loading
        load_optimizer: Whether to load optimizer state
        load_scheduler: Whether to load scheduler state
        
    Returns:
        dict: Checkpoint metadata (epoch, step, loss, etc.)
    
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
    if optimizer is not None and checkpoint_type == "full" and load_optimizer:
        try:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Successfully loaded optimizer state")
            else:
                logger.warning("No optimizer state found in checkpoint")
        except Exception as e:
            if strict:
                raise RuntimeError(f"Error loading optimizer state: {e}")
            logger.warning(f"Error loading optimizer state: {e}")
            
            # Skip optimizer loading if architecture changed
            if architecture_changed:
                logger.warning("Skipping optimizer loading due to model architecture changes")
    
    # Load scheduler state if available and requested
    if scheduler is not None and checkpoint_type == "full" and load_scheduler:
        try:
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("Successfully loaded scheduler state")
            else:
                logger.warning("No scheduler state found in checkpoint")
        except Exception as e:
            if strict:
                raise RuntimeError(f"Error loading scheduler state: {e}")
            logger.warning(f"Error loading scheduler state: {e}")
    
    # Extract metadata
    metadata = get_checkpoint_metadata(checkpoint_path, checkpoint)
    
    return metadata


class CheckpointManager:
    """
    Checkpoint manager that handles saving, loading, and tracking checkpoints.
    
    This class provides a centralized way to manage checkpoints, including:
    - Automatic checkpoint rotation to save disk space
    - Best checkpoint tracking based on metrics
    - Regular checkpoint scheduling
    - Checkpoint metadata management
    
    It consolidates checkpoint management logic found duplicated in several
    files in the codebase.
    """
    
    def __init__(
        self,
        output_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        save_best_metric: Optional[str] = "loss",
        lower_better: bool = True,
        save_interval_epochs: int = 1,
        save_interval_steps: int = 0,
        emergency_dir: Optional[str] = None
    ):
        """
        Initialize the checkpoint manager.
        
        Args:
            output_dir: Directory to save checkpoints to
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_metric: Metric to use for saving best checkpoint
            lower_better: Whether lower values of the metric are better
            save_interval_epochs: Save checkpoint every N epochs
            save_interval_steps: Save checkpoint every N steps
            emergency_dir: Directory for emergency saves
        """
        self.output_dir = output_dir
        self.max_checkpoints = max_checkpoints
        self.save_best_metric = save_best_metric
        self.lower_better = lower_better
        self.save_interval_epochs = save_interval_epochs
        self.save_interval_steps = save_interval_steps
        self.emergency_dir = emergency_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        if emergency_dir:
            os.makedirs(emergency_dir, exist_ok=True)
        
        # Track best checkpoint
        self.best_metric_value = float('inf') if lower_better else float('-inf')
        self.best_checkpoint_path = None
        
        # Track checkpoint history
        self.checkpoint_history: List[Dict[str, Any]] = []
        self.last_saved_epoch = -1
        self.last_saved_step = -1
        
        # Load checkpoint history from output directory
        self._load_checkpoint_history()
    
    def _load_checkpoint_history(self) -> None:
        """Load checkpoint history from existing checkpoints in the output directory."""
        if not os.path.exists(self.output_dir):
            return
            
        # Get all checkpoint files
        checkpoint_files = []
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.pt') and not filename.endswith('.bak'):
                checkpoint_path = os.path.join(self.output_dir, filename)
                checkpoint_files.append(checkpoint_path)
        
        # Extract metadata from each checkpoint
        for checkpoint_path in checkpoint_files:
            metadata = get_checkpoint_metadata(checkpoint_path)
            self.checkpoint_history.append(metadata)
        
        # Sort by epoch and step
        self.checkpoint_history.sort(key=lambda x: (x.get("epoch", 0), x.get("step", 0)))
        
        # Update last saved epoch/step
        if self.checkpoint_history:
            latest = self.checkpoint_history[-1]
            self.last_saved_epoch = latest.get("epoch", -1)
            self.last_saved_step = latest.get("step", -1)
            
            # Update best checkpoint if metric is available
            if self.save_best_metric and self.save_best_metric in latest.get("metrics", {}):
                metric_value = latest["metrics"][self.save_best_metric]
                if ((self.lower_better and metric_value < self.best_metric_value) or
                    (not self.lower_better and metric_value > self.best_metric_value)):
                    self.best_metric_value = metric_value
                    self.best_checkpoint_path = latest["path"]
    
    def should_save_checkpoint(self, epoch: int, step: int) -> bool:
        """
        Determine if a checkpoint should be saved based on the current epoch and step.
        
        Args:
            epoch: Current epoch
            step: Current step
            
        Returns:
            bool: True if a checkpoint should be saved
        """
        # Save if this is the first checkpoint
        if self.last_saved_epoch == -1 and self.last_saved_step == -1:
            return True
            
        # Save based on epoch interval
        if (self.save_interval_epochs > 0 and 
            (epoch - self.last_saved_epoch) >= self.save_interval_epochs):
            return True
            
        # Save based on step interval
        if (self.save_interval_steps > 0 and 
            (step - self.last_saved_step) >= self.save_interval_steps):
            return True
            
        return False
    
    def save_checkpoint(
        self,
        model: Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        loss: float = 0.0,
        metrics: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
        force: bool = False,
        **extra_data
    ) -> Optional[str]:
        """
        Save a checkpoint if the save criteria are met.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Scheduler to save
            epoch: Current epoch
            step: Current step
            loss: Current loss value
            metrics: Additional metrics
            filename: Specific filename
            force: Force save regardless of criteria
            **extra_data: Additional data to include
            
        Returns:
            str: Path to saved checkpoint or None if not saved
        """
        # Check if we should save
        if not force and not self.should_save_checkpoint(epoch, step):
            return None
        
        # Create default filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_epoch_{epoch}_step_{step}_{timestamp}.pt"
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            loss=loss,
            metrics=metrics,
            output_dir=self.output_dir,
            filename=filename,
            emergency_save_dir=self.emergency_dir,
            **extra_data
        )
        
        # Update history
        metadata = get_checkpoint_metadata(checkpoint_path)
        if metrics is not None:
            metadata["metrics"] = metrics
        self.checkpoint_history.append(metadata)
        
        # Update last saved epoch/step
        self.last_saved_epoch = epoch
        self.last_saved_step = step
        
        # Check if this is a new best checkpoint
        if (metrics is not None and self.save_best_metric and
            self.save_best_metric in metrics):
            metric_value = metrics[self.save_best_metric]
            if ((self.lower_better and metric_value < self.best_metric_value) or
                (not self.lower_better and metric_value > self.best_metric_value)):
                self.best_metric_value = metric_value
                self.best_checkpoint_path = checkpoint_path
                
                # Create a symlink or copy to best.pt
                best_path = os.path.join(self.output_dir, "best.pt")
                try:
                    if os.path.exists(best_path):
                        os.remove(best_path)
                    # Try symlink first
                    try:
                        os.symlink(checkpoint_path, best_path)
                    except:
                        # Fall back to copy
                        shutil.copy2(checkpoint_path, best_path)
                    logger.info(f"New best checkpoint: {metric_value} (previous: {self.best_metric_value})")
                except Exception as e:
                    logger.warning(f"Failed to create best checkpoint link: {e}")
        
        # Rotate old checkpoints
        self._rotate_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        model: Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        path: Optional[str] = None,
        load_best: bool = False,
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            path: Path to specific checkpoint file
            load_best: Whether to load the best checkpoint
            map_location: Device to map tensors to
            strict: Whether to use strict state dict loading
            
        Returns:
            dict: Checkpoint metadata
        """
        # Determine which checkpoint to load
        if path is not None:
            checkpoint_path = path
        elif load_best and self.best_checkpoint_path is not None:
            checkpoint_path = self.best_checkpoint_path
        else:
            # Find the latest checkpoint
            checkpoint_path = find_latest_checkpoint(self.output_dir)
            
        if checkpoint_path is None:
            logger.warning("No checkpoint found to load")
            return {}
            
        # Load the checkpoint
        metadata = load_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location,
            strict=strict
        )
        
        # Update state
        if "epoch" in metadata:
            self.last_saved_epoch = metadata["epoch"]
        if "step" in metadata:
            self.last_saved_step = metadata["step"]
            
        return metadata
    
    def _rotate_checkpoints(self) -> None:
        """
        Remove old checkpoints to stay within max_checkpoints limit.
        Keeps the best checkpoint even if it's old.
        """
        if self.max_checkpoints <= 0 or len(self.checkpoint_history) <= self.max_checkpoints:
            return
            
        # Sort by timestamp (oldest first)
        sorted_checkpoints = sorted(
            self.checkpoint_history, 
            key=lambda x: x.get("timestamp", 0)
        )
        
        # Keep the best checkpoint regardless of age
        checkpoints_to_remove = []
        for checkpoint in sorted_checkpoints[:-self.max_checkpoints]:
            # Skip best checkpoint
            if checkpoint["path"] == self.best_checkpoint_path:
                continue
                
            checkpoints_to_remove.append(checkpoint)
            
        # Remove old checkpoints
        for checkpoint in checkpoints_to_remove:
            try:
                if os.path.exists(checkpoint["path"]):
                    os.remove(checkpoint["path"])
                    logger.info(f"Removed old checkpoint: {checkpoint['path']}")
                self.checkpoint_history.remove(checkpoint)
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint['path']}: {e}")