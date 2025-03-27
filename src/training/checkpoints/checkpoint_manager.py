"""
Checkpoint management for the enhanced training system.

This module provides utilities for saving, loading, and managing
model checkpoints during training.
"""

import os
import glob
import json
import logging
import torch
import time
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import asdict

from src.config.training_config import TrainingConfig
from src.config.model_config import ModelConfig


class CheckpointManager:
    """
    Manager for model checkpoints.
    
    This class handles saving, loading, and managing model checkpoints,
    including checkpoint rotation, versioning, and metadata tracking.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        output_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        save_total_limit: Optional[int] = None
    ):
        """
        Initialize the checkpoint manager.
        
        Args:
            config: Training configuration
            output_dir: Directory for checkpoints (overrides config)
            logger: Logger instance
            save_total_limit: Maximum number of checkpoints to keep
        """
        self.config = config
        self.logger = logger or logging.getLogger("quantum_resonance")
        
        # Set output directory
        self.output_dir = output_dir or getattr(config, "output_dir", "runs/quantum_resonance")
        
        # Set checkpoint directory
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Set save limit
        self.save_total_limit = save_total_limit
        if save_total_limit is None:
            self.save_total_limit = getattr(config, "save_total_limit", 3)
        
        # Get checkpoint settings from config
        self.save_steps = getattr(config, "save_steps", 0)
        self.save_every_epoch = getattr(config, "save_every_epoch", True)
        self.disable_optimizer_saving = getattr(config, "disable_optimizer_saving", False)
        self.auto_resume = getattr(config, "auto_resume", True)
        self.ignore_checkpoints = getattr(config, "ignore_checkpoints", False)
        
        # Track latest checkpoint
        self.last_checkpoint_path = None
        
        self.logger.info(f"Checkpoint manager initialized with checkpoint_dir: {self.checkpoint_dir}")
        self.logger.info(f"Save limit: {self.save_total_limit}, Save every epoch: {self.save_every_epoch}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        metrics: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None,
        extension_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a model checkpoint.
        
        Args:
            model: Model instance
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            epoch: Current epoch
            global_step: Current global step
            model_config: Model configuration
            training_config: Training configuration
            metrics: Training/validation metrics
            path: Path to save checkpoint (if None, generates a path)
            extension_data: Additional data from extensions
            metadata: Additional metadata to save
            
        Returns:
            Path to the saved checkpoint
        """
        # Generate checkpoint path if not provided
        if path is None:
            path = self._get_checkpoint_path(epoch, global_step)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Create checkpoint dictionary
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "timestamp": time.time()
        }
        
        # Add optimizer state
        if optimizer is not None and not self.disable_optimizer_saving:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        # Add scheduler state
        if scheduler is not None and not self.disable_optimizer_saving:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        # Add configurations
        if model_config is not None:
            checkpoint["model_config"] = asdict(model_config) if hasattr(model_config, "__dataclass_fields__") else model_config.to_dict()
        
        if training_config is not None:
            checkpoint["training_config"] = asdict(training_config) if hasattr(training_config, "__dataclass_fields__") else training_config.to_dict()
        
        # Add metrics
        if metrics is not None:
            checkpoint["metrics"] = metrics
        
        # Add extension data
        if extension_data is not None:
            checkpoint["extension_data"] = extension_data
        
        # Add metadata
        if metadata is not None:
            checkpoint["metadata"] = metadata
        else:
            checkpoint["metadata"] = {
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": "1.0.0"  # Version of checkpoint format
            }
        
        # Save the checkpoint
        try:
            if self.disable_optimizer_saving:
                # Save only model weights in a simpler format
                torch.save(checkpoint["model_state_dict"], path)
                
                # Save metadata separately
                metadata_path = path + ".meta.json"
                with open(metadata_path, "w") as f:
                    meta_dict = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
                    json.dump(meta_dict, f, indent=2)
            else:
                # Save full checkpoint
                torch.save(checkpoint, path)
            
            self.last_checkpoint_path = path
            self.logger.info(f"Saved checkpoint to {path}")
            
            # Cleanup old checkpoints
            if self.save_total_limit > 0:
                self._cleanup_old_checkpoints()
            
            return path
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint to {path}: {e}")
            return ""
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        path: Optional[str] = None,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        
        Args:
            model: Model instance
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            path: Path to checkpoint (if None, loads the latest checkpoint)
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            device: Device to load the checkpoint to
            
        Returns:
            Dictionary of loaded checkpoint info
        """
        # Find latest checkpoint if path not provided
        if path is None:
            path = self.get_latest_checkpoint()
            if path is None:
                self.logger.warning("No checkpoint found to load")
                return {}
        
        # Ensure the checkpoint exists
        if not os.path.exists(path):
            self.logger.error(f"Checkpoint {path} not found")
            return {}
        
        # Determine device
        if device is None:
            device = next(model.parameters()).device
        
        try:
            # Load checkpoint
            checkpoint = torch.load(path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Full checkpoint format
                model.load_state_dict(checkpoint["model_state_dict"])
                
                # Load optimizer if available and requested
                if optimizer is not None and load_optimizer and "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                
                # Load scheduler if available and requested
                if scheduler is not None and load_scheduler and "scheduler_state_dict" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                
                # Return checkpoint info
                self.logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', 'unknown')})")
                return {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
                
            else:
                # Model-only checkpoint format
                model.load_state_dict(checkpoint)
                
                # Try to load metadata if available
                metadata_path = path + ".meta.json"
                metadata = {}
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                    except Exception as e:
                        self.logger.warning(f"Error loading checkpoint metadata: {e}")
                
                self.logger.info(f"Loaded model-only checkpoint from {path}")
                return metadata
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint from {path}: {e}")
            return {}
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Find the latest checkpoint in the checkpoint directory.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints found
        """
        if self.ignore_checkpoints:
            self.logger.info("Ignoring existing checkpoints as configured")
            return None
        
        # First, check if we have a best_model checkpoint
        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            self.logger.info(f"Found best model checkpoint: {best_model_path}")
            return best_model_path
        
        # Search for step-based checkpoints
        step_pattern = os.path.join(self.checkpoint_dir, "step_*.pt")
        step_checkpoints = glob.glob(step_pattern)
        
        # Search for epoch-based checkpoints
        epoch_pattern = os.path.join(self.checkpoint_dir, "epoch_*.pt")
        epoch_checkpoints = glob.glob(epoch_pattern)
        
        # Combine all checkpoints
        all_checkpoints = step_checkpoints + epoch_checkpoints
        
        if not all_checkpoints:
            return None
        
        # Find the latest checkpoint by timestamp
        latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
        self.logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        
        return latest_checkpoint
    
    def get_checkpoint_info(self, path: str) -> Dict[str, Any]:
        """
        Get information about a checkpoint without loading the model weights.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Dictionary of checkpoint info
        """
        if not os.path.exists(path):
            self.logger.error(f"Checkpoint {path} not found")
            return {}
        
        try:
            # Check if there's a metadata file
            metadata_path = path + ".meta.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    return json.load(f)
            
            # Otherwise, load the checkpoint but only extract metadata
            checkpoint = torch.load(path, map_location="cpu")
            
            if isinstance(checkpoint, dict):
                # Return everything except model weights
                return {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
            else:
                # Checkpoint is just the model weights
                return {"format": "model_only"}
                
        except Exception as e:
            self.logger.error(f"Error getting info from checkpoint {path}: {e}")
            return {}
    
    def should_save_checkpoint(
        self,
        epoch: int,
        global_step: int
    ) -> bool:
        """
        Determine if a checkpoint should be saved at the current step/epoch.
        
        Args:
            epoch: Current epoch
            global_step: Current global step
            
        Returns:
            True if checkpoint should be saved, False otherwise
        """
        # Save at specified steps
        if self.save_steps > 0 and global_step % self.save_steps == 0:
            return True
        
        # Save at the end of each epoch
        if self.save_every_epoch:
            return True
        
        return False
    
    def _get_checkpoint_path(
        self,
        epoch: int = 0,
        global_step: int = 0
    ) -> str:
        """
        Generate a path for a new checkpoint.
        
        Args:
            epoch: Current epoch
            global_step: Current global step
            
        Returns:
            Path for the new checkpoint
        """
        # Determine whether to use step or epoch in filename
        if self.save_steps > 0 and global_step > 0:
            return os.path.join(self.checkpoint_dir, f"step_{global_step}.pt")
        else:
            return os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pt")
    
    def _cleanup_old_checkpoints(self) -> None:
        """
        Remove old checkpoints to stay within the save limit.
        
        Keeps the most recent checkpoints based on modification time.
        """
        # Don't cleanup if limit is 0 or negative
        if self.save_total_limit <= 0:
            return
        
        # Get all step and epoch checkpoints
        step_pattern = os.path.join(self.checkpoint_dir, "step_*.pt")
        epoch_pattern = os.path.join(self.checkpoint_dir, "epoch_*.pt")
        
        checkpoint_files = glob.glob(step_pattern) + glob.glob(epoch_pattern)
        
        # Don't do anything if we're under the limit
        if len(checkpoint_files) <= self.save_total_limit:
            return
        
        # Sort by modification time (oldest first)
        checkpoint_files = sorted(checkpoint_files, key=os.path.getmtime)
        
        # Remove the oldest checkpoints
        for path in checkpoint_files[:-self.save_total_limit]:
            try:
                os.remove(path)
                
                # Also remove metadata file if it exists
                metadata_path = path + ".meta.json"
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                
                self.logger.info(f"Removed old checkpoint: {path}")
            except Exception as e:
                self.logger.warning(f"Error removing old checkpoint {path}: {e}")
    
    def get_epoch_from_checkpoint(self, path: str) -> int:
        """
        Extract the epoch number from a checkpoint.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Epoch number or 0 if not found
        """
        try:
            info = self.get_checkpoint_info(path)
            return info.get("epoch", 0)
        except Exception:
            # Extract from filename if possible
            try:
                epoch_match = re.search(r"epoch_(\d+)\.pt", path)
                if epoch_match:
                    return int(epoch_match.group(1))
            except Exception:
                pass
            
            return 0
    
    def get_step_from_checkpoint(self, path: str) -> int:
        """
        Extract the global step from a checkpoint.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Global step or 0 if not found
        """
        try:
            info = self.get_checkpoint_info(path)
            return info.get("global_step", 0)
        except Exception:
            # Extract from filename if possible
            try:
                step_match = re.search(r"step_(\d+)\.pt", path)
                if step_match:
                    return int(step_match.group(1))
            except Exception:
                pass
            
            return 0