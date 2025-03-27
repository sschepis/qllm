"""
Metrics logging for the enhanced training system.

This module provides metrics logging functionality, supporting console output,
file logging, and TensorBoard integration.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List

import torch


class MetricsLogger:
    """
    Logger for training and evaluation metrics.
    
    This class provides logging functionality for metrics, supporting console output,
    file logging, and TensorBoard integration if available.
    """
    
    def __init__(
        self,
        output_dir: str,
        log_to_console: bool = True,
        log_to_file: bool = True,
        log_to_tensorboard: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the metrics logger.
        
        Args:
            output_dir: Directory for output files
            log_to_console: Whether to log to console
            log_to_file: Whether to log to files
            log_to_tensorboard: Whether to log to TensorBoard
            logger: Logger instance
        """
        self.output_dir = output_dir
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        self.log_to_tensorboard = log_to_tensorboard
        self.logger = logger or logging.getLogger("quantum_resonance")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize tensorboard writer if requested
        self.writer = None
        if self.log_to_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tensorboard"))
                self.logger.info(f"TensorBoard logs will be saved to {os.path.join(self.output_dir, 'tensorboard')}")
            except ImportError:
                self.logger.warning("TensorBoard not available, disabling TensorBoard logging")
                self.log_to_tensorboard = False
        
        # Initialize log files
        if self.log_to_file:
            self.train_log_file = os.path.join(self.output_dir, "train_metrics.jsonl")
            self.val_log_file = os.path.join(self.output_dir, "val_metrics.jsonl")
            
            # Create log files with empty lists
            with open(self.train_log_file, "w") as f:
                f.write("")
            with open(self.val_log_file, "w") as f:
                f.write("")
            
            self.logger.info(f"Training metrics will be saved to {self.train_log_file}")
            self.logger.info(f"Validation metrics will be saved to {self.val_log_file}")
    
    def log_training_step(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Log metrics for a training step.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step
        """
        # Add step to metrics
        metrics_with_step = {"step": step, **metrics}
        
        # Log to console
        if self.log_to_console:
            metrics_str = ", ".join(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" 
                              for k, v in metrics.items())
            self.logger.debug(f"Step {step}: {metrics_str}")
        
        # Log to tensorboard
        if self.log_to_tensorboard and self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"train/{key}", value, step)
            
            # Log learning rate if available
            if "learning_rate" in metrics:
                self.writer.add_scalar("train/learning_rate", metrics["learning_rate"], step)
        
        # Log to file
        if self.log_to_file:
            with open(self.train_log_file, "a") as f:
                f.write(json.dumps(self._serialize_metrics(metrics_with_step)) + "\n")
    
    def log_epoch_metrics(self, metrics: Dict[str, Any], epoch: int) -> None:
        """
        Log metrics for a training epoch.
        
        Args:
            metrics: Dictionary of metrics
            epoch: Current epoch
        """
        # Add epoch to metrics
        metrics_with_epoch = {"epoch": epoch, **metrics}
        
        # Log to console
        if self.log_to_console:
            metrics_str = ", ".join(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" 
                              for k, v in metrics.items())
            self.logger.info(f"Epoch {epoch}: {metrics_str}")
        
        # Log to tensorboard
        if self.log_to_tensorboard and self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"epoch/{key}", value, epoch)
        
        # Log to file
        if self.log_to_file:
            with open(self.train_log_file, "a") as f:
                f.write(json.dumps(self._serialize_metrics(metrics_with_epoch)) + "\n")
    
    def log_validation_metrics(self, metrics: Dict[str, Any], epoch: int) -> None:
        """
        Log metrics for validation.
        
        Args:
            metrics: Dictionary of metrics
            epoch: Current epoch
        """
        # Add epoch to metrics
        metrics_with_epoch = {"epoch": epoch, **metrics}
        
        # Log to console
        if self.log_to_console:
            metrics_str = ", ".join(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" 
                              for k, v in metrics.items())
            self.logger.info(f"Validation metrics (Epoch {epoch}): {metrics_str}")
        
        # Log to tensorboard
        if self.log_to_tensorboard and self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"val/{key}", value, epoch)
        
        # Log to file
        if self.log_to_file:
            with open(self.val_log_file, "a") as f:
                f.write(json.dumps(self._serialize_metrics(metrics_with_epoch)) + "\n")
    
    def close(self) -> None:
        """Close the logger and release resources."""
        if self.log_to_tensorboard and self.writer is not None:
            self.writer.close()
    
    def _serialize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize metrics to JSON-compatible types.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Dictionary with serialized metrics
        """
        serialized = {}
        
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                serialized[key] = value.item() if value.numel() == 1 else value.tolist()
            elif isinstance(value, (int, float, str, bool, list, dict)):
                serialized[key] = value
            elif value is None:
                serialized[key] = None
            else:
                serialized[key] = str(value)
        
        return serialized