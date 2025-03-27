"""
Metrics logging for the enhanced training system.

This module provides components for tracking, aggregating, and reporting
metrics during model training and evaluation.
"""

import os
import json
import time
import logging
import math
from typing import Dict, Any, List, Optional, Union, Callable
from collections import defaultdict

import numpy as np
import torch


class MetricsLogger:
    """
    Logger for tracking and reporting training and evaluation metrics.
    
    This class collects, aggregates, and reports metrics throughout the
    training process, supporting various output formats and statistics.
    """
    
    def __init__(
        self,
        output_dir: str,
        log_to_console: bool = True,
        log_to_tensorboard: bool = True,
        log_to_file: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the metrics logger.
        
        Args:
            output_dir: Directory for metric outputs
            log_to_console: Whether to log metrics to console
            log_to_tensorboard: Whether to log metrics to TensorBoard
            log_to_file: Whether to log metrics to JSON files
            logger: Logger instance
        """
        self.output_dir = output_dir
        self.log_to_console = log_to_console
        self.log_to_tensorboard = log_to_tensorboard
        self.log_to_file = log_to_file
        self.logger = logger or logging.getLogger("quantum_resonance")
        
        # Create metrics directory
        self.metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize TensorBoard writer if requested
        self.tb_writer = None
        if log_to_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
                self.logger.info(f"TensorBoard logging enabled at {os.path.join(output_dir, 'tensorboard')}")
            except ImportError:
                self.logger.warning("Could not import TensorBoard. Please install tensorboard to enable logging.")
                self.log_to_tensorboard = False
        
        # Metrics storage
        self.train_metrics = defaultdict(list)
        self.eval_metrics = defaultdict(list)
        self.step_metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)
        
        # Track steps and epochs
        self.global_step = 0
        self.current_epoch = 0
        
        # Keep track of best metrics
        self.best_metrics = {}
        
        self.logger.info(f"Metrics logger initialized with output_dir: {output_dir}")
    
    def log_training_step(
        self,
        metrics: Dict[str, Any],
        step: int,
        epoch: int,
        log_to_tb: bool = True
    ) -> None:
        """
        Log metrics for a training step.
        
        Args:
            metrics: Dictionary of step metrics
            step: Current step
            epoch: Current epoch
            log_to_tb: Whether to log to TensorBoard
        """
        self.global_step = step
        self.current_epoch = epoch
        
        # Add step and epoch to metrics
        metrics_with_meta = {**metrics, "step": step, "epoch": epoch}
        
        # Store metrics
        for key, value in metrics.items():
            # Skip non-numeric values
            if not self._is_numeric(value):
                continue
                
            # Convert to Python float/int for JSON serialization
            value = self._to_python_numeric(value)
                
            self.train_metrics[key].append((step, value))
            self.step_metrics[key].append((step, value))
        
        # Log to console if requested
        if self.log_to_console:
            self._log_to_console(metrics, step=step, epoch=epoch, prefix="Train")
        
        # Log to TensorBoard if requested
        if self.log_to_tensorboard and log_to_tb and self.tb_writer is not None:
            self._log_to_tensorboard(metrics, step, prefix="train")
        
        # Log to file periodically
        if self.log_to_file and step % 100 == 0:
            self._save_metrics_to_file()
    
    def log_evaluation(
        self,
        metrics: Dict[str, Any],
        step: int,
        epoch: int,
        split: str = "val",
        log_to_tb: bool = True
    ) -> None:
        """
        Log metrics for an evaluation run.
        
        Args:
            metrics: Dictionary of evaluation metrics
            step: Current step
            epoch: Current epoch
            split: Dataset split ("val", "test")
            log_to_tb: Whether to log to TensorBoard
        """
        # Add step and epoch to metrics
        metrics_with_meta = {**metrics, "step": step, "epoch": epoch}
        
        # Store metrics
        for key, value in metrics.items():
            # Skip non-numeric values
            if not self._is_numeric(value):
                continue
                
            # Convert to Python float/int for JSON serialization
            value = self._to_python_numeric(value)
                
            self.eval_metrics[f"{split}_{key}"].append((step, value))
            
            # Track in epoch metrics if it's validation
            if split == "val":
                self.epoch_metrics[key].append((epoch, value))
                
                # Update best metrics
                if key not in self.best_metrics or self._is_better(key, value, self.best_metrics[key]):
                    self.best_metrics[key] = value
        
        # Log to console if requested
        if self.log_to_console:
            self._log_to_console(metrics, step=step, epoch=epoch, prefix=f"{split.capitalize()}")
        
        # Log to TensorBoard if requested
        if self.log_to_tensorboard and log_to_tb and self.tb_writer is not None:
            self._log_to_tensorboard(metrics, step, prefix=split)
        
        # Always log evaluations to file
        if self.log_to_file:
            filename = f"{split}_metrics_epoch_{epoch}.json"
            self._save_specific_metrics_to_file(
                {**metrics_with_meta, "timestamp": time.time()},
                os.path.join(self.metrics_dir, filename)
            )
            
            # Also update the overall metrics file
            self._save_metrics_to_file()
    
    def log_epoch(
        self,
        metrics: Dict[str, Any],
        epoch: int,
        log_to_tb: bool = True
    ) -> None:
        """
        Log metrics for a completed epoch.
        
        Args:
            metrics: Dictionary of epoch metrics
            epoch: Current epoch
            log_to_tb: Whether to log to TensorBoard
        """
        self.current_epoch = epoch
        
        # Add epoch to metrics
        metrics_with_meta = {**metrics, "epoch": epoch}
        
        # Store metrics
        for key, value in metrics.items():
            # Skip non-numeric values
            if not self._is_numeric(value):
                continue
                
            # Convert to Python float/int for JSON serialization
            value = self._to_python_numeric(value)
                
            self.epoch_metrics[key].append((epoch, value))
        
        # Log to console if requested
        if self.log_to_console:
            self._log_to_console(metrics, epoch=epoch, prefix="Epoch")
        
        # Log to TensorBoard if requested
        if self.log_to_tensorboard and log_to_tb and self.tb_writer is not None:
            self._log_to_tensorboard(metrics, epoch, prefix="epoch")
        
        # Always log epochs to file
        if self.log_to_file:
            filename = f"epoch_{epoch}_metrics.json"
            self._save_specific_metrics_to_file(
                {**metrics_with_meta, "timestamp": time.time()},
                os.path.join(self.metrics_dir, filename)
            )
            
            # Also update the overall metrics file
            self._save_metrics_to_file()
    
    def log_best_metrics(self) -> None:
        """Log the best metrics achieved during training."""
        if not self.best_metrics:
            return
        
        # Log to console
        if self.log_to_console:
            self.logger.info("Best metrics achieved during training:")
            for key, value in self.best_metrics.items():
                self.logger.info(f"  {key}: {value}")
        
        # Save to file
        if self.log_to_file:
            filename = "best_metrics.json"
            self._save_specific_metrics_to_file(
                {**self.best_metrics, "timestamp": time.time()},
                os.path.join(self.metrics_dir, filename)
            )
    
    def get_metrics_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get the complete metrics history.
        
        Returns:
            Dictionary of metric histories
        """
        history = {
            "train": self._format_metrics_history(self.train_metrics),
            "eval": self._format_metrics_history(self.eval_metrics),
            "epoch": self._format_metrics_history(self.epoch_metrics),
            "best": self.best_metrics.copy()
        }
        
        return history
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """
        Get the best metrics achieved during training.
        
        Returns:
            Dictionary of best metrics
        """
        return self.best_metrics.copy()
    
    def get_learning_rate_history(self) -> List[Tuple[int, float]]:
        """
        Get the learning rate history during training.
        
        Returns:
            List of (step, learning_rate) tuples
        """
        return self.train_metrics.get("learning_rate", [])
    
    def get_last_metrics(self, prefix: str = "val") -> Dict[str, Any]:
        """
        Get the most recent metrics for a specific prefix.
        
        Args:
            prefix: Metric prefix (train, val, test, etc.)
            
        Returns:
            Dictionary of most recent metrics
        """
        result = {}
        
        for key, values in self.eval_metrics.items():
            if key.startswith(f"{prefix}_") and values:
                metric_name = key[len(f"{prefix}_"):]
                result[metric_name] = values[-1][1]  # Last value
        
        return result
    
    def close(self) -> None:
        """Close the metrics logger and release resources."""
        # Save final metrics
        self._save_metrics_to_file()
        
        # Close TensorBoard writer
        if self.tb_writer is not None:
            self.tb_writer.close()
            self.tb_writer = None
        
        # Log best metrics before closing
        self.log_best_metrics()
    
    def _log_to_console(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        Log metrics to console.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step (optional)
            epoch: Current epoch (optional)
            prefix: Prefix for log messages
        """
        # Format the metrics message
        parts = []
        
        if prefix:
            parts.append(prefix)
        
        if epoch is not None:
            parts.append(f"Epoch: {epoch}")
        
        if step is not None:
            parts.append(f"Step: {step}")
        
        # Add metrics
        metric_strs = []
        for key, value in sorted(metrics.items()):
            # Skip step, epoch which are already in the header
            if key in ["step", "epoch"]:
                continue
                
            if not self._is_numeric(value):
                continue
                
            # Format based on value magnitude
            if isinstance(value, (int, np.integer)):
                metric_strs.append(f"{key}: {value}")
            else:
                # Use scientific notation for very small values
                if abs(value) < 0.001:
                    metric_strs.append(f"{key}: {value:.2e}")
                else:
                    # Show different precision based on value range
                    if abs(value) < 0.1:
                        metric_strs.append(f"{key}: {value:.6f}")
                    elif abs(value) < 10:
                        metric_strs.append(f"{key}: {value:.4f}")
                    else:
                        metric_strs.append(f"{key}: {value:.2f}")
        
        parts.extend(metric_strs)
        
        # Log the message
        self.logger.info(", ".join(parts))
    
    def _log_to_tensorboard(
        self,
        metrics: Dict[str, Any],
        step: int,
        prefix: str = ""
    ) -> None:
        """
        Log metrics to TensorBoard.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step
            prefix: Prefix for metric names
        """
        if self.tb_writer is None:
            return
        
        for key, value in metrics.items():
            # Skip non-numeric values
            if not self._is_numeric(value):
                continue
            
            # Add prefix to key if provided
            tb_key = f"{prefix}/{key}" if prefix else key
            
            # Log to TensorBoard
            try:
                self.tb_writer.add_scalar(tb_key, value, step)
            except Exception as e:
                self.logger.warning(f"Error logging {tb_key} to TensorBoard: {e}")
    
    def _save_metrics_to_file(self) -> None:
        """Save all metrics to a JSON file."""
        if not self.log_to_file:
            return
        
        try:
            # Prepare metrics data
            metrics_data = {
                "train": self._format_metrics_history(self.train_metrics),
                "eval": self._format_metrics_history(self.eval_metrics),
                "epoch": self._format_metrics_history(self.epoch_metrics),
                "best": self.best_metrics,
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "timestamp": time.time()
            }
            
            # Save to file
            filename = os.path.join(self.metrics_dir, "metrics.json")
            with open(filename, "w") as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Error saving metrics to file: {e}")
    
    def _save_specific_metrics_to_file(
        self,
        metrics: Dict[str, Any],
        filename: str
    ) -> None:
        """
        Save specific metrics to a JSON file.
        
        Args:
            metrics: Dictionary of metrics
            filename: Path to save the metrics
        """
        try:
            with open(filename, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Error saving metrics to {filename}: {e}")
    
    def _format_metrics_history(
        self,
        metrics_dict: Dict[str, List[Tuple[int, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Format metrics history for JSON serialization.
        
        Args:
            metrics_dict: Dictionary of metric histories
            
        Returns:
            Formatted metrics history
        """
        formatted = {}
        
        for key, values in metrics_dict.items():
            formatted[key] = [{"step": step, "value": value} for step, value in values]
        
        return formatted
    
    def _is_numeric(self, value: Any) -> bool:
        """
        Check if a value is numeric.
        
        Args:
            value: Value to check
            
        Returns:
            True if value is numeric, False otherwise
        """
        if isinstance(value, (int, float, np.number)):
            return True
        
        if isinstance(value, torch.Tensor):
            return value.numel() == 1 and torch.is_floating_point(value) or torch.is_integral(value)
        
        return False
    
    def _to_python_numeric(self, value: Any) -> Union[int, float]:
        """
        Convert a numeric value to a Python int or float.
        
        Args:
            value: Value to convert
            
        Returns:
            Python int or float equivalent
        """
        if isinstance(value, (int, float)):
            return value
        
        if isinstance(value, np.number):
            return value.item()
        
        if isinstance(value, torch.Tensor):
            return value.item()
        
        # Try to convert to float
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    
    def _is_better(self, metric_name: str, new_value: float, old_value: float) -> bool:
        """
        Check if a new metric value is better than the old one.
        
        Args:
            metric_name: Name of the metric
            new_value: New metric value
            old_value: Old metric value
            
        Returns:
            True if new value is better, False otherwise
        """
        # For loss-like metrics, lower is better
        if any(name in metric_name.lower() for name in ["loss", "error", "perplexity"]):
            return new_value < old_value
        
        # For everything else, higher is better
        return new_value > old_value