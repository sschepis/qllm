"""
Logging utilities for QLLM.

This module provides logging functionality for the Quantum Resonance
Language Model, including setup for file and console logging.
"""

import os
import sys
import logging
from typing import Optional


def setup_logger(
    name: str = "qllm",
    log_file: Optional[str] = None,
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Name of the logger
        log_file: Path to log file (if None, only console logging is enabled)
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file specified
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """Logger for training progress with detailed metrics."""
    
    def __init__(self, logger: Optional[logging.Logger] = None, log_interval: int = 10):
        """
        Initialize training logger.
        
        Args:
            logger: Logger instance (if None, a new one is created)
            log_interval: How often to log detailed metrics
        """
        self.logger = logger or setup_logger("training")
        self.log_interval = log_interval
        self.last_log_time = 0
    
    def start_epoch(self, epoch: int, max_epochs: int) -> None:
        """
        Log epoch start.
        
        Args:
            epoch: Current epoch
            max_epochs: Maximum number of epochs
        """
        self.logger.info(f"Starting epoch {epoch+1}/{max_epochs}")
    
    def end_epoch(self, epoch: int, loss: float) -> None:
        """
        Log epoch end.
        
        Args:
            epoch: Current epoch
            loss: Final loss value
        """
        self.logger.info(f"Epoch {epoch+1} completed with loss {loss:.4f}")
    
    def log_batch(
        self,
        epoch: int,
        batch: int,
        batch_size: int,
        total_batches: int,
        loss: float,
        learning_rate: float
    ) -> None:
        """
        Log batch metrics.
        
        Args:
            epoch: Current epoch
            batch: Current batch
            batch_size: Batch size
            total_batches: Total number of batches
            loss: Batch loss
            learning_rate: Current learning rate
        """
        if batch % self.log_interval == 0:
            progress = batch / total_batches * 100
            self.logger.info(
                f"Epoch {epoch+1}, Batch {batch}/{total_batches} ({progress:.1f}%) "
                f"- Loss: {loss:.4f}, LR: {learning_rate:.6f}"
            )
    
    def log_validation(self, metrics: dict) -> None:
        """
        Log validation metrics.
        
        Args:
            metrics: Dictionary of validation metrics
        """
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Validation: {metrics_str}")
    
    def log_checkpoint(self, path: str) -> None:
        """
        Log checkpoint save.
        
        Args:
            path: Path to saved checkpoint
        """
        self.logger.info(f"Saved checkpoint to {path}")
    
    def log_memory(self, prefix: str = "") -> None:
        """
        Log memory usage.
        
        Args:
            prefix: Prefix string for the log message
        """
        try:
            import torch
            import psutil
            
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            
            # System memory
            system_mem = psutil.virtual_memory()
            
            # Log memory info
            mem_str = (
                f"{prefix} Memory - "
                f"System: {system_mem.percent}% used, "
                f"Process: {mem_info.rss / (1024 ** 3):.2f} GB"
            )
            
            # Add GPU info if available
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                gpu_mem_cached = torch.cuda.memory_reserved() / (1024 ** 3)
                mem_str += f", GPU allocated: {gpu_mem_alloc:.2f} GB, cached: {gpu_mem_cached:.2f} GB"
            
            self.logger.info(mem_str)
        except ImportError:
            self.logger.warning("Memory logging requires psutil. Install with 'pip install psutil'")
        except Exception as e:
            self.logger.warning(f"Memory logging failed: {e}")