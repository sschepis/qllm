"""
Logging utilities for the Quantum Resonance Language Model.
Provides structured, configurable logging for training, evaluation,
and other operations.
"""

import logging
import os
import sys
import time
from typing import Optional

# Custom log levels
VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log messages.
    """
    COLORS = {
        'DEBUG': '\033[0;36m',     # Cyan
        'VERBOSE': '\033[0;34m',   # Blue
        'INFO': '\033[0;32m',      # Green
        'WARNING': '\033[0;33m',   # Yellow
        'ERROR': '\033[0;31m',     # Red
        'CRITICAL': '\033[1;31m',  # Bold Red
        'RESET': '\033[0m',        # Reset color
    }

    def format(self, record):
        log_message = super().format(record)
        level_name = record.levelname
        
        # Add color if terminal supports it and not redirected
        if sys.stdout.isatty() and level_name in self.COLORS:
            return f"{self.COLORS[level_name]}{log_message}{self.COLORS['RESET']}"
        else:
            return log_message


def setup_logger(
    name: str = "quantum_resonance",
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    log_format: str = "%(asctime)s - %(levelname)s - %(message)s",
    date_format: str = "%m/%d/%Y %H:%M:%S",
    file_level: Optional[int] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        log_level: Console logging level
        log_format: Format for log messages
        date_format: Format for timestamps
        file_level: Separate logging level for file (defaults to log_level)
        
    Returns:
        Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels in handlers
    
    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = ColoredFormatter(log_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if requested
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level if file_level is not None else log_level)
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Add verbose method
    def verbose(self, message, *args, **kwargs):
        self.log(VERBOSE, message, *args, **kwargs)
    
    logging.Logger.verbose = verbose
    
    return logger


def get_default_logger():
    """
    Get the default logger (creates it if doesn't exist).
    
    Returns:
        Logger: Default quantum_resonance logger
    """
    logger = logging.getLogger("quantum_resonance")
    
    # If logger has no handlers, set up the default
    if not logger.handlers:
        return setup_logger()
    
    return logger


class TrainingLogger:
    """
    Specialized logger for training with progress tracking.
    """
    def __init__(self, logger=None, log_interval=10):
        """
        Initialize training logger.
        
        Args:
            logger: Logger instance (creates default if None)
            log_interval: How often to log detailed info (in batches)
        """
        self.logger = logger or get_default_logger()
        self.log_interval = log_interval
        self.epoch_start_time = None
        self.batch_start_time = None
        self.global_step = 0
        
    def start_epoch(self, epoch, total_epochs):
        """Log start of epoch"""
        self.epoch_start_time = time.time()
        self.logger.info(f"Starting epoch {epoch+1}/{total_epochs}")
        
    def end_epoch(self, epoch, epoch_loss, val_metrics=None):
        """Log end of epoch with metrics"""
        epoch_time = time.time() - self.epoch_start_time
        self.logger.info(f"Epoch {epoch+1} completed in {epoch_time:.1f}s. "
                        f"Average loss: {epoch_loss:.4f}")
        
        if val_metrics:
            self.logger.info("Validation metrics:")
            for key, value in val_metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value}")
    
    def log_batch(self, epoch, batch_idx, batch_size, num_batches, loss, lr=None):
        """Log batch metrics, showing progress bar in integer percentages"""
        self.global_step += 1
        
        # Calculate progress percentage
        progress = 100. * batch_idx / num_batches
        
        # Only log at intervals to avoid spamming
        if batch_idx % self.log_interval == 0:
            # Calculate estimated time remaining
            if self.batch_start_time is not None:
                batch_time = time.time() - self.batch_start_time
                remaining_batches = num_batches - batch_idx
                eta = remaining_batches * batch_time
                eta_str = f", ETA: {eta:.1f}s"
            else:
                eta_str = ""
            
            # Log with progress bar
            log_msg = f"Epoch {epoch+1} [{batch_idx}/{num_batches} ({progress:.0f}%)]"
            log_msg += f" Loss: {loss:.4f}"
            
            if lr is not None:
                log_msg += f", LR: {lr:.6f}"
                
            log_msg += eta_str
            
            self.logger.info(log_msg)
        
        # Update batch start time
        self.batch_start_time = time.time()
    
    def log_validation(self, metrics):
        """Log validation metrics"""
        log_msg = "Validation results: "
        
        # Format each metric
        formatted_metrics = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_metrics.append(f"{key}={value:.4f}")
            else:
                formatted_metrics.append(f"{key}={value}")
        
        log_msg += ", ".join(formatted_metrics)
        self.logger.info(log_msg)
    
    def log_checkpoint(self, path):
        """Log checkpoint saving"""
        self.logger.info(f"Saving checkpoint to {path}")
    
    def log_error(self, error, context=""):
        """Log error with context"""
        if context:
            self.logger.error(f"{context}: {error}")
        else:
            self.logger.error(f"{error}")
    
    def log_memory(self, prefix=""):
        """Log memory usage if running with CUDA"""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
                
                msg = f"{prefix} CUDA Memory: {allocated:.1f}MB"
                msg += f" (peak: {max_allocated:.1f}MB)"
                self.logger.verbose(msg)
        except:
            pass