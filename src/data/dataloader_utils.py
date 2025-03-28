"""
Common utilities for data loaders.

This module provides common utilities for creating data loaders
that can be used across different dataset types.
"""

import os
import logging
import torch
from typing import Dict, Any, Optional, List

# Set up logging
logger = logging.getLogger("qllm_dataloaders")

def setup_cache_dir(cache_dir: Optional[str] = None) -> Optional[str]:
    """
    Set up the cache directory for datasets.
    
    Args:
        cache_dir: Optional cache directory path
        
    Returns:
        The cache directory path if specified and created, otherwise None
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using cache directory: {cache_dir}")
    return cache_dir

def get_default_eval_batch_size(batch_size: int, eval_batch_size: Optional[int] = None) -> int:
    """
    Get the evaluation batch size.
    
    Args:
        batch_size: Training batch size
        eval_batch_size: Optional evaluation batch size
        
    Returns:
        The evaluation batch size (defaults to batch_size if not specified)
    """
    return eval_batch_size if eval_batch_size is not None else batch_size

def save_to_cache(obj: Any, cache_path: str) -> None:
    """
    Save an object to the cache.
    
    Args:
        obj: Object to save
        cache_path: Path to save the object to
    """
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(obj, cache_path)
        logger.info(f"Cached object to {cache_path}")
    except Exception as e:
        logger.error(f"Error caching object to {cache_path}: {e}")

def load_from_cache(cache_path: str) -> Optional[Any]:
    """
    Load an object from the cache.
    
    Args:
        cache_path: Path to load the object from
        
    Returns:
        The loaded object, or None if loading failed
    """
    if not os.path.exists(cache_path):
        return None
    
    try:
        logger.info(f"Loading cached object from {cache_path}")
        obj = torch.load(cache_path)
        logger.info("Object loaded from cache successfully")
        return obj
    except Exception as e:
        logger.error(f"Error loading cached object: {e}")
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                logger.info("Removed corrupt cache file")
            except Exception:
                pass
        return None
        
def create_dataloaders(
    data_config: Any,
    tokenizer: Any,
    batch_size: int = 16,
    num_workers: int = 4
) -> Dict[str, Any]:
    """
    Create dataloaders based on data config.
    
    This is a compatibility function for the enhanced trainer system.
    
    Args:
        data_config: Data configuration
        tokenizer: Tokenizer to use
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary of data loaders
    """
    # Import here to avoid circular imports
    from src.data.dataloaders import get_appropriate_dataloaders
    
    # Get eval batch size if available
    eval_batch_size = getattr(data_config, "eval_batch_size", batch_size)
    if eval_batch_size is None:
        eval_batch_size = batch_size
        
    # Call the existing function
    return get_appropriate_dataloaders(
        data_config=data_config,
        tokenizer=tokenizer,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers
    )