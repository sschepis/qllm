"""
Cache utilities for data handling.

This module provides functions for caching and retrieving datasets to improve
performance and reduce network usage when working with large datasets.
"""

import os
import pickle
import logging

logger = logging.getLogger("qllm_dataloaders")

def setup_cache_dir(cache_dir=None):
    """
    Ensure the cache directory exists.
    
    Args:
        cache_dir: Optional path to the cache directory
        
    Returns:
        Path to the cache directory
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "qllm")
    
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def load_from_cache(cached_path):
    """
    Load a dataset from cache if it exists.
    
    Args:
        cached_path: Path to the cached file
        
    Returns:
        The cached dataset if found, None otherwise
    """
    try:
        if os.path.exists(cached_path):
            logger.info(f"Loading cached dataset from {cached_path}")
            with open(cached_path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading from cache: {e}")
    
    return None

def save_to_cache(data, cached_path):
    """
    Save a dataset to cache.
    
    Args:
        data: The dataset to save
        cached_path: Path where the dataset should be saved
        
    Returns:
        None
    """
    try:
        logger.info(f"Saving dataset to cache: {cached_path}")
        os.makedirs(os.path.dirname(cached_path), exist_ok=True)
        with open(cached_path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.error(f"Error saving to cache: {e}")