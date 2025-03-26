"""
Tensor collation utilities for ensuring all batch data is properly converted to tensors.
"""

import torch
import logging
from typing import Dict, List, Any, Union

# Get logger
logger = logging.getLogger("quantum_resonance")


def tensor_collate_fn(batch):
    """
    Custom collate function to ensure proper tensor conversion.
    
    Args:
        batch: Batch of data from the dataset
        
    Returns:
        Dict[str, torch.Tensor]: Batch with all values converted to tensors
    """
    if not batch:
        return {}
    
    result = {}
    
    # Process each feature
    for key in batch[0].keys():
        # Extract values for this key from all items
        values = [item[key] for item in batch]
        
        # Check if values are already tensors
        if isinstance(values[0], torch.Tensor):
            # Stack tensors
            try:
                result[key] = torch.stack(values)
            except RuntimeError as e:
                logger.warning(f"Could not stack tensors for {key}: {e}")
                # Fallback to just returning the list of tensors
                result[key] = values
        else:
            # Convert lists to tensors
            try:
                result[key] = torch.tensor(values)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert {key} to tensor: {e}")
                # If conversion fails, keep as list but log a warning
                result[key] = values
    
    return result


def debug_batch_structure(batch, name="Batch"):
    """
    Debug helper to print batch structure and tensor types.
    
    Args:
        batch: Batch data to debug
        name: Name to use in log messages
    """
    logger.debug(f"{name} structure:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            logger.debug(f"  {k}: tensor, shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, list):
            list_type = type(v[0]).__name__ if v else "empty"
            list_len = len(v)
            logger.debug(f"  {k}: list of {list_type}, len={list_len}")
        else:
            logger.debug(f"  {k}: {type(v).__name__}")