"""
Batch handling utilities for safe tensor processing.
"""

import torch
import logging
from typing import Dict, List, Any, Union

# Get logger
logger = logging.getLogger("quantum_resonance")


def batch_to_device(batch, device):
    """
    Safely convert batch to tensors and move to device.
    
    This is a defensive implementation that handles various types of batch formats
    and ensures they're properly moved to the correct device.
    
    Args:
        batch: Input batch that may contain lists or tensors
        device: Target device to move tensors to
        
    Returns:
        Dict with all values converted to tensors on the target device
    """
    if not isinstance(batch, dict):
        logger.error(f"Unexpected batch type: {type(batch)}")
        if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], dict):
            logger.warning("Batch appears to be a list of dicts - using first item")
            batch = batch[0]
        else:
            raise ValueError(f"Cannot process batch of type {type(batch)}")
    
    result = {}
    for k, v in batch.items():
        if k not in ["input_ids", "attention_mask", "labels"]:
            continue
            
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, list):
            logger.warning(f"Converting {k} from list to tensor (collate_fn should handle this)")
            try:
                # Handle nested lists if necessary
                if v and isinstance(v[0], list):
                    result[k] = torch.tensor(v, device=device)
                else:
                    result[k] = torch.tensor(v, device=device)
            except Exception as e:
                logger.error(f"Failed to convert {k} list to tensor: {e}")
                # Last resort: try to cast individual elements
                tensors = []
                for item in v:
                    if isinstance(item, torch.Tensor):
                        tensors.append(item.to(device))
                    else:
                        tensors.append(torch.tensor(item, device=device))
                result[k] = torch.stack(tensors)
        else:
            logger.error(f"Unexpected type for {k}: {type(v)}")
            try:
                result[k] = torch.tensor(v, device=device)
            except:
                logger.error(f"Could not convert {k} to tensor, skipping")
                
    return result


def debug_batch_contents(batch, max_display=200):
    """
    Debug helper to print detailed batch contents, even for complex nested structures.
    
    Args:
        batch: Batch data to debug
        max_display: Maximum characters to display for each value
    """
    logger.debug("Detailed batch contents:")
    
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            # For tensors, show shape, dtype, and some values
            logger.debug(f"  {k}: tensor shape={v.shape}, dtype={v.dtype}")
            try:
                # Try to show some values safely
                flat_vals = v.flatten()[:5].cpu().tolist()
                logger.debug(f"    First 5 values: {flat_vals}")
            except:
                logger.debug("    (Cannot display values)")
        elif isinstance(v, list):
            # For lists, show length and some structure info
            logger.debug(f"  {k}: list length={len(v)}")
            if v:
                if isinstance(v[0], list):
                    nested_lens = [len(x) for x in v[:3]]
                    logger.debug(f"    Nested list structure: {nested_lens}")
                else:
                    logger.debug(f"    First few elements: {str(v[:3])[:max_display]}")
        else:
            # For other types, just show the type and string representation
            logger.debug(f"  {k}: {type(v).__name__} = {str(v)[:max_display]}")