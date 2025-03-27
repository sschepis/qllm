"""
Tensor collate functions for batching data.

This module provides collate functions for batching data from different datasets,
including dialogue datasets, which have specialized tensor formats.
"""

import torch
from typing import Dict, List, Any, Union, Tuple


def dialogue_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for dialogue batches.
    
    Args:
        batch: List of tokenized dialogues
        
    Returns:
        Batched tensors
    """
    # Check for empty batch
    if not batch:
        return {}
    
    # Get keys from the first element
    keys = batch[0].keys()
    
    # Initialize result dictionary
    result = {}
    
    # Stack tensors for each key
    for key in keys:
        # Collect tensors for this key
        tensors = [item[key] for item in batch]
        
        # Stack tensors into a batch
        if all(isinstance(t, torch.Tensor) for t in tensors):
            result[key] = torch.stack(tensors)
        else:
            # Handle non-tensor data (e.g., lists, scalars)
            result[key] = tensors
    
    return result


def default_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Default collate function for general data batches.
    
    Args:
        batch: List of data items with dictionary structure
        
    Returns:
        Batched data
    """
    # Check for empty batch
    if not batch:
        return {}
    
    # Get keys from the first element
    keys = batch[0].keys()
    
    # Initialize result dictionary
    result = {}
    
    # Process each key
    for key in keys:
        # Collect values for this key
        values = [item[key] for item in batch]
        
        # Handle different types
        if all(isinstance(v, torch.Tensor) for v in values):
            # Stack tensors
            result[key] = torch.stack(values)
        elif all(isinstance(v, (int, float)) for v in values):
            # Convert numbers to tensor
            result[key] = torch.tensor(values)
        else:
            # Keep as list for other types
            result[key] = values
    
    return result


def tensor_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for simple tensor pairs (e.g., input and target tensors).
    
    Args:
        batch: List of (input, target) tensor pairs
        
    Returns:
        Tuple of batched input and target tensors
    """
    # Split batch into inputs and targets
    inputs, targets = zip(*batch)
    
    # Stack inputs and targets
    inputs_batch = torch.stack(inputs)
    targets_batch = torch.stack(targets)
    
    return inputs_batch, targets_batch