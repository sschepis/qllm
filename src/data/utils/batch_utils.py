"""
Batch Utilities for QLLM.

This module provides utility functions for handling batches of data,
including batch creation, manipulation, and preprocessing operations.
"""

import math
import logging
from typing import Dict, Any, List, Tuple, Union, Optional, Callable, Iterable

import torch
import numpy as np


logger = logging.getLogger("qllm.data")


def create_dynamic_batches(
    data: List[Any],
    batch_size: int,
    get_size_fn: Optional[Callable[[Any], int]] = None,
    max_tokens: Optional[int] = None,
    drop_last: bool = False
) -> List[List[Any]]:
    """
    Create dynamically sized batches based on token count.
    
    This is useful for efficient processing of variable-length sequences,
    as it avoids wasting computation on padding. It tries to maximize
    batch utilization while respecting max_tokens constraint.
    
    Args:
        data: List of data samples
        batch_size: Maximum number of samples per batch
        get_size_fn: Function to get the size (token count) of a sample
        max_tokens: Maximum number of tokens per batch (if None, only batch_size is used)
        drop_last: Whether to drop the last batch if it's smaller than batch_size
        
    Returns:
        List of batches, where each batch is a list of samples
    """
    # If no size function provided, default to a constant size of 1
    if get_size_fn is None:
        get_size_fn = lambda x: 1
    
    # Initialize result
    batches = []
    current_batch = []
    current_batch_size = 0
    current_batch_tokens = 0
    
    # Process each sample
    for sample in data:
        sample_tokens = get_size_fn(sample)
        
        # Start a new batch if adding this sample would exceed limits
        if (len(current_batch) >= batch_size or
            (max_tokens is not None and current_batch_tokens + sample_tokens > max_tokens)):
            # Add the current batch to results if not empty
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_size = 0
                current_batch_tokens = 0
        
        # Add sample to current batch
        current_batch.append(sample)
        current_batch_size += 1
        current_batch_tokens += sample_tokens
    
    # Add the last batch if not empty and not dropping last
    if current_batch and (not drop_last or len(current_batch) == batch_size):
        batches.append(current_batch)
    
    return batches


def pad_batch(
    batch: List[List[int]],
    pad_value: int = 0,
    max_length: Optional[int] = None,
    padding_side: str = "right"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a batch of sequences to the same length.
    
    Args:
        batch: List of integer sequences
        pad_value: Value to use for padding
        max_length: Maximum length to pad to (None for length of longest sequence)
        padding_side: Whether to pad on the "left" or "right"
        
    Returns:
        Tuple of (padded_batch, attention_mask)
    """
    # Find the maximum length if not provided
    if max_length is None:
        max_length = max(len(seq) for seq in batch)
    
    # Initialize padded batch and attention mask
    padded_batch = []
    attention_mask = []
    
    # Pad each sequence
    for seq in batch:
        seq_length = len(seq)
        padding_length = max_length - seq_length
        
        if padding_side == "right":
            padded_seq = seq + [pad_value] * padding_length
            mask = [1] * seq_length + [0] * padding_length
        else:  # padding_side == "left"
            padded_seq = [pad_value] * padding_length + seq
            mask = [0] * padding_length + [1] * seq_length
        
        padded_batch.append(padded_seq)
        attention_mask.append(mask)
    
    # Convert to tensors
    return torch.tensor(padded_batch), torch.tensor(attention_mask)


def pack_sequences(
    sequences: List[torch.Tensor],
    lengths: List[int]
) -> torch.nn.utils.rnn.PackedSequence:
    """
    Pack a batch of sequences for RNN processing.
    
    Args:
        sequences: List of sequence tensors
        lengths: List of sequence lengths
        
    Returns:
        PackedSequence object
    """
    # Sort sequences by length in descending order
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    sorted_sequences = [sequences[i] for i in sorted_indices]
    sorted_lengths = [lengths[i] for i in sorted_indices]
    
    # Pad sequences to the same length
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        sorted_sequences, batch_first=True
    )
    
    # Pack padded sequences
    packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(
        padded_sequences, sorted_lengths, batch_first=True
    )
    
    return packed_sequences


def unpack_sequences(
    packed_sequences: torch.nn.utils.rnn.PackedSequence,
    total_length: Optional[int] = None
) -> Tuple[torch.Tensor, List[int]]:
    """
    Unpack a batch of sequences from RNN processing.
    
    Args:
        packed_sequences: PackedSequence object
        total_length: Total length to pad to (None for length of longest sequence)
        
    Returns:
        Tuple of (padded_sequences, sequence_lengths)
    """
    # Unpack sequences
    padded_sequences, sequence_lengths = torch.nn.utils.rnn.pad_packed_sequence(
        packed_sequences, batch_first=True, total_length=total_length
    )
    
    return padded_sequences, sequence_lengths.tolist()


def create_sliding_window_batches(
    data: List[Any],
    window_size: int,
    stride: int = 1,
    drop_remainder: bool = False
) -> List[List[Any]]:
    """
    Create batches using a sliding window approach.
    
    This is useful for tasks like language modeling where we want to
    process overlapping sequences of data.
    
    Args:
        data: List of data samples
        window_size: Size of the sliding window
        stride: Step size for the window
        drop_remainder: Whether to drop the last batch if it's smaller than window_size
        
    Returns:
        List of batches created using sliding windows
    """
    # Initialize result
    batches = []
    
    # Calculate the number of windows
    n_samples = len(data)
    n_windows = ((n_samples - window_size) // stride + 1) if n_samples >= window_size else 0
    
    # Create sliding windows
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        batch = data[start_idx:end_idx]
        batches.append(batch)
    
    # Handle the remaining samples
    if not drop_remainder and n_samples > n_windows * stride:
        remaining = data[n_windows * stride:]
        if len(remaining) > 0:
            # Pad the last batch with the final element if needed
            while len(remaining) < window_size:
                remaining.append(remaining[-1])
            batches.append(remaining)
    
    return batches


def merge_batches(batches: List[List[Any]]) -> List[Any]:
    """
    Merge multiple batches into a single list.
    
    Args:
        batches: List of batches, where each batch is a list of samples
        
    Returns:
        Merged list of all samples
    """
    # Flatten the list of batches
    return [sample for batch in batches for sample in batch]


def shuffle_batch(batch: List[Any], seed: Optional[int] = None) -> List[Any]:
    """
    Shuffle the samples within a batch.
    
    Args:
        batch: List of samples to shuffle
        seed: Optional random seed for reproducibility
        
    Returns:
        Shuffled batch
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Create a shuffled copy of the batch
    shuffled_batch = batch.copy()
    np.random.shuffle(shuffled_batch)
    
    return shuffled_batch


def batch_to_tensors(
    batch: List[Dict[str, Any]],
    tensor_keys: List[str],
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Convert dictionary batches to tensors.
    
    Args:
        batch: List of dictionaries, where each dictionary is a sample
        tensor_keys: List of keys to convert to tensors
        device: Optional device to move tensors to
        
    Returns:
        Dictionary mapping keys to tensor batches
    """
    result = {}
    
    for key in tensor_keys:
        # Skip if key not in all samples
        if not all(key in sample for sample in batch):
            continue
        
        # Extract values
        values = [sample[key] for sample in batch]
        
        # Convert to tensors
        if all(isinstance(v, (int, float)) for v in values):
            # Numeric values
            tensor = torch.tensor(values)
        elif all(isinstance(v, (list, tuple)) and all(isinstance(i, (int, float)) for i in v) for v in values):
            # Lists/tuples of numeric values
            tensor = torch.tensor(values)
        elif all(isinstance(v, np.ndarray) for v in values):
            # NumPy arrays
            tensor = torch.tensor(np.stack(values))
        elif all(isinstance(v, torch.Tensor) for v in values):
            # Already tensors
            tensor = torch.stack(values)
        else:
            # Skip if values can't be converted to tensor
            logger.warning(f"Values for key {key} couldn't be converted to tensor")
            continue
        
        # Move to device if specified
        if device is not None:
            tensor = tensor.to(device)
        
        result[key] = tensor
    
    return result


def get_batch_statistics(
    batch: List[Any],
    size_fn: Optional[Callable[[Any], int]] = None
) -> Dict[str, Any]:
    """
    Calculate statistics about a batch.
    
    Args:
        batch: List of samples
        size_fn: Function to get the size of a sample
        
    Returns:
        Dictionary of batch statistics
    """
    # Calculate basic statistics
    batch_size = len(batch)
    
    # Use size function if provided
    if size_fn is not None:
        sizes = [size_fn(sample) for sample in batch]
        size_mean = sum(sizes) / batch_size if batch_size > 0 else 0
        size_min = min(sizes) if batch_size > 0 else 0
        size_max = max(sizes) if batch_size > 0 else 0
        size_std = math.sqrt(sum((s - size_mean) ** 2 for s in sizes) / batch_size) if batch_size > 0 else 0
    else:
        size_mean = size_min = size_max = size_std = None
    
    # Return statistics
    return {
        "batch_size": batch_size,
        "size_mean": size_mean,
        "size_min": size_min,
        "size_max": size_max,
        "size_std": size_std
    }