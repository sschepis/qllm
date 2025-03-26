"""
Memory Extension Utilities Module.

This module provides utility functions for memory extensions.
"""

import time
from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import torch.nn.functional as F


def calculate_importance_score(
    access_count: int,
    last_access_time: float,
    current_time: Optional[float] = None,
    frequency_weight: float = 0.7,
    recency_weight: float = 0.3
) -> float:
    """
    Calculate importance score for a memory entry based on access patterns.
    
    Args:
        access_count: Number of times the entry has been accessed
        last_access_time: Timestamp of the last access
        current_time: Current timestamp (defaults to time.time())
        frequency_weight: Weight for frequency component (0-1)
        recency_weight: Weight for recency component (0-1)
        
    Returns:
        Importance score (higher is more important)
    """
    if current_time is None:
        current_time = time.time()
    
    # Recency: inverse of time since last access (newer = higher score)
    time_diff = max(1.0, current_time - last_access_time)
    recency = 1.0 / time_diff
    
    # Combine frequency and recency
    importance = frequency_weight * access_count + recency_weight * recency
    
    return importance


def normalize_keys(keys: torch.Tensor) -> torch.Tensor:
    """
    Normalize keys for memory operations.
    
    Args:
        keys: Tensor of keys to normalize
        
    Returns:
        Normalized keys tensor
    """
    return F.normalize(keys, dim=1)


def compute_similarity(
    query_keys: torch.Tensor,
    memory_keys: torch.Tensor,
    valid_mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute similarity between query keys and memory keys.
    
    Args:
        query_keys: Tensor of query keys [batch_size, key_dim]
        memory_keys: Tensor of memory keys [memory_size, key_dim]
        valid_mask: Boolean mask for valid memory entries [memory_size]
        
    Returns:
        Similarity scores [batch_size, memory_size]
    """
    # Compute similarity with all memory keys
    similarities = torch.matmul(query_keys, memory_keys.t())  # [batch_size, memory_size]
    
    # Mask out invalid entries if a mask is provided
    if valid_mask is not None:
        similarities = similarities.masked_fill(~valid_mask, -float('inf'))
    
    return similarities


def get_top_k_results(
    similarities: torch.Tensor,
    memory_values: torch.Tensor,
    top_k: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get top-k results from similarity scores.
    
    Args:
        similarities: Similarity scores [batch_size, memory_size]
        memory_values: Memory values [memory_size, value_dim]
        top_k: Number of top results to return
        
    Returns:
        Tuple of (top_values, top_similarities, top_indices)
    """
    # Get top-k similar entities
    top_similarities, top_indices = torch.topk(
        similarities, 
        k=min(top_k, similarities.size(1)), 
        dim=1
    )
    
    # Get corresponding values
    top_values = memory_values[top_indices]  # [batch_size, top_k, value_dim]
    
    return top_values, top_similarities, top_indices


def update_memory_stats(
    stats: Dict[str, Any],
    total_entries: int,
    memory_size: int
) -> Dict[str, Any]:
    """
    Update memory statistics.
    
    Args:
        stats: Dictionary of current statistics
        total_entries: Total number of valid entries
        memory_size: Total memory capacity
        
    Returns:
        Updated statistics dictionary
    """
    stats["total_entries"] = total_entries
    
    # Calculate hit rate if retrievals have been made
    if stats.get("total_retrievals", 0) > 0:
        stats["hit_rate"] = stats.get("total_hits", 0) / stats["total_retrievals"]
    
    # Calculate memory usage (percentage of capacity used)
    stats["memory_usage"] = total_entries / memory_size
    
    return stats