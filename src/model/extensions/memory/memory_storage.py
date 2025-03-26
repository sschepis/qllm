"""
Memory Storage Module.

This module provides functionality for storing and retrieving data in memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import OrderedDict

from .memory_utils import normalize_keys, compute_similarity, get_top_k_results
from .memory_tracking import MemoryTracker


class MemoryStorage:
    """
    Manages storage and retrieval of data in memory.
    
    This class handles the core tensor operations for storing keys and values,
    as well as finding and retrieving entries.
    """
    
    def __init__(
        self,
        memory_size: int,
        key_dim: int,
        value_dim: int,
        device: Optional[torch.device] = None
    ):
        """
        Initialize memory storage.
        
        Args:
            memory_size: Maximum number of entries in memory
            key_dim: Dimension of memory keys
            value_dim: Dimension of memory values
            device: Device to store tensors on
        """
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Create tensors for memory storage
        self.memory_keys = torch.zeros(memory_size, key_dim)
        self.memory_values = torch.zeros(memory_size, value_dim)
        self.valid_mask = torch.zeros(memory_size, dtype=torch.bool)
        
        # Move tensors to device if specified
        if device is not None:
            self.memory_keys = self.memory_keys.to(device)
            self.memory_values = self.memory_values.to(device)
            self.valid_mask = self.valid_mask.to(device)
        
        # Create memory tracker
        self.tracker = MemoryTracker()
        
        # Create normalization and projection layers
        self.key_norm = nn.LayerNorm(key_dim)
        self.key_projection = nn.Linear(value_dim, key_dim)
        self.value_projection = nn.Identity()
    
    def clear(self) -> None:
        """Clear all memory entries."""
        # Reset tensors
        self.memory_keys.zero_()
        self.memory_values.zero_()
        self.valid_mask.fill_(False)
        
        # Reset tracker
        self.tracker.reset()
    
    def add_entries(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add new entries to memory.
        
        Args:
            keys: Keys for the new entries [batch_size, key_dim]
            values: Values for the new entries [batch_size, value_dim]
            indices: Optional specific indices to place entries (if None, will find slots)
            
        Returns:
            Tensor of indices where entries were added
        """
        batch_size = keys.size(0)
        
        # Normalize keys
        keys = normalize_keys(keys)
        
        # Find slots for new entries if indices not provided
        if indices is None:
            indices = self.find_replacement_indices(batch_size)
        
        # Add entries to memory
        for i, idx in enumerate(indices):
            if i < batch_size:
                # Add key, value, and mark as valid
                self.memory_keys[idx] = keys[i]
                self.memory_values[idx] = values[i]
                self.valid_mask[idx] = True
                
                # Initialize tracking for new entry
                self.tracker.initialize_entry(idx.item())
        
        return indices
    
    def retrieve_entries(
        self,
        query_keys: torch.Tensor,
        top_k: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve entries from memory based on query keys.
        
        Args:
            query_keys: Query keys to search for [batch_size, key_dim]
            top_k: Number of top matches to retrieve
            
        Returns:
            Tuple of (retrieved_values, similarity_scores, indices)
        """
        # Normalize query keys
        query_keys = normalize_keys(query_keys)
        
        # Compute similarity with all memory keys
        similarities = compute_similarity(query_keys, self.memory_keys, self.valid_mask)
        
        # Get top-k results
        values, scores, indices = get_top_k_results(
            similarities, self.memory_values, top_k
        )
        
        # Update access statistics
        if indices.size(1) > 0:  # Ensure we have at least one result
            # Convert to list of indices (take first match for each batch item)
            accessed_indices = [idx.item() for idx in indices[:, 0]]
            self.tracker.update_access_stats(accessed_indices)
        
        return values, scores, indices
    
    def update_entries(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
    ) -> bool:
        """
        Update existing memory entries.
        
        Args:
            indices: Indices of entries to update
            values: New values for the entries
            
        Returns:
            True if update was successful
        """
        batch_size = indices.size(0)
        
        # Update values
        for i, idx in enumerate(indices):
            if i < batch_size and self.valid_mask[idx]:
                self.memory_values[idx] = values[i]
        
        return True
    
    def find_replacement_indices(self, num_needed: int) -> torch.Tensor:
        """
        Find indices to replace in memory when it's full.
        
        Args:
            num_needed: Number of slots needed
            
        Returns:
            Tensor of indices to replace
        """
        if not self.valid_mask.any():
            # If no valid entries, return the first n indices
            return torch.arange(min(num_needed, self.memory_size), 
                              device=self.memory_keys.device)
        
        # If memory isn't full yet, return the first n invalid indices
        if (~self.valid_mask).sum() >= num_needed:
            invalid_indices = torch.where(~self.valid_mask)[0]
            return invalid_indices[:num_needed]
        
        # If we need to replace valid entries, use importance scores
        valid_indices_set = {i for i in range(self.memory_size) if self.valid_mask[i]}
        least_important = self.tracker.get_least_important_indices(
            num_needed, valid_indices_set
        )
        
        return torch.tensor(least_important, device=self.memory_keys.device)
    
    def get_tensor_state(self) -> Dict[str, torch.Tensor]:
        """
        Get the tensor state for persistence.
        
        Returns:
            Dictionary of tensors representing the current state
        """
        return {
            "memory_keys": self.memory_keys,
            "memory_values": self.memory_values,
            "valid_mask": self.valid_mask
        }
    
    def set_tensor_state(self, state: Dict[str, torch.Tensor]) -> None:
        """
        Set the tensor state from loaded data.
        
        Args:
            state: Dictionary of tensors to restore
        """
        self.memory_keys.copy_(state["memory_keys"])
        self.memory_values.copy_(state["memory_values"])
        self.valid_mask.copy_(state["valid_mask"])
    
    def get_tracker_state(self) -> Dict[str, Any]:
        """
        Get the tracker state for persistence.
        
        Returns:
            Dictionary representing the current tracker state
        """
        return {
            "access_counts": self.tracker.access_counts,
            "last_access_time": self.tracker.last_access_time,
            "importance_scores": self.tracker.importance_scores
        }
    
    def set_tracker_state(self, state: Dict[str, Any]) -> None:
        """
        Set the tracker state from loaded data.
        
        Args:
            state: Dictionary representing tracker state to restore
        """
        self.tracker.access_counts = state["access_counts"]
        self.tracker.last_access_time = state["last_access_time"]
        self.tracker.importance_scores = state["importance_scores"]
    
    def get_valid_count(self) -> int:
        """
        Get the number of valid entries in memory.
        
        Returns:
            Count of valid entries
        """
        return self.valid_mask.sum().item()
    
    def to(self, device: torch.device) -> 'MemoryStorage':
        """
        Move memory storage to specified device.
        
        Args:
            device: Device to move tensors to
            
        Returns:
            Self for chaining
        """
        self.memory_keys = self.memory_keys.to(device)
        self.memory_values = self.memory_values.to(device)
        self.valid_mask = self.valid_mask.to(device)
        self.key_norm = self.key_norm.to(device)
        self.key_projection = self.key_projection.to(device)
        self.value_projection = self.value_projection.to(device)
        
        return self