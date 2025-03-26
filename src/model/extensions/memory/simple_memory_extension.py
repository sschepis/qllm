"""
Simple Memory Extension Module.

This module provides a concrete implementation of BaseMemoryExtension
with basic key-value memory functionality.
"""

import torch
from typing import Dict, Any, Optional, List, Tuple

from .base_memory_core import BaseMemoryExtension


class SimpleMemoryExtension(BaseMemoryExtension):
    """
    Simple memory extension implementation.
    
    This class implements the abstract methods of BaseMemoryExtension
    with straightforward memory operations for storing and retrieving knowledge.
    """
    
    def add_to_memory(self, 
                     keys: torch.Tensor, 
                     values: torch.Tensor, 
                     metadata: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Add new entries to memory.
        
        Args:
            keys: Keys for the new entries
            values: Values for the new entries
            metadata: Additional metadata for the entries
            
        Returns:
            Indices of the added entries
        """
        # Use storage to add entries
        indices = self.storage.add_entries(keys, values)
        
        # Update statistics
        self.stats["total_entries"] = self.storage.get_valid_count()
        self.stats["total_updates"] += 1
        
        # Update memory persistence if needed
        self.update_memory_if_needed()
        
        return indices
    
    def retrieve_from_memory(self, 
                           query_keys: torch.Tensor, 
                           top_k: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve entries from memory based on query keys.
        
        Args:
            query_keys: Query keys to search for
            top_k: Number of top matches to retrieve
            
        Returns:
            Tuple of (retrieved_values, similarity_scores, indices)
        """
        # Use storage to retrieve entries
        values, scores, indices = self.storage.retrieve_entries(query_keys, top_k)
        
        # Update statistics
        self.stats["total_hits"] = self.stats.get("total_hits", 0) + query_keys.size(0)
        
        return values, scores, indices
    
    def update_memory(self, 
                     indices: torch.Tensor, 
                     values: torch.Tensor, 
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update existing memory entries.
        
        Args:
            indices: Indices of entries to update
            values: New values for the entries
            metadata: Additional metadata to update
            
        Returns:
            True if update was successful
        """
        # Use storage to update entries
        success = self.storage.update_entries(indices, values)
        
        # Update statistics if successful
        if success:
            self.stats["total_updates"] += 1
            
            # Update memory persistence if needed
            self.update_memory_if_needed()
        
        return success