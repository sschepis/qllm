"""
Memory Tracking Module.

This module provides functionality for tracking memory access patterns and
calculating importance scores for memory entries.
"""

import time
from typing import Dict, Any, Optional, List, Set
from collections import OrderedDict

from .memory_utils import calculate_importance_score


class MemoryTracker:
    """
    Tracks memory access patterns and importance for memory management.
    
    This class is responsible for tracking how memory entries are used,
    including access frequency, recency, and importance scoring.
    """
    
    def __init__(self):
        """Initialize memory tracker."""
        # Memory usage tracking
        self.access_counts = OrderedDict()  # Track access frequency
        self.last_access_time = OrderedDict()  # Track recency
        self.importance_scores = OrderedDict()  # Track importance
    
    def reset(self):
        """Reset all tracking counters and scores."""
        self.access_counts.clear()
        self.last_access_time.clear()
        self.importance_scores.clear()
    
    def update_access_stats(self, indices: List[int]) -> None:
        """
        Update access statistics for memory entries.
        
        Args:
            indices: List of indices of accessed entries
        """
        current_time = time.time()
        
        for idx in indices:
            idx_str = str(idx)
            # Update access count
            self.access_counts[idx_str] = self.access_counts.get(idx_str, 0) + 1
            # Update last access time
            self.last_access_time[idx_str] = current_time
            
        # Remove from OrderedDict and add back to maintain order by recency
        for idx_str in [str(idx) for idx in indices]:
            if idx_str in self.access_counts:
                count = self.access_counts.pop(idx_str)
                self.access_counts[idx_str] = count
    
    def initialize_entry(self, idx: int) -> None:
        """
        Initialize tracking for a new memory entry.
        
        Args:
            idx: Index of the new entry
        """
        idx_str = str(idx)
        self.access_counts[idx_str] = 0
        self.last_access_time[idx_str] = time.time()
        # We don't initialize importance score until it's needed
    
    def get_importance_score(self, idx: int) -> float:
        """
        Get importance score for a memory entry.
        
        Args:
            idx: Index of the entry
            
        Returns:
            Importance score (higher is more important)
        """
        idx_str = str(idx)
        
        # If we haven't calculated an importance score yet, compute it
        if idx_str not in self.importance_scores:
            # Get the necessary data
            frequency = self.access_counts.get(idx_str, 0)
            last_access = self.last_access_time.get(idx_str, 0)
            
            # Calculate importance score
            importance = calculate_importance_score(
                access_count=frequency,
                last_access_time=last_access
            )
            
            # Cache the result
            self.importance_scores[idx_str] = importance
        
        return self.importance_scores[idx_str]
    
    def invalidate_importance_scores(self, indices: Optional[List[int]] = None) -> None:
        """
        Invalidate cached importance scores to force recalculation.
        
        Args:
            indices: Optional list of indices to invalidate. If None, invalidates all.
        """
        if indices is None:
            self.importance_scores.clear()
        else:
            for idx in indices:
                idx_str = str(idx)
                if idx_str in self.importance_scores:
                    del self.importance_scores[idx_str]
    
    def get_least_important_indices(self, 
                                    num_needed: int, 
                                    valid_indices: Set[int]) -> List[int]:
        """
        Get the least important indices for replacement.
        
        Args:
            num_needed: Number of indices needed
            valid_indices: Set of valid indices to consider
            
        Returns:
            List of least important indices
        """
        # Calculate importance scores for all valid indices
        importance_dict = {
            idx: self.get_importance_score(idx)
            for idx in valid_indices
        }
        
        # Sort by importance (ascending)
        sorted_indices = sorted(importance_dict.keys(), key=lambda x: importance_dict[x])
        
        # Return the least important indices
        return sorted_indices[:num_needed]
    
    def get_order_by_recency(self, limit: Optional[int] = None) -> List[int]:
        """
        Get indices ordered by recency (most recent first).
        
        Args:
            limit: Optional limit on number of indices to return
            
        Returns:
            List of indices ordered by recency
        """
        # The OrderedDict maintains insertion order, which we update on access
        # So the most recently accessed items are at the end
        indices = [int(idx) for idx in self.access_counts.keys()]
        
        # Reverse to get most recent first
        indices.reverse()
        
        # Apply limit if specified
        if limit is not None:
            indices = indices[:limit]
            
        return indices
    
    def get_order_by_frequency(self, limit: Optional[int] = None) -> List[int]:
        """
        Get indices ordered by access frequency (most frequent first).
        
        Args:
            limit: Optional limit on number of indices to return
            
        Returns:
            List of indices ordered by frequency
        """
        # Sort by access count (descending)
        sorted_items = sorted(
            self.access_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        indices = [int(idx) for idx, _ in sorted_items]
        
        # Apply limit if specified
        if limit is not None:
            indices = indices[:limit]
            
        return indices