"""
Base Memory Extension Module.

This module defines the base class for memory extensions in the
Semantic Resonance Language Model.
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Set
import abc
import time
from collections import OrderedDict

import torch
import torch.nn as nn

from ..base_extension import BaseExtension


class BaseMemoryExtension(BaseExtension):
    """
    Base class for all memory extensions.
    
    This class extends the BaseExtension to provide common functionality
    specific to memory management, such as storing, retrieving, and
    updating knowledge.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the memory extension.
        
        Args:
            name (str): Unique name for this extension instance
            config (Dict[str, Any]): Configuration dictionary for the extension
        """
        super().__init__(name, config)
        
        # Memory configuration
        self.memory_size = config.get("memory_size", 10000)
        self.key_dim = config.get("memory_key_dim", 128)
        self.value_dim = config.get("memory_value_dim", 768)
        
        # Memory usage tracking
        self.access_counts = OrderedDict()  # Track access frequency
        self.last_access_time = OrderedDict()  # Track recency
        self.importance_scores = OrderedDict()  # Track importance
        
        # Persistence settings
        self.persistence_enabled = config.get("persistence_enabled", False)
        self.persistence_path = config.get("persistence_path", "memory/knowledge_store.pt")
        self.persistence_interval = config.get("persistence_interval", 1000)
        self.update_counter = 0
        
        # Memory statistics
        self.stats = {
            "total_entries": 0,
            "total_retrievals": 0,
            "total_updates": 0,
            "hit_rate": 0.0,
            "memory_usage": 0.0,
        }
        
        # Initialize memory components
        self._initialize_memory_components()
    
    def _initialize_memory_components(self):
        """Initialize memory storage components."""
        # Default memory components - will be overridden by subclasses
        # with more sophisticated storage mechanisms
        
        # Memory keys and values as tensors
        self.register_buffer(
            "memory_keys", 
            torch.zeros(self.memory_size, self.key_dim)
        )
        self.register_buffer(
            "memory_values", 
            torch.zeros(self.memory_size, self.value_dim)
        )
        
        # Mask for valid entries
        self.register_buffer(
            "valid_mask",
            torch.zeros(self.memory_size, dtype=torch.bool)
        )
        
        # Key normalization layer
        self.key_norm = nn.LayerNorm(self.key_dim)
        
        # Memory projection layers
        self.key_projection = nn.Linear(self.value_dim, self.key_dim)
        self.value_projection = nn.Identity()
    
    def get_extension_type(self) -> str:
        """
        Get the type of this extension.
        
        Returns:
            str: Extension type
        """
        return "memory"
    
    @abc.abstractmethod
    def add_to_memory(self, 
                     keys: torch.Tensor, 
                     values: torch.Tensor, 
                     metadata: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Add new entries to memory.
        
        Args:
            keys (torch.Tensor): Keys for the new entries
            values (torch.Tensor): Values for the new entries
            metadata (Dict[str, Any], optional): Additional metadata for the entries
            
        Returns:
            torch.Tensor: Indices of the added entries
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def retrieve_from_memory(self, 
                           query_keys: torch.Tensor, 
                           top_k: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve entries from memory based on query keys.
        
        Args:
            query_keys (torch.Tensor): Query keys to search for
            top_k (int): Number of top matches to retrieve
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                Retrieved values, similarity scores, and indices
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def update_memory(self, 
                     indices: torch.Tensor, 
                     values: torch.Tensor, 
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update existing memory entries.
        
        Args:
            indices (torch.Tensor): Indices of entries to update
            values (torch.Tensor): New values for the entries
            metadata (Dict[str, Any], optional): Additional metadata to update
            
        Returns:
            bool: True if update was successful
        """
        raise NotImplementedError
    
    def clear_memory(self) -> None:
        """Clear all memory entries."""
        # Reset tensors
        self.memory_keys.zero_()
        self.memory_values.zero_()
        self.valid_mask.fill_(False)
        
        # Reset tracking dictionaries
        self.access_counts = OrderedDict()
        self.last_access_time = OrderedDict()
        self.importance_scores = OrderedDict()
        
        # Reset statistics
        self.stats["total_entries"] = 0
        self.update_counter = 0
        
        print(f"Memory cleared for extension {self.name}")
    
    def update_access_stats(self, indices: torch.Tensor) -> None:
        """
        Update access statistics for memory entries.
        
        Args:
            indices (torch.Tensor): Indices of accessed entries
        """
        current_time = time.time()
        
        for idx in indices.cpu().numpy():
            idx_str = str(idx)
            # Update access count
            self.access_counts[idx_str] = self.access_counts.get(idx_str, 0) + 1
            # Update last access time
            self.last_access_time[idx_str] = current_time
            
        # Remove from OrderedDict and add back to maintain order by recency
        for idx_str in [str(idx) for idx in indices.cpu().numpy()]:
            if idx_str in self.access_counts:
                count = self.access_counts.pop(idx_str)
                self.access_counts[idx_str] = count
    
    def get_importance_score(self, idx: int) -> float:
        """
        Get importance score for a memory entry.
        
        Args:
            idx (int): Index of the entry
            
        Returns:
            float: Importance score
        """
        idx_str = str(idx)
        
        # If we haven't calculated an importance score yet, compute it
        if idx_str not in self.importance_scores:
            # Default importance calculation based on frequency and recency
            frequency = self.access_counts.get(idx_str, 0)
            
            # Recency: inverse of time since last access (newer = higher score)
            current_time = time.time()
            last_access = self.last_access_time.get(idx_str, 0)
            recency = 1.0 / max(1.0, current_time - last_access)
            
            # Combine frequency and recency (simple weighted average)
            importance = 0.7 * frequency + 0.3 * recency
            self.importance_scores[idx_str] = importance
        
        return self.importance_scores[idx_str]
    
    def find_replacement_indices(self, num_needed: int) -> torch.Tensor:
        """
        Find indices to replace in memory when it's full.
        
        Args:
            num_needed (int): Number of slots needed
            
        Returns:
            torch.Tensor: Indices to replace
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
        importance_scores = torch.zeros_like(self.valid_mask, dtype=torch.float)
        
        # Fill in importance scores for valid entries
        for i in range(self.memory_size):
            if self.valid_mask[i]:
                importance_scores[i] = self.get_importance_score(i)
        
        # Find the lowest importance valid entries
        _, indices = torch.topk(importance_scores, k=num_needed, largest=False)
        return indices
    
    def update_memory_if_needed(self) -> None:
        """Check if memory should be persisted and do so if necessary."""
        if self.persistence_enabled:
            self.update_counter += 1
            
            if self.update_counter >= self.persistence_interval:
                self.persist_memory()
                self.update_counter = 0
    
    def persist_memory(self) -> None:
        """Save memory to disk."""
        if self.persistence_enabled:
            # Create a serializable memory state
            memory_state = {
                "memory_keys": self.memory_keys.cpu(),
                "memory_values": self.memory_values.cpu(),
                "valid_mask": self.valid_mask.cpu(),
                "access_counts": self.access_counts,
                "last_access_time": self.last_access_time,
                "importance_scores": self.importance_scores,
                "stats": self.stats,
                "update_counter": self.update_counter
            }
            
            try:
                torch.save(memory_state, self.persistence_path)
                print(f"Memory persisted to {self.persistence_path}")
            except Exception as e:
                print(f"Error persisting memory: {e}")
    
    def load_memory(self) -> bool:
        """
        Load memory from disk.
        
        Returns:
            bool: True if loading was successful
        """
        if not self.persistence_enabled:
            return False
        
        try:
            memory_state = torch.load(self.persistence_path)
            
            # Load the tensors
            self.memory_keys.copy_(memory_state["memory_keys"])
            self.memory_values.copy_(memory_state["memory_values"])
            self.valid_mask.copy_(memory_state["valid_mask"])
            
            # Load the dictionaries
            self.access_counts = memory_state["access_counts"]
            self.last_access_time = memory_state["last_access_time"]
            self.importance_scores = memory_state["importance_scores"]
            
            # Load the statistics
            self.stats = memory_state["stats"]
            self.update_counter = memory_state["update_counter"]
            
            print(f"Memory loaded from {self.persistence_path}")
            return True
        except Exception as e:
            print(f"Error loading memory: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dict[str, Any]: Memory statistics
        """
        # Update current stats
        self.stats["total_entries"] = self.valid_mask.sum().item()
        if self.stats["total_retrievals"] > 0:
            self.stats["hit_rate"] = self.stats.get("total_hits", 0) / self.stats["total_retrievals"]
        
        # Calculate memory usage (percentage of capacity used)
        self.stats["memory_usage"] = self.stats["total_entries"] / self.memory_size
        
        return self.stats
    
    def process_memory(self, 
                      x: torch.Tensor, 
                      model_outputs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process input through memory.
        
        Args:
            x (torch.Tensor): Input tensor
            model_outputs (Dict[str, Any], optional): Outputs from the main model
            
        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Modified tensor and memory metadata
        """
        # Basic implementation - subclasses should override
        batch_size, seq_len, hidden_dim = x.shape
        
        # Generate query keys from the last token in each sequence
        last_hidden = x[:, -1, :]
        query_keys = self.key_projection(last_hidden)
        query_keys = self.key_norm(query_keys)
        
        # Retrieve from memory
        retrieved_values, similarities, indices = self.retrieve_from_memory(query_keys)
        
        # Update access statistics
        self.update_access_stats(indices)
        
        # Increment retrieval counter
        self.stats["total_retrievals"] += batch_size
        
        # Create memory metadata
        metadata = {
            "retrieved_values": retrieved_values,
            "similarities": similarities,
            "indices": indices,
            "memory_stats": self.get_memory_stats()
        }
        
        # The simplest way to integrate memory is just to add the retrieved value
        # to the last hidden state - more sophisticated methods in subclasses
        if retrieved_values is not None and retrieved_values.size(0) > 0:
            # Reshape retrieved values if needed
            if retrieved_values.dim() == 3:  # [batch, top_k, dim]
                # Use only the top match for now
                retrieved_values = retrieved_values[:, 0, :]
            
            # Add retrieved value to the last hidden state
            x[:, -1, :] = x[:, -1, :] + retrieved_values
        
        return x, metadata
    
    def forward(self, 
               x: torch.Tensor,
               model_outputs: Optional[Dict[str, Any]] = None, 
               extension_outputs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass of the memory extension.
        
        Args:
            x (torch.Tensor): Input tensor
            model_outputs (Dict[str, Any], optional): Outputs from the main model
            extension_outputs (Dict[str, Any], optional): Outputs from other extensions
            
        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Modified tensor and extension metadata
        """
        # Process through memory
        processed_x, metadata = self.process_memory(x, model_outputs)
        
        # Check if we need to update memory
        # (e.g., if this is a training sequence, we might want to add it to memory)
        if model_outputs is not None and model_outputs.get("is_training", False):
            # Generate keys and values from the sequence
            last_hidden = x[:, -1, :]
            keys = self.key_projection(last_hidden)
            keys = self.key_norm(keys)
            values = self.value_projection(last_hidden)
            
            # Add to memory
            indices = self.add_to_memory(keys, values)
            metadata["added_indices"] = indices
            
            # Update memory persistence if needed
            self.update_memory_if_needed()
        
        return processed_x, metadata