"""
Base Memory Core Module.

This module defines the base class for memory extensions, providing
core functionality for memory operations.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import abc
import torch
import torch.nn as nn

from ..base_extension import BaseExtension
from .memory_config import MemoryConfig
from .memory_storage import MemoryStorage
from .memory_persistence import MemoryPersistence
from .memory_utils import update_memory_stats


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
        
        # Create memory configuration
        self.memory_config = MemoryConfig.from_dict(config)
        
        # Extract configuration parameters for convenience
        self.memory_size = self.memory_config.memory_size
        self.key_dim = self.memory_config.memory_key_dim
        self.value_dim = self.memory_config.memory_value_dim
        
        # Create memory storage
        self.storage = MemoryStorage(
            memory_size=self.memory_size,
            key_dim=self.key_dim,
            value_dim=self.value_dim
        )
        
        # Create persistence manager
        self.persistence = MemoryPersistence(
            enabled=self.memory_config.persistence_enabled,
            path=self.memory_config.persistence_path,
            interval=self.memory_config.persistence_interval
        )
        
        # Register memory components as buffers
        self._register_memory_buffers()
        
        # Memory statistics
        self.stats = {
            "total_entries": 0,
            "total_retrievals": 0,
            "total_updates": 0,
            "hit_rate": 0.0,
            "memory_usage": 0.0,
        }
    
    def _register_memory_buffers(self):
        """Register memory tensors as module buffers."""
        # Register memory storage tensors as buffers
        self.register_buffer("memory_keys", self.storage.memory_keys)
        self.register_buffer("memory_values", self.storage.memory_values)
        self.register_buffer("valid_mask", self.storage.valid_mask)
        
        # Create convenience references to memory layers
        self.key_norm = self.storage.key_norm
        self.key_projection = self.storage.key_projection
        self.value_projection = self.storage.value_projection
    
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
        # Use storage to clear memory
        self.storage.clear()
        
        # Reset statistics
        self.stats["total_entries"] = 0
        
        print(f"Memory cleared for extension {self.name}")
    
    def update_access_stats(self, indices: torch.Tensor) -> None:
        """
        Update access statistics for memory entries.
        
        Args:
            indices (torch.Tensor): Indices of accessed entries
        """
        # Convert tensor indices to list
        indices_list = [idx.item() for idx in indices.cpu()]
        
        # Use tracker to update access stats
        self.storage.tracker.update_access_stats(indices_list)
    
    def get_importance_score(self, idx: int) -> float:
        """
        Get importance score for a memory entry.
        
        Args:
            idx (int): Index of the entry
            
        Returns:
            float: Importance score
        """
        return self.storage.tracker.get_importance_score(idx)
    
    def find_replacement_indices(self, num_needed: int) -> torch.Tensor:
        """
        Find indices to replace in memory when it's full.
        
        Args:
            num_needed (int): Number of slots needed
            
        Returns:
            torch.Tensor: Indices to replace
        """
        return self.storage.find_replacement_indices(num_needed)
    
    def update_memory_if_needed(self) -> None:
        """Check if memory should be persisted and do so if necessary."""
        if self.persistence.should_persist():
            self.persist_memory()
    
    def persist_memory(self) -> None:
        """Save memory to disk."""
        if not self.persistence.enabled:
            return
        
        # Create a serializable memory state
        memory_state = {
            **self.storage.get_tensor_state(),
            **self.storage.get_tracker_state(),
            "stats": self.stats
        }
        
        # Save to disk
        self.persistence.save_memory(memory_state)
    
    def load_memory(self) -> bool:
        """
        Load memory from disk.
        
        Returns:
            bool: True if loading was successful
        """
        if not self.persistence.enabled:
            return False
        
        # Load from disk
        memory_state = self.persistence.load_memory()
        if memory_state is None:
            return False
        
        # Update tensors
        self.storage.set_tensor_state(memory_state)
        
        # Update tracker
        self.storage.set_tracker_state(memory_state)
        
        # Update statistics
        self.stats = memory_state["stats"]
        
        return True
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dict[str, Any]: Memory statistics
        """
        # Get the number of valid entries
        total_entries = self.storage.get_valid_count()
        
        # Update statistics
        return update_memory_stats(self.stats, total_entries, self.memory_size)
    
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
        self.update_access_stats(indices[:, 0])  # Update at least top-1 matches
        
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