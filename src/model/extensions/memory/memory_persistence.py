"""
Memory Persistence Module.

This module provides functionality for persisting memory data to storage.
"""

import os
import json
import torch
from typing import Dict, Any, Optional, List, Union, Tuple


class MemoryPersistence:
    """
    Manages persistence of memory data.
    
    This class handles saving and loading memory data to/from storage.
    """
    
    def __init__(
        self,
        enabled: bool = False,
        path: str = "memory/knowledge_store.pt",
        interval: int = 1000
    ):
        """
        Initialize memory persistence manager.
        
        Args:
            enabled: Whether persistence is enabled
            path: Path where memory will be stored
            interval: Number of updates between persistence operations
        """
        self.enabled = enabled
        self.path = path
        self.interval = interval
        self.update_counter = 0
    
    def should_persist(self) -> bool:
        """
        Check if memory should be persisted based on the update counter.
        
        Returns:
            True if memory should be persisted, False otherwise
        """
        if not self.enabled:
            return False
        
        self.update_counter += 1
        if self.update_counter >= self.interval:
            self.update_counter = 0
            return True
        
        return False
    
    def save_memory(self, memory_state: Dict[str, Any]) -> bool:
        """
        Save memory state to disk.
        
        Args:
            memory_state: Dictionary containing memory state to save
            
        Returns:
            True if save was successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            
            # Save to disk
            torch.save(memory_state, self.path)
            print(f"Memory persisted to {self.path}")
            return True
        except Exception as e:
            print(f"Error persisting memory: {e}")
            return False
    
    def load_memory(self) -> Optional[Dict[str, Any]]:
        """
        Load memory state from disk.
        
        Returns:
            Dictionary containing memory state if load was successful, None otherwise
        """
        if not self.enabled:
            return None
        
        try:
            memory_state = torch.load(self.path)
            print(f"Memory loaded from {self.path}")
            return memory_state
        except Exception as e:
            print(f"Error loading memory: {e}")
            return None
    
    def prepare_tensor_state(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare tensor state for saving by moving to CPU.
        
        Args:
            tensors: Dictionary of tensors to prepare
            
        Returns:
            Dictionary of prepared tensors
        """
        return {name: tensor.cpu() for name, tensor in tensors.items()}


class GraphMemoryPersistence(MemoryPersistence):
    """
    Manages persistence of graph memory data with additional graph structure handling.
    """
    
    def save_graph_structure(self, graph_state: Dict[str, Any]) -> bool:
        """
        Save graph structure to disk as JSON.
        
        Args:
            graph_state: Dictionary containing graph structure to save
            
        Returns:
            True if save was successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Extract path and base filename
            base_path = os.path.splitext(self.path)[0]
            graph_path = f"{base_path}_graph.json"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            
            # Convert state to JSON (with string keys)
            serializable_state = self._prepare_graph_state_for_serialization(graph_state)
            
            with open(graph_path, 'w') as f:
                json.dump(serializable_state, f)
            
            print(f"Graph structure persisted to {graph_path}")
            return True
        except Exception as e:
            print(f"Error persisting graph structure: {e}")
            return False
    
    def load_graph_structure(self) -> Optional[Dict[str, Any]]:
        """
        Load graph structure from disk.
        
        Returns:
            Dictionary containing graph structure if load was successful, None otherwise
        """
        if not self.enabled:
            return None
        
        try:
            # Extract path and base filename
            base_path = os.path.splitext(self.path)[0]
            graph_path = f"{base_path}_graph.json"
            
            with open(graph_path, 'r') as f:
                graph_state = json.load(f)
            
            # Convert from serialized format
            loaded_state = self._prepare_graph_state_after_loading(graph_state)
            
            print(f"Graph structure loaded from {graph_path}")
            return loaded_state
        except Exception as e:
            print(f"Error loading graph structure: {e}")
            return None
    
    def _prepare_graph_state_for_serialization(self, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare graph state for serialization to JSON.
        
        Args:
            graph_state: Dictionary containing graph structure to prepare
            
        Returns:
            Serializable dictionary
        """
        serializable_state = {}
        
        # Convert integer keys to strings for JSON serialization
        for key, value in graph_state.items():
            if isinstance(value, dict) and any(isinstance(k, int) for k in value.keys()):
                # Convert int keys to strings
                serializable_state[key] = {str(k): v for k, v in value.items()}
            else:
                serializable_state[key] = value
        
        return serializable_state
    
    def _prepare_graph_state_after_loading(self, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare graph state after loading from JSON.
        
        Args:
            graph_state: Dictionary containing loaded graph structure
            
        Returns:
            Dictionary with properly typed keys
        """
        loaded_state = {}
        
        # Convert string keys back to integers where appropriate
        for key, value in graph_state.items():
            if key in ["adjacency_list", "inverse_adjacency_list", "entity_types"]:
                # These dictionaries use integer keys
                loaded_state[key] = {int(k): v for k, v in value.items()}
            else:
                loaded_state[key] = value
        
        return loaded_state