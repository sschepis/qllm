"""
Memory Extension Configuration Module.

This module defines configuration parameters for memory extensions.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class MemoryConfig:
    """Configuration parameters for memory extensions."""
    
    # Memory size and dimensions
    memory_size: int = 10000
    memory_key_dim: int = 128
    memory_value_dim: int = 768
    
    # Persistence settings
    persistence_enabled: bool = False
    persistence_path: str = "memory/knowledge_store.pt"
    persistence_interval: int = 1000
    
    # Advanced configuration options can be added via the extra_config dict
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MemoryConfig':
        """
        Create a configuration object from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            MemoryConfig object with parameters from the dictionary
        """
        # Extract known parameters
        memory_size = config_dict.get("memory_size", cls.memory_size)
        memory_key_dim = config_dict.get("memory_key_dim", cls.memory_key_dim)
        memory_value_dim = config_dict.get("memory_value_dim", cls.memory_value_dim)
        persistence_enabled = config_dict.get("persistence_enabled", cls.persistence_enabled)
        persistence_path = config_dict.get("persistence_path", cls.persistence_path)
        persistence_interval = config_dict.get("persistence_interval", cls.persistence_interval)
        
        # Store any extra configuration parameters
        extra_config = {k: v for k, v in config_dict.items() if k not in {
            "memory_size", "memory_key_dim", "memory_value_dim",
            "persistence_enabled", "persistence_path", "persistence_interval"
        }}
        
        return cls(
            memory_size=memory_size,
            memory_key_dim=memory_key_dim,
            memory_value_dim=memory_value_dim,
            persistence_enabled=persistence_enabled,
            persistence_path=persistence_path,
            persistence_interval=persistence_interval,
            extra_config=extra_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        config_dict = {
            "memory_size": self.memory_size,
            "memory_key_dim": self.memory_key_dim,
            "memory_value_dim": self.memory_value_dim,
            "persistence_enabled": self.persistence_enabled,
            "persistence_path": self.persistence_path,
            "persistence_interval": self.persistence_interval,
        }
        
        # Add any extra configuration parameters
        config_dict.update(self.extra_config)
        
        return config_dict


@dataclass
class GraphMemoryConfig(MemoryConfig):
    """Configuration parameters specific to graph-based memory extensions."""
    
    # Graph structure configuration
    use_graph_structure: bool = True
    max_relations: int = 100
    relation_embedding_dim: int = 64
    
    # Entity configuration
    use_entity_types: bool = True
    entity_type_embedding_dim: int = 32
    
    # Retrieval settings
    num_neighbors: int = 10
    use_importance_sampling: bool = True
    temperature: float = 0.1
    max_path_length: int = 5
    similarity_threshold: float = 0.7
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GraphMemoryConfig':
        """
        Create a graph memory configuration object from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            GraphMemoryConfig object with parameters from the dictionary
        """
        # First create base memory config
        base_config = MemoryConfig.from_dict(config_dict)
        
        # Extract graph-specific parameters
        use_graph_structure = config_dict.get("use_graph_structure", cls.use_graph_structure)
        max_relations = config_dict.get("max_relations", cls.max_relations)
        relation_embedding_dim = config_dict.get("relation_embedding_dim", cls.relation_embedding_dim)
        use_entity_types = config_dict.get("use_entity_types", cls.use_entity_types)
        entity_type_embedding_dim = config_dict.get("entity_type_embedding_dim", cls.entity_type_embedding_dim)
        num_neighbors = config_dict.get("num_neighbors", cls.num_neighbors)
        use_importance_sampling = config_dict.get("use_importance_sampling", cls.use_importance_sampling)
        temperature = config_dict.get("temperature", cls.temperature)
        max_path_length = config_dict.get("max_path_length", cls.max_path_length)
        similarity_threshold = config_dict.get("similarity_threshold", cls.similarity_threshold)
        
        # Store any extra configuration parameters
        extra_config = base_config.extra_config
        
        return cls(
            memory_size=base_config.memory_size,
            memory_key_dim=base_config.memory_key_dim,
            memory_value_dim=base_config.memory_value_dim,
            persistence_enabled=base_config.persistence_enabled,
            persistence_path=base_config.persistence_path,
            persistence_interval=base_config.persistence_interval,
            extra_config=extra_config,
            use_graph_structure=use_graph_structure,
            max_relations=max_relations,
            relation_embedding_dim=relation_embedding_dim,
            use_entity_types=use_entity_types,
            entity_type_embedding_dim=entity_type_embedding_dim,
            num_neighbors=num_neighbors,
            use_importance_sampling=use_importance_sampling,
            temperature=temperature,
            max_path_length=max_path_length,
            similarity_threshold=similarity_threshold,
        )