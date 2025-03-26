"""
Graph Storage Module.

This module provides functionality for storing and retrieving data in a
graph structure for the knowledge graph memory extension.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Set, Union
from collections import defaultdict, Counter

from .memory_storage import MemoryStorage
from .relation_metadata import Relation, EntityMetadata, RelationMetadata
from .entity_types import EntityType
from .relation_types import RelationType


class GraphStorage(MemoryStorage):
    """
    Manages storage and retrieval of data in a graph structure.
    
    This class extends MemoryStorage to add graph-specific functionality,
    including management of relations, entity types, and graph indices.
    """
    
    def __init__(
        self,
        memory_size: int,
        key_dim: int,
        value_dim: int,
        max_relations: int = 100,
        relation_embedding_dim: int = 64,
        entity_type_embedding_dim: int = 32,
        device: Optional[torch.device] = None
    ):
        """
        Initialize graph storage.
        
        Args:
            memory_size: Maximum number of entities in memory
            key_dim: Dimension of entity keys
            value_dim: Dimension of entity values
            max_relations: Maximum number of relation types
            relation_embedding_dim: Dimension of relation embeddings
            entity_type_embedding_dim: Dimension of entity type embeddings
            device: Device to store tensors on
        """
        super().__init__(memory_size, key_dim, value_dim, device)
        
        self.max_relations = max_relations
        self.relation_embedding_dim = relation_embedding_dim
        self.entity_type_embedding_dim = entity_type_embedding_dim
        
        # Initialize graph components
        self._initialize_graph_components()
        
        # Graph statistics
        self.graph_stats = {
            "num_nodes": 0,
            "num_edges": 0,
            "average_degree": 0.0,
            "density": 0.0,
            "relation_count": 0,
            "relation_types": Counter(),
            "entity_types": Counter(),
        }
    
    def _initialize_graph_components(self):
        """Initialize components specific to graph structure."""
        # Initialize relation embeddings
        self.relation_embeddings = nn.Embedding(
            self.max_relations + 1,  # +1 for unknown relation
            self.relation_embedding_dim
        )
        
        # Initialize entity type embeddings
        max_entity_types = len(EntityType) + 10  # Add buffer for custom types
        self.entity_type_embeddings = nn.Embedding(
            max_entity_types,
            self.entity_type_embedding_dim
        )
        
        # Entity metadata storage
        self.entity_metadata = {}  # {entity_id: EntityMetadata}
        
        # Relation storage
        self.relations = {}  # {relation_id: Relation}
        self.relation_counter = 0  # Counter for generating relation IDs
        
        # Adjacency structure: for each node, store outgoing edges
        # Format: {node_idx: [(neighbor_idx, relation_idx), ...]}
        self.adjacency_list = defaultdict(list)
        
        # Inverse adjacency for incoming edges
        # Format: {node_idx: [(neighbor_idx, relation_idx), ...]}
        self.inverse_adjacency_list = defaultdict(list)
        
        # Entity type storage
        self.entity_types = {}  # {node_idx: type_idx}
        
        # Maps for entity lookup by type
        self.entities_by_type = defaultdict(set)
        
        # Maps for relation lookup
        self.relations_by_type = defaultdict(list)
        self.relations_by_subject = defaultdict(list)
        self.relations_by_object = defaultdict(list)
        
        # Maps for time-based lookups
        self.relations_by_creation_time = {}
        self.entities_by_creation_time = {}
        
        # Maps for confidence-based lookups
        self.high_confidence_entities = set()  # Entities with confidence > 0.8
        self.high_confidence_relations = set()  # Relations with confidence > 0.8
        
        # Relation projection for query -> relation
        self.relation_projection = nn.Linear(
            self.key_dim,
            self.relation_embedding_dim
        )
        
        # Projection for combining node and relation features
        self.node_relation_projection = nn.Linear(
            self.value_dim + self.relation_embedding_dim,
            self.value_dim
        )
    
    def clear(self) -> None:
        """Clear all graph data."""
        # Clear base memory
        super().clear()
        
        # Clear graph-specific data
        self.entity_metadata.clear()
        self.relations.clear()
        self.relation_counter = 0
        
        self.adjacency_list.clear()
        self.inverse_adjacency_list.clear()
        self.entity_types.clear()
        
        self.entities_by_type.clear()
        self.relations_by_type.clear()
        self.relations_by_subject.clear()
        self.relations_by_object.clear()
        
        self.relations_by_creation_time.clear()
        self.entities_by_creation_time.clear()
        
        self.high_confidence_entities.clear()
        self.high_confidence_relations.clear()
        
        # Reset graph statistics
        self.graph_stats = {
            "num_nodes": 0,
            "num_edges": 0,
            "average_degree": 0.0,
            "density": 0.0,
            "relation_count": 0,
            "relation_types": Counter(),
            "entity_types": Counter(),
        }
    
    def add_entity(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: Optional[EntityMetadata] = None,
        idx: Optional[int] = None
    ) -> int:
        """
        Add an entity to the graph.
        
        Args:
            key: Key for the entity
            value: Value (embedding) for the entity
            metadata: Metadata for the entity
            idx: Optional specific index to place the entity
            
        Returns:
            Index where entity was added
        """
        # Convert single tensors to batches for the base method
        key_batch = key.unsqueeze(0) if key.dim() == 1 else key
        value_batch = value.unsqueeze(0) if value.dim() == 1 else value
        
        # Determine index for new entity
        if idx is None:
            # Find space for a single entity
            indices = self.find_replacement_indices(1)
            idx = indices[0].item()
        else:
            # Use provided index
            indices = torch.tensor([idx], device=key.device)
        
        # Add to base memory storage
        self.add_entries(key_batch, value_batch, indices)
        
        # Process entity metadata
        if metadata is None:
            metadata = EntityMetadata()
        
        # Store entity metadata
        self.entity_metadata[idx] = metadata
        
        # Store entity type
        entity_type = metadata.entity_type
        type_id = entity_type.value if isinstance(entity_type, EntityType) else entity_type
        self.entity_types[idx] = type_id
        
        # Update indexes
        self.entities_by_type[type_id].add(idx)
        self.entities_by_creation_time[idx] = metadata.created_at
        
        # Track high confidence entities
        if metadata.confidence > 0.8:
            self.high_confidence_entities.add(idx)
        
        # Update entity type statistics
        self.graph_stats["entity_types"][type_id] = self.graph_stats["entity_types"].get(type_id, 0) + 1
        
        return idx
    
    def add_relation(
        self,
        subject_idx: int,
        relation_type: Union[RelationType, int],
        object_idx: int,
        metadata: Optional[Union[RelationMetadata, Dict[str, Any]]] = None
    ) -> int:
        """
        Add a relation between two entities in the graph.
        
        Args:
            subject_idx: Index of the subject entity
            relation_type: Type of the relation
            object_idx: Index of the object entity
            metadata: Metadata for the relation
            
        Returns:
            ID of the new relation, or -1 if failed
        """
        # Validate indices
        if not (0 <= subject_idx < self.memory_size and 0 <= object_idx < self.memory_size):
            print(f"Cannot add relation: indices {subject_idx} and {object_idx} out of range")
            return -1
        
        # Validate that both entities exist
        if not (self.valid_mask[subject_idx] and self.valid_mask[object_idx]):
            print(f"Cannot add relation: entities {subject_idx} and {object_idx} must both exist")
            return -1
        
        # Convert dictionary to RelationMetadata if needed
        if isinstance(metadata, dict):
            metadata = RelationMetadata(**metadata)
        elif metadata is None:
            metadata = RelationMetadata()
        
        # Create relation object
        relation = Relation(
            subject_id=subject_idx,
            relation_type=relation_type,
            object_id=object_idx,
            metadata=metadata
        )
        
        # Generate relation ID
        relation_id = self.relation_counter
        self.relation_counter += 1
        
        # Store relation
        self.relations[relation_id] = relation
        
        # Get relation type ID
        if isinstance(relation_type, RelationType):
            type_id = relation_type.value
        else:
            type_id = relation_type
        
        # Update indexes
        self.relations_by_type[type_id].append(relation_id)
        self.relations_by_subject[subject_idx].append(relation_id)
        self.relations_by_object[object_idx].append(relation_id)
        self.relations_by_creation_time[relation_id] = metadata.created_at
        
        if metadata.confidence > 0.8:
            self.high_confidence_relations.add(relation_id)
        
        # Update adjacency lists
        if (object_idx, type_id) not in self.adjacency_list[subject_idx]:
            self.adjacency_list[subject_idx].append((object_idx, type_id))
        
        if (subject_idx, type_id) not in self.inverse_adjacency_list[object_idx]:
            self.inverse_adjacency_list[object_idx].append((subject_idx, type_id))
        
        # Update graph statistics
        self._update_graph_stats()
        
        # Update relation type statistics
        self.graph_stats["relation_count"] = len(self.relations)
        self.graph_stats["relation_types"][type_id] = self.graph_stats["relation_types"].get(type_id, 0) + 1
        
        return relation_id
    
    def remove_relation(
        self,
        relation_id: Optional[int] = None,
        subject_idx: Optional[int] = None,
        relation_type: Optional[Union[RelationType, int]] = None,
        object_idx: Optional[int] = None
    ) -> bool:
        """
        Remove a relation from the graph.
        
        Can identify the relation either by relation_id or by the triple
        (subject_idx, relation_type, object_idx).
        
        Args:
            relation_id: ID of the relation to remove
            subject_idx: Index of the subject entity
            relation_type: Type of the relation
            object_idx: Index of the object entity
            
        Returns:
            True if removal was successful
        """
        # Find relation by ID
        if relation_id is not None:
            if relation_id not in self.relations:
                return False
            
            relation = self.relations[relation_id]
            subject_idx = relation.subject_id
            object_idx = relation.object_id
            type_id = relation.relation_type.value if isinstance(relation.relation_type, RelationType) else relation.relation_type
            
            # Remove relation
            del self.relations[relation_id]
            
            # Remove from indices
            self._remove_relation_from_indices(relation_id, subject_idx, object_idx, type_id)
            
            return True
        
        # Find relation by triple
        elif subject_idx is not None and relation_type is not None and object_idx is not None:
            # Convert relation type to ID if needed
            if isinstance(relation_type, RelationType):
                type_id = relation_type.value
            else:
                type_id = relation_type
            
            # Find relation ID
            for rel_id in self.relations_by_subject.get(subject_idx, []):
                relation = self.relations.get(rel_id)
                if relation is None:
                    continue
                
                rel_type = relation.relation_type.value if isinstance(relation.relation_type, RelationType) else relation.relation_type
                
                if relation.object_id == object_idx and rel_type == type_id:
                    # Found matching relation
                    del self.relations[rel_id]
                    
                    # Remove from indices
                    self._remove_relation_from_indices(rel_id, subject_idx, object_idx, type_id)
                    
                    return True
            
            return False
        
        else:
            # Not enough information to identify relation
            return False
    
    def _remove_relation_from_indices(
        self,
        relation_id: int,
        subject_idx: int,
        object_idx: int,
        type_id: int
    ) -> None:
        """
        Remove a relation from all indices.
        
        Args:
            relation_id: ID of the relation
            subject_idx: Index of the subject entity
            object_idx: Index of the object entity
            type_id: Type ID of the relation
        """
        # Remove from relation indices
        if type_id in self.relations_by_type and relation_id in self.relations_by_type[type_id]:
            self.relations_by_type[type_id].remove(relation_id)
        
        if subject_idx in self.relations_by_subject and relation_id in self.relations_by_subject[subject_idx]:
            self.relations_by_subject[subject_idx].remove(relation_id)
        
        if object_idx in self.relations_by_object and relation_id in self.relations_by_object[object_idx]:
            self.relations_by_object[object_idx].remove(relation_id)
        
        if relation_id in self.relations_by_creation_time:
            del self.relations_by_creation_time[relation_id]
        
        if relation_id in self.high_confidence_relations:
            self.high_confidence_relations.remove(relation_id)
        
        # Remove from adjacency lists
        if subject_idx in self.adjacency_list:
            self.adjacency_list[subject_idx] = [
                (o, r) for o, r in self.adjacency_list[subject_idx]
                if not (o == object_idx and r == type_id)
            ]
        
        if object_idx in self.inverse_adjacency_list:
            self.inverse_adjacency_list[object_idx] = [
                (s, r) for s, r in self.inverse_adjacency_list[object_idx]
                if not (s == subject_idx and r == type_id)
            ]
        
        # Update graph statistics
        self._update_graph_stats()
    
    def _update_graph_stats(self) -> None:
        """Update graph statistics."""
        self.graph_stats["num_edges"] = sum(len(edges) for edges in self.adjacency_list.values())
        self.graph_stats["num_nodes"] = len(self.adjacency_list)
        
        if self.graph_stats["num_nodes"] > 0:
            self.graph_stats["average_degree"] = self.graph_stats["num_edges"] / self.graph_stats["num_nodes"]
            max_possible_edges = self.graph_stats["num_nodes"] * (self.graph_stats["num_nodes"] - 1)
            if max_possible_edges > 0:
                self.graph_stats["density"] = self.graph_stats["num_edges"] / max_possible_edges
    
    def get_entity_neighbors(
        self,
        entity_idx: int,
        relation_type: Optional[Union[RelationType, int]] = None
    ) -> List[Tuple[int, int]]:
        """
        Get neighbors of an entity.
        
        Args:
            entity_idx: Index of the entity
            relation_type: Optional filter by relation type
            
        Returns:
            List of (neighbor_idx, relation_type) pairs
        """
        if entity_idx not in self.adjacency_list:
            return []
        
        if relation_type is None:
            return self.adjacency_list[entity_idx]
        else:
            # Convert relation type to ID if needed
            if isinstance(relation_type, RelationType):
                type_id = relation_type.value
            else:
                type_id = relation_type
            
            # Filter by relation type
            return [(n, r) for n, r in self.adjacency_list[entity_idx] if r == type_id]
    
    def get_entity_incoming(
        self,
        entity_idx: int,
        relation_type: Optional[Union[RelationType, int]] = None
    ) -> List[Tuple[int, int]]:
        """
        Get incoming connections to an entity.
        
        Args:
            entity_idx: Index of the entity
            relation_type: Optional filter by relation type
            
        Returns:
            List of (source_idx, relation_type) pairs
        """
        if entity_idx not in self.inverse_adjacency_list:
            return []
        
        if relation_type is None:
            return self.inverse_adjacency_list[entity_idx]
        else:
            # Convert relation type to ID if needed
            if isinstance(relation_type, RelationType):
                type_id = relation_type.value
            else:
                type_id = relation_type
            
            # Filter by relation type
            return [(s, r) for s, r in self.inverse_adjacency_list[entity_idx] if r == type_id]
    
    def get_entities_by_type(
        self,
        entity_type: Union[EntityType, int]
    ) -> Set[int]:
        """
        Get all entities of a specific type.
        
        Args:
            entity_type: Type of entities to retrieve
            
        Returns:
            Set of entity indices
        """
        # Convert entity type to ID if needed
        if isinstance(entity_type, EntityType):
            type_id = entity_type.value
        else:
            type_id = entity_type
        
        return self.entities_by_type.get(type_id, set())
    
    def get_relations_by_type(
        self,
        relation_type: Union[RelationType, int]
    ) -> List[int]:
        """
        Get all relations of a specific type.
        
        Args:
            relation_type: Type of relations to retrieve
            
        Returns:
            List of relation IDs
        """
        # Convert relation type to ID if needed
        if isinstance(relation_type, RelationType):
            type_id = relation_type.value
        else:
            type_id = relation_type
        
        return self.relations_by_type.get(type_id, [])
    
    def get_relations_between(
        self,
        subject_idx: int,
        object_idx: int,
        relation_type: Optional[Union[RelationType, int]] = None
    ) -> List[int]:
        """
        Get all relations between two entities.
        
        Args:
            subject_idx: Index of the subject entity
            object_idx: Index of the object entity
            relation_type: Optional filter by relation type
            
        Returns:
            List of relation IDs
        """
        # Convert relation type to ID if needed
        if relation_type is not None:
            if isinstance(relation_type, RelationType):
                type_id = relation_type.value
            else:
                type_id = relation_type
        
        # Get candidate relations from subject
        candidate_relations = self.relations_by_subject.get(subject_idx, [])
        
        # Filter by object
        result = []
        for rel_id in candidate_relations:
            relation = self.relations.get(rel_id)
            if relation is None:
                continue
            
            if relation.object_id != object_idx:
                continue
            
            # Filter by relation type if specified
            if relation_type is not None:
                rel_type = relation.relation_type.value if isinstance(relation.relation_type, RelationType) else relation.relation_type
                if rel_type != type_id:
                    continue
            
            result.append(rel_id)
        
        return result
    
    def to(self, device: torch.device) -> 'GraphStorage':
        """
        Move graph storage to specified device.
        
        Args:
            device: Device to move tensors to
            
        Returns:
            Self for chaining
        """
        # Move base storage
        super().to(device)
        
        # Move graph-specific components
        self.relation_embeddings = self.relation_embeddings.to(device)
        self.entity_type_embeddings = self.entity_type_embeddings.to(device)
        self.relation_projection = self.relation_projection.to(device)
        self.node_relation_projection = self.node_relation_projection.to(device)
        
        return self