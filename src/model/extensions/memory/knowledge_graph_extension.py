"""
Knowledge Graph Extension Module.

This module defines the extension for managing knowledge in a graph structure
for the Semantic Resonance Language Model.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

from .base_memory_core import BaseMemoryExtension
from .memory_config import GraphMemoryConfig
from .graph_storage import GraphStorage
from .graph_traversal import GraphTraversal
from .graph_query import GraphQuery
from .memory_persistence import GraphMemoryPersistence
from .relation_metadata import EntityMetadata, RelationMetadata, Relation
from .entity_types import EntityType
from .relation_types import RelationType

logger = logging.getLogger(__name__)


class KnowledgeGraphExtension(BaseMemoryExtension):
    """
    Extension for managing knowledge in a graph structure.
    
    This extension extends the base memory with graph relationships,
    allowing for more structured knowledge representation and retrieval.
    It can store not just key-value pairs but also relationships between
    different entities in the knowledge graph.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the knowledge graph extension.
        
        Args:
            name: Unique name for this extension instance
            config: Configuration dictionary for the extension
        """
        super().__init__(name, config)
        
        # Create graph-specific configuration
        self.graph_config = GraphMemoryConfig.from_dict(config)
        
        # Replace standard storage with graph storage
        self.storage = GraphStorage(
            memory_size=self.memory_size,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            max_relations=self.graph_config.max_relations,
            relation_embedding_dim=self.graph_config.relation_embedding_dim,
            entity_type_embedding_dim=self.graph_config.entity_type_embedding_dim
        )
        
        # Replace standard persistence with graph persistence
        self.persistence = GraphMemoryPersistence(
            enabled=self.graph_config.persistence_enabled,
            path=self.graph_config.persistence_path,
            interval=self.graph_config.persistence_interval
        )
        
        # Create graph traversal and query components
        self.traversal = GraphTraversal(self.storage)
        self.query = GraphQuery(self.storage)
        
        # Re-register buffers with new storage
        self._register_memory_buffers()
    
    def initialize(self, model: nn.Module) -> None:
        """
        Initialize the extension with the main model.
        
        This method is called once when the extension is attached to the model.
        
        Args:
            model (nn.Module): The main model instance
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        # Move the storage to the model's device
        self.storage.to(self.device)
        
        # Mark as initialized
        self.initialized = True
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dict[str, Any]: Dictionary of graph statistics
        """
        total_relations = 0
        if hasattr(self.storage, 'relations'):
            total_relations = len(self.storage.relations) if hasattr(self.storage.relations, '__len__') else 0
        # Count current size by summing the valid entries
        current_size = self.storage.valid_mask.sum().item() if hasattr(self.storage, 'valid_mask') else 0
        
        stats = {
            "total_entities": current_size,
            "total_relations": total_relations,
            "entity_types": {},
            "relation_types": {}
        }
        
        # Count entity types if available
        if hasattr(self.storage, 'entity_types') and self.storage.entity_types is not None:
            entity_type_counts = {}
            for i in range(int(current_size)):
                if i < len(self.storage.entity_types):
                    entity_type = int(self.storage.entity_types[i].item())
                    entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
            
            stats["entity_types"] = entity_type_counts
            
        return stats
        
    def add_to_memory(self,
                     keys: torch.Tensor,
                     values: torch.Tensor,
                     metadata: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Add new entities to the knowledge graph with rich metadata.
        
        Args:
            keys: Keys for the new entities
            values: Values (embeddings) for the new entities
            metadata: Additional metadata for the entities
                Can include:
                - 'entity_types': Tensor of entity type indices or EntityType enums
                - 'entity_names': List of entity names
                - 'entity_descriptions': List of entity descriptions
                - 'entity_confidence': Tensor or List of confidence scores
                - 'entity_sources': List of sources
                - 'entity_importance': Tensor or List of importance scores
                - 'entity_attributes': List of attribute dictionaries
                - 'relations': List of relation specifications (tuples or dicts)
            
        Returns:
            Indices of the added entities
        """
        batch_size = keys.size(0)
        
        # Find slots for new entities
        indices = self.storage.find_replacement_indices(batch_size)
        
        # Process metadata
        if metadata is None:
            metadata = {}
            
        # Extract metadata fields with proper fallbacks
        entity_types = metadata.get('entity_types', [EntityType.GENERIC] * batch_size)
        entity_names = metadata.get('entity_names', [None] * batch_size)
        entity_descriptions = metadata.get('entity_descriptions', [None] * batch_size)
        entity_confidence = metadata.get('entity_confidence', [1.0] * batch_size)
        entity_sources = metadata.get('entity_sources', [None] * batch_size)
        entity_importance = metadata.get('entity_importance', [0.5] * batch_size)
        entity_attributes = metadata.get('entity_attributes', [{}] * batch_size)
        
        # Add entities to memory
        for i, idx in enumerate(indices):
            if i < batch_size:
                # Create entity metadata
                entity_type = entity_types[i] if i < len(entity_types) else EntityType.GENERIC
                
                # Handle different types of confidence value
                confidence = entity_confidence[i] if i < len(entity_confidence) else 1.0
                if isinstance(confidence, torch.Tensor):
                    confidence = confidence.item()
                
                # Handle different types of importance value
                importance = entity_importance[i] if i < len(entity_importance) else 0.5
                if isinstance(importance, torch.Tensor):
                    importance = importance.item()
                
                # Create entity metadata
                entity_metadata = EntityMetadata(
                    entity_type=entity_type,
                    name=entity_names[i] if i < len(entity_names) else None,
                    description=entity_descriptions[i] if i < len(entity_descriptions) else None,
                    confidence=confidence,
                    source=entity_sources[i] if i < len(entity_sources) else None,
                    importance=importance,
                    attributes=entity_attributes[i].copy() if i < len(entity_attributes) else {}
                )
                
                # Add entity to graph storage
                self.storage.add_entity(
                    key=keys[i],
                    value=values[i],
                    metadata=entity_metadata,
                    idx=idx.item()
                )
        
        # Add relations if provided
        if 'relations' in metadata:
            for relation in metadata['relations']:
                if isinstance(relation, tuple) and len(relation) == 3:
                    # Simple (subject, relation, object) triple
                    subject_idx, relation_type, object_idx = relation
                    self.storage.add_relation(subject_idx, relation_type, object_idx)
                elif isinstance(relation, dict):
                    # Rich relation specification with metadata
                    subject_idx = relation.get('subject_idx')
                    relation_type = relation.get('relation_type')
                    object_idx = relation.get('object_idx')
                    relation_metadata = relation.get('metadata', {})
                    
                    if subject_idx is not None and relation_type is not None and object_idx is not None:
                        self.storage.add_relation(subject_idx, relation_type, object_idx, relation_metadata)
        
        # Update memory persistence if needed
        self.update_memory_if_needed()
        
        # Update statistics
        self.stats["total_entries"] = self.storage.get_valid_count()
        self.stats["total_updates"] += 1
        
        return indices
    
    def retrieve_from_memory(self, 
                           query_keys: torch.Tensor, 
                           top_k: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve entities from the knowledge graph based on query keys.
        
        Args:
            query_keys: Query keys to search for
            top_k: Number of top matches to retrieve
            
        Returns:
            Tuple of (retrieved_values, similarity_scores, indices)
        """
        # Use the query component for retrieval
        values, scores, indices = self.query.entity_query(query_keys, top_k)
        
        # Update statistics
        self.stats["total_hits"] = self.stats.get("total_hits", 0) + query_keys.size(0)
        
        return values, scores, indices
    
    def retrieve_with_path_attention(self,
                                   query_keys: torch.Tensor,
                                   max_hops: int = 2,
                                   top_k: int = 5) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Retrieve entities with path-based attention for more focused knowledge traversal.
        
        Args:
            query_keys: Query keys to search for
            max_hops: Maximum number of hops in traversal
            top_k: Number of top matches at each hop
            
        Returns:
            Tuple of (retrieved_values, traversal_metadata)
        """
        # Use path query for advanced traversal
        return self.query.path_query(query_keys, max_hops, top_k)
    
    def update_memory(self, 
                     indices: torch.Tensor, 
                     values: torch.Tensor, 
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update existing entities in the knowledge graph.
        
        Args:
            indices: Indices of entities to update
            values: New values for the entities
            metadata: Additional metadata to update
                Can include:
                - 'entity_types': Tensor of entity type indices
                - 'relations': List of (subject_idx, relation_type, object_idx) triples to add
                - 'remove_relations': List of (subject_idx, relation_type, object_idx) triples to remove
            
        Returns:
            True if update was successful
        """
        batch_size = indices.size(0)
        
        # Update entity values
        success = self.storage.update_entries(indices, values)
        if not success:
            return False
        
        # Update entity types if provided
        if metadata and 'entity_types' in metadata:
            for i, idx in enumerate(indices):
                if i < batch_size and i < len(metadata['entity_types']):
                    entity_type = metadata['entity_types'][i]
                    type_id = entity_type.value if hasattr(entity_type, 'value') else entity_type
                    
                    # Update entity type
                    if idx.item() in self.storage.entity_types:
                        old_type = self.storage.entity_types[idx.item()]
                        
                        # Remove from old type index
                        if old_type in self.storage.entities_by_type and idx.item() in self.storage.entities_by_type[old_type]:
                            self.storage.entities_by_type[old_type].remove(idx.item())
                        
                        # Update type
                        self.storage.entity_types[idx.item()] = type_id
                        
                        # Add to new type index
                        self.storage.entities_by_type[type_id].add(idx.item())
        
        # Add relations if provided
        if metadata and 'relations' in metadata:
            for relation in metadata['relations']:
                if len(relation) == 3:
                    subject_idx, relation_type, object_idx = relation
                    self.storage.add_relation(subject_idx, relation_type, object_idx)
        
        # Remove relations if specified
        if metadata and 'remove_relations' in metadata:
            for relation in metadata['remove_relations']:
                if len(relation) == 3:
                    subject_idx, relation_type, object_idx = relation
                    self.storage.remove_relation(
                        subject_idx=subject_idx,
                        relation_type=relation_type,
                        object_idx=object_idx
                    )
        
        # Update statistics
        self.stats["total_updates"] += 1
        
        # Update memory persistence if needed
        self.update_memory_if_needed()
        
        return True
    
    def persist_memory(self) -> None:
        """Save memory and graph structure to disk."""
        if not self.persistence.enabled:
            return
        
        # Create a serializable memory state
        memory_state = {
            **self.storage.get_tensor_state(),
            **self.storage.get_tracker_state(),
            "stats": self.stats
        }
        
        # Save memory state
        self.persistence.save_memory(memory_state)
        
        # Create graph state for serialization
        graph_state = {
            "adjacency_list": self.storage.adjacency_list,
            "inverse_adjacency_list": self.storage.inverse_adjacency_list,
            "entity_types": self.storage.entity_types,
            "graph_stats": self.storage.graph_stats
        }
        
        # Save graph state
        if isinstance(self.persistence, GraphMemoryPersistence):
            self.persistence.save_graph_structure(graph_state)
    
    def load_memory(self) -> bool:
        """
        Load memory and graph structure from disk.
        
        Returns:
            True if loading was successful
        """
        if not self.persistence.enabled:
            return False
        
        # Load memory state
        memory_state = self.persistence.load_memory()
        if memory_state is None:
            return False
        
        # Update tensors
        self.storage.set_tensor_state(memory_state)
        
        # Update tracker
        self.storage.set_tracker_state(memory_state)
        
        # Update statistics
        self.stats = memory_state["stats"]
        
        # Load graph structure
        if isinstance(self.persistence, GraphMemoryPersistence):
            graph_state = self.persistence.load_graph_structure()
            
            if graph_state is not None:
                # Load graph structures
                self.storage.adjacency_list = graph_state["adjacency_list"]
                self.storage.inverse_adjacency_list = graph_state["inverse_adjacency_list"]
                self.storage.entity_types = graph_state["entity_types"]
                self.storage.graph_stats = graph_state["graph_stats"]
        
        return True
    
    def process_memory(self, 
                       x: torch.Tensor, 
                       model_outputs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process input through knowledge graph memory.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            model_outputs: Outputs from the main model
            
        Returns:
            Tuple of (modified_tensor, memory_metadata)
        """
        batch_size, seq_len, hidden_dim = x.shape
        device = x.device
        
        # Generate query keys from the last token in each sequence
        last_hidden = x[:, -1, :]
        query_keys = self.key_projection(last_hidden)
        query_keys = self.key_norm(query_keys)
        
        # Retrieve from memory
        retrieved_values, similarities, indices = self.retrieve_from_memory(
            query_keys, top_k=self.graph_config.num_neighbors
        )
        
        # Update access statistics
        self.update_access_stats(indices[:, 0])  # Update at least top-1 matches
        
        # Enhanced knowledge through traversal
        if self.graph_config.use_graph_structure:
            # Use path-based attention for more focused knowledge retrieval
            traversal_values, traversal_metadata = self.retrieve_with_path_attention(
                query_keys, max_hops=2, top_k=3
            )
            
            # Use enhanced values if available
            enhanced_values = traversal_values
            
            # Create memory metadata
            metadata = {
                "retrieved_values": retrieved_values,
                "similarities": similarities,
                "indices": indices,
                "enhanced_values": enhanced_values,
                "traversal_metadata": traversal_metadata,
                "memory_stats": self.get_memory_stats(),
                "graph_stats": self.storage.graph_stats
            }
            
            # Apply the knowledge
            if enhanced_values is not None and enhanced_values.size(0) > 0:
                # Add retrieved values to the last hidden state
                x[:, -1, :] = x[:, -1, :] + enhanced_values
        else:
            # Use basic retrieval without graph traversal
            metadata = {
                "retrieved_values": retrieved_values,
                "similarities": similarities,
                "indices": indices,
                "memory_stats": self.get_memory_stats(),
                "graph_stats": self.storage.graph_stats
            }
            
            # Apply the knowledge (simple approach)
            if retrieved_values is not None and retrieved_values.size(0) > 0:
                # Use only the top match for each batch item
                top_values = retrieved_values[:, 0, :]
                x[:, -1, :] = x[:, -1, :] + top_values
        
        return x, metadata
    
    def forward(self, 
                x: torch.Tensor,
                model_outputs: Optional[Dict[str, Any]] = None, 
                extension_outputs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass of the knowledge graph extension.
        
        Args:
            x: Input tensor
            model_outputs: Outputs from the main model
            extension_outputs: Outputs from other extensions
            
        Returns:
            Tuple of (modified_tensor, extension_metadata)
        """
        # Process through memory
        processed_x, metadata = self.process_memory(x, model_outputs)
        
        # Check if we need to update memory
        if model_outputs is not None:
            # Several scenarios for updating memory:
            
            # 1. Explicit update signal
            if model_outputs.get("update_memory", False):
                last_hidden = x[:, -1, :]
                
                # Generate keys and values
                keys = self.key_projection(last_hidden)
                keys = self.key_norm(keys)
                values = self.value_projection(last_hidden)
                
                # Add relation metadata if present
                metadata_dict = None
                if "memory_metadata" in model_outputs:
                    metadata_dict = model_outputs["memory_metadata"]
                
                # Add to memory
                indices = self.add_to_memory(keys, values, metadata_dict)
                metadata["added_indices"] = indices
            
            # 2. Training mode update
            elif model_outputs.get("is_training", False):
                # Generate keys and values from the sequence
                last_hidden = x[:, -1, :]
                keys = self.key_projection(last_hidden)
                keys = self.key_norm(keys)
                values = self.value_projection(last_hidden)
                
                # Add to memory
                indices = self.add_to_memory(keys, values)
                metadata["added_indices"] = indices
            
            # 3. External knowledge source (e.g., fact injection)
            if "external_knowledge" in model_outputs:
                external_data = model_outputs["external_knowledge"]
                # Process external knowledge and add to graph
                # (specific implementation depends on the format of external_knowledge)
            
            # Update memory persistence if needed
            self.update_memory_if_needed()
        
        return processed_x, metadata