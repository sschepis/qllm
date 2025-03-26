"""
Graph Query Module.

This module provides specialized query functionality for the knowledge graph
memory extension, including composite and relation-based queries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union

from .graph_storage import GraphStorage
from .graph_traversal import GraphTraversal


class GraphQuery:
    """
    Implements specialized query operations for graph-based memory.
    
    This class provides advanced query capabilities beyond simple key-based
    retrieval, including relation-aware queries and composite queries.
    """
    
    def __init__(self, graph_storage: GraphStorage):
        """
        Initialize graph query.
        
        Args:
            graph_storage: Graph storage instance to query
        """
        self.storage = graph_storage
        self.device = graph_storage.memory_keys.device
        
        # Create traversal for path-based queries
        self.traversal = GraphTraversal(graph_storage)
        
        # Initialize specialized query networks
        self._initialize_query_networks()
    
    def _initialize_query_networks(self):
        """Initialize specialized query networks."""
        # Create specialized query encoders for different types of queries
        self.query_encoders = nn.ModuleDict({
            "entity": nn.Linear(self.storage.key_dim, self.storage.key_dim),
            "relation": nn.Linear(self.storage.key_dim, self.storage.relation_embedding_dim),
            "path": nn.Linear(self.storage.key_dim, self.storage.relation_embedding_dim),
            "composite": nn.Sequential(
                nn.Linear(self.storage.key_dim, self.storage.key_dim),
                nn.LayerNorm(self.storage.key_dim),
                nn.ReLU(),
                nn.Linear(self.storage.key_dim, self.storage.key_dim + self.storage.relation_embedding_dim)
            )
        })
    
    def entity_query(
        self,
        query_key: torch.Tensor,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a basic entity query to find similar entities.
        
        Args:
            query_key: Query key [batch_size, key_dim]
            top_k: Number of top matches to retrieve
            
        Returns:
            Tuple of (values, similarity_scores, indices)
        """
        batch_size = query_key.size(0)
        
        # Encode query for entity search
        entity_query = self.query_encoders["entity"](query_key)
        
        # Normalize query
        entity_query = F.normalize(entity_query, dim=1)
        
        # Use base retrieval
        return self.storage.retrieve_entries(entity_query, top_k)
    
    def relation_query(
        self,
        query_key: torch.Tensor,
        relation_type: Optional[int] = None,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a relation-aware query to find entities related in a specific way.
        
        Args:
            query_key: Query key [batch_size, key_dim]
            relation_type: Optional relation type to focus on
            top_k: Number of top matches to retrieve
            
        Returns:
            Tuple of (values, similarity_scores, indices)
        """
        batch_size = query_key.size(0)
        
        # First, do a standard entity query to find base entities
        _, _, base_indices = self.entity_query(query_key, top_k=top_k)
        
        # Encode relation query
        relation_embedding = self.query_encoders["relation"](query_key)
        
        # For each base entity, find related entities through relations
        related_values_list = []
        related_scores_list = []
        related_indices_list = []
        
        for b in range(batch_size):
            base_entities = base_indices[b]
            
            # Collect all related entities for this batch item
            batch_related_values = []
            batch_related_scores = []
            batch_related_indices = []
            
            for entity_idx in base_entities:
                # Get neighbors with optional relation type filter
                if relation_type is not None:
                    neighbors = self.storage.get_entity_neighbors(entity_idx.item(), relation_type)
                else:
                    neighbors = self.storage.get_entity_neighbors(entity_idx.item())
                
                for neighbor_idx, rel_type in neighbors:
                    # Get relation embedding
                    rel_embedding = self.storage.relation_embeddings(torch.tensor([rel_type], device=self.device))
                    
                    # Calculate relevance score using relation embedding
                    rel_score = F.cosine_similarity(relation_embedding[b:b+1], rel_embedding, dim=1)
                    
                    # Only include if similarity is high enough
                    if rel_score.item() > 0.5:
                        batch_related_indices.append(neighbor_idx)
                        batch_related_scores.append(rel_score.item())
                        batch_related_values.append(self.storage.memory_values[neighbor_idx])
            
            # Convert lists to tensors
            if batch_related_indices:
                related_indices_tensor = torch.tensor(batch_related_indices, device=self.device)
                related_scores_tensor = torch.tensor(batch_related_scores, device=self.device)
                related_values_tensor = torch.stack(batch_related_values)
                
                # Take top-k if we have more than k
                if len(batch_related_indices) > top_k:
                    top_k_scores, top_k_indices = torch.topk(related_scores_tensor, top_k)
                    related_indices_tensor = related_indices_tensor[top_k_indices]
                    related_scores_tensor = top_k_scores
                    related_values_tensor = related_values_tensor[top_k_indices]
            else:
                # Fallback if no related entities found
                related_indices_tensor = torch.zeros(0, dtype=torch.long, device=self.device)
                related_scores_tensor = torch.zeros(0, device=self.device)
                related_values_tensor = torch.zeros(0, self.storage.value_dim, device=self.device)
            
            related_indices_list.append(related_indices_tensor)
            related_scores_list.append(related_scores_tensor)
            related_values_list.append(related_values_tensor)
        
        # Pad tensors to same size
        max_size = max(tensor.size(0) for tensor in related_indices_list)
        padded_values = []
        padded_scores = []
        padded_indices = []
        
        for values, scores, indices in zip(related_values_list, related_scores_list, related_indices_list):
            if values.size(0) == 0:
                # Handle empty tensor case
                padded_values.append(torch.zeros(max_size, self.storage.value_dim, device=self.device))
                padded_scores.append(torch.zeros(max_size, device=self.device))
                padded_indices.append(torch.zeros(max_size, dtype=torch.long, device=self.device))
            else:
                # Pad to max size
                pad_size = max_size - values.size(0)
                if pad_size > 0:
                    padded_values.append(F.pad(values, (0, 0, 0, pad_size)))
                    padded_scores.append(F.pad(scores, (0, pad_size), value=0))
                    padded_indices.append(F.pad(indices, (0, pad_size), value=0))
                else:
                    padded_values.append(values)
                    padded_scores.append(scores)
                    padded_indices.append(indices)
        
        # Stack into batch tensors
        if padded_values:
            stacked_values = torch.stack(padded_values)
            stacked_scores = torch.stack(padded_scores)
            stacked_indices = torch.stack(padded_indices)
            return stacked_values, stacked_scores, stacked_indices
        else:
            # Empty result fallback
            return (
                torch.zeros(batch_size, 0, self.storage.value_dim, device=self.device),
                torch.zeros(batch_size, 0, device=self.device),
                torch.zeros(batch_size, 0, dtype=torch.long, device=self.device)
            )
    
    def path_query(
        self,
        query_key: torch.Tensor,
        max_hops: int = 2,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform a path-based query to retrieve knowledge via graph traversal.
        
        Args:
            query_key: Query key [batch_size, key_dim]
            max_hops: Maximum number of hops in traversal
            top_k: Number of top matches to start with
            
        Returns:
            Tuple of (retrieved_values, traversal_metadata)
        """
        # First, do a standard entity query to find starting entities
        _, _, start_indices = self.entity_query(query_key, top_k=top_k)
        
        # Use traversal with path-based attention
        return self.traversal.path_based_attention_traversal(
            query_key, start_indices, max_hops, top_k
        )
    
    def composite_query(
        self,
        query_key: torch.Tensor,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform a composite query that combines entity and relation aspects.
        
        Args:
            query_key: Query key [batch_size, key_dim]
            top_k: Number of top matches to retrieve
            
        Returns:
            Tuple of (retrieved_values, query_metadata)
        """
        batch_size = query_key.size(0)
        
        # Encode query for composite search (entity + relation components)
        composite_features = self.query_encoders["composite"](query_key)
        
        # Split into entity and relation components
        entity_query = composite_features[:, :self.storage.key_dim]
        relation_query = composite_features[:, self.storage.key_dim:]
        
        # Normalize queries
        entity_query = F.normalize(entity_query, dim=1)
        
        # Get entity matches
        entity_values, entity_scores, entity_indices = self.entity_query(entity_query, top_k)
        
        # Get relation-based matches
        relation_values, relation_scores, relation_indices = self.relation_query(query_key, top_k=top_k)
        
        # Combine results
        # Simple approach: concatenate and take top-k
        combined_values = []
        combined_scores = []
        combined_indices = []
        combined_sources = []  # Track source (0=entity, 1=relation)
        
        for b in range(batch_size):
            # Collect all values, scores, indices for this batch item
            values_list = []
            scores_list = []
            indices_list = []
            sources_list = []
            
            # Add entity results
            if entity_values.size(1) > 0:
                values_list.append(entity_values[b])
                scores_list.append(entity_scores[b])
                indices_list.append(entity_indices[b])
                sources_list.append(torch.zeros_like(entity_indices[b]))
            
            # Add relation results
            if relation_values.size(1) > 0:
                values_list.append(relation_values[b])
                scores_list.append(relation_scores[b])
                indices_list.append(relation_indices[b])
                sources_list.append(torch.ones_like(relation_indices[b]))
            
            # Concatenate
            if values_list:
                batch_values = torch.cat(values_list, dim=0)
                batch_scores = torch.cat(scores_list, dim=0)
                batch_indices = torch.cat(indices_list, dim=0)
                batch_sources = torch.cat(sources_list, dim=0)
                
                # Take top-k overall
                if batch_indices.size(0) > top_k:
                    top_k_scores, top_indices = torch.topk(batch_scores, top_k)
                    batch_values = batch_values[top_indices]
                    batch_scores = top_k_scores
                    batch_indices = batch_indices[top_indices]
                    batch_sources = batch_sources[top_indices]
            else:
                # Fallback for empty results
                batch_values = torch.zeros(0, self.storage.value_dim, device=self.device)
                batch_scores = torch.zeros(0, device=self.device)
                batch_indices = torch.zeros(0, dtype=torch.long, device=self.device)
                batch_sources = torch.zeros(0, dtype=torch.long, device=self.device)
            
            combined_values.append(batch_values)
            combined_scores.append(batch_scores)
            combined_indices.append(batch_indices)
            combined_sources.append(batch_sources)
        
        # Pad tensors to same size
        max_size = max(tensor.size(0) for tensor in combined_indices)
        padded_values = []
        padded_scores = []
        padded_indices = []
        padded_sources = []
        
        for values, scores, indices, sources in zip(combined_values, combined_scores, combined_indices, combined_sources):
            if values.size(0) == 0:
                # Handle empty tensor case
                padded_values.append(torch.zeros(max_size, self.storage.value_dim, device=self.device))
                padded_scores.append(torch.zeros(max_size, device=self.device))
                padded_indices.append(torch.zeros(max_size, dtype=torch.long, device=self.device))
                padded_sources.append(torch.zeros(max_size, dtype=torch.long, device=self.device))
            else:
                # Pad to max size
                pad_size = max_size - values.size(0)
                if pad_size > 0:
                    padded_values.append(F.pad(values, (0, 0, 0, pad_size)))
                    padded_scores.append(F.pad(scores, (0, pad_size), value=0))
                    padded_indices.append(F.pad(indices, (0, pad_size), value=0))
                    padded_sources.append(F.pad(sources, (0, pad_size), value=0))
                else:
                    padded_values.append(values)
                    padded_scores.append(scores)
                    padded_indices.append(indices)
                    padded_sources.append(sources)
        
        # Stack into batch tensors
        if padded_values:
            result_values = torch.stack(padded_values)
            result_scores = torch.stack(padded_scores)
            result_indices = torch.stack(padded_indices)
            result_sources = torch.stack(padded_sources)
        else:
            # Empty result fallback
            result_values = torch.zeros(batch_size, 0, self.storage.value_dim, device=self.device)
            result_scores = torch.zeros(batch_size, 0, device=self.device)
            result_indices = torch.zeros(batch_size, 0, dtype=torch.long, device=self.device)
            result_sources = torch.zeros(batch_size, 0, dtype=torch.long, device=self.device)
        
        # Create metadata
        metadata = {
            "scores": result_scores,
            "indices": result_indices,
            "sources": result_sources,  # 0=entity, 1=relation
            "query_type": "composite"
        }
        
        return result_values, metadata
    
    def to(self, device: torch.device) -> 'GraphQuery':
        """
        Move query components to specified device.
        
        Args:
            device: Device to move to
            
        Returns:
            Self for chaining
        """
        self.device = device
        self.traversal = self.traversal.to(device)
        
        for encoder in self.query_encoders.values():
            encoder.to(device)
        
        return self