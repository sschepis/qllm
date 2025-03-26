"""
Graph Traversal Module.

This module provides functionality for traversing the graph structure
to retrieve related entities and knowledge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Set, Union
from collections import defaultdict, deque

from .graph_storage import GraphStorage


class GraphTraversal:
    """
    Provides traversal algorithms for graph-based knowledge retrieval.
    
    This class implements various algorithms for traversing the knowledge
    graph to find relevant entities and aggregate knowledge.
    """
    
    def __init__(self, graph_storage: GraphStorage):
        """
        Initialize graph traversal.
        
        Args:
            graph_storage: Graph storage instance to traverse
        """
        self.storage = graph_storage
        self.device = graph_storage.memory_keys.device
        
        # Create attention mechanisms for path-based attention
        self._initialize_attention_mechanisms()
    
    def _initialize_attention_mechanisms(self):
        """Initialize attention mechanisms for path-based traversal."""
        # Bilinear attention: W * [query, value]
        self.attention_bilinear = nn.Bilinear(
            self.storage.key_dim,
            self.storage.value_dim,
            1
        )
        
        # MLP attention: MLP([query; value])
        self.attention_mlp = nn.Sequential(
            nn.Linear(self.storage.key_dim + self.storage.value_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def get_subgraph(self, entity_indices: List[int], max_hops: int = 1) -> Dict[str, Any]:
        """
        Get a subgraph around specified entities.
        
        Args:
            entity_indices: Indices of entities to get subgraph for
            max_hops: Maximum number of hops from seed entities
            
        Returns:
            Dictionary containing subgraph information
        """
        # Initialize sets for nodes and edges
        nodes = set(entity_indices)
        edges = set()
        
        # BFS traversal
        frontier = list(nodes)
        for hop in range(max_hops):
            next_frontier = []
            
            for node_idx in frontier:
                # Get outgoing edges
                for neighbor_idx, relation_type in self.storage.get_entity_neighbors(node_idx):
                    # Add edge
                    edges.add((node_idx, relation_type, neighbor_idx))
                    
                    # Add new node if not visited
                    if neighbor_idx not in nodes:
                        nodes.add(neighbor_idx)
                        next_frontier.append(neighbor_idx)
                
                # Get incoming edges
                for neighbor_idx, relation_type in self.storage.get_entity_incoming(node_idx):
                    # Add edge
                    edges.add((neighbor_idx, relation_type, node_idx))
                    
                    # Add new node if not visited
                    if neighbor_idx not in nodes:
                        nodes.add(neighbor_idx)
                        next_frontier.append(neighbor_idx)
            
            # Update frontier
            frontier = next_frontier
            
            # Break if no new nodes
            if not frontier:
                break
        
        # Convert to lists for serialization
        nodes_list = list(nodes)
        edges_list = list(edges)
        
        # Get node values and types
        node_values = self.storage.memory_values[torch.tensor(nodes_list, device=self.device)]
        node_types = {idx: self.storage.entity_types.get(idx, 0) for idx in nodes_list}
        
        return {
            "nodes": nodes_list,
            "edges": edges_list,
            "node_values": node_values,
            "node_types": node_types
        }
    
    def simple_traversal(
        self,
        start_indices: torch.Tensor,
        max_hops: int = 2,
        top_k: int = 5
    ) -> Tuple[List[torch.Tensor], List[List[List[int]]]]:
        """
        Perform simple traversal from starting entities.
        
        Args:
            start_indices: Tensor of starting entity indices [batch_size, top_k]
            max_hops: Maximum number of hops in traversal
            top_k: Number of top matches to consider from each start point
            
        Returns:
            Tuple of (retrieved_values_list, paths_list)
        """
        batch_size = start_indices.size(0)
        
        # Initialize results
        all_values = []
        all_paths = []
        
        # For each query in the batch
        for b in range(batch_size):
            # Start with initial matches
            current_indices = start_indices[b].tolist()
            
            # Track visited nodes to avoid cycles
            visited = set(current_indices)
            
            # Track paths: (node_idx, path_to_node)
            paths = {idx: [idx] for idx in current_indices}
            
            # Perform traversal
            for hop in range(max_hops):
                next_indices = []
                
                # Expand each current node
                for node_idx in current_indices:
                    # Get neighbors
                    neighbors = self.storage.get_entity_neighbors(node_idx)
                    
                    for neighbor_idx, relation_type in neighbors:
                        if neighbor_idx not in visited:
                            next_indices.append(neighbor_idx)
                            visited.add(neighbor_idx)
                            # Record path
                            paths[neighbor_idx] = paths[node_idx] + [neighbor_idx]
                
                # Add new nodes to current set
                current_indices.extend(next_indices)
                
                # Break if no new nodes found
                if not next_indices:
                    break
            
            # Collect results for this query
            values = self.storage.memory_values[torch.tensor(list(visited), device=self.device)]
            all_values.append(values)
            all_paths.append([paths[idx] for idx in visited])
        
        return all_values, all_paths
    
    def path_based_attention_traversal(
        self,
        query_keys: torch.Tensor,
        start_indices: torch.Tensor,
        max_hops: int = 2,
        top_k: int = 5,
        attention_type: str = "bilinear"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Retrieve entities with path-based attention for more focused knowledge traversal.
        
        Args:
            query_keys: Query keys to use for attention [batch_size, key_dim]
            start_indices: Tensor of starting entity indices [batch_size, top_k]
            max_hops: Maximum number of hops in traversal
            top_k: Number of top matches at each hop
            attention_type: Type of attention to use (bilinear, dot, mlp)
            
        Returns:
            Tuple of (retrieved_values, traversal_metadata)
        """
        batch_size = query_keys.size(0)
        
        # Initialize results
        all_values = []
        all_attentions = []
        all_paths = []
        all_path_types = []  # Store relation types along the path
        
        # For each query in the batch
        for b in range(batch_size):
            # Get initial similarities for this query
            initial_similarities = torch.matmul(
                query_keys[b:b+1], 
                self.storage.memory_keys[start_indices[b]].t()
            ).squeeze(0)
            
            # Start with initial matches and their similarities
            current_indices = start_indices[b].tolist()
            current_similarities = initial_similarities.tolist()
            
            # Track visited nodes to avoid cycles
            visited = set(current_indices)
            
            # Track paths, attention scores, and relation types
            paths = {idx: [idx] for idx in current_indices}
            path_attentions = {idx: [sim] for idx, sim in zip(current_indices, current_similarities)}
            path_relation_types = {idx: [] for idx in current_indices}  # No relation to get to starting nodes
            
            # Perform multi-hop traversal
            for hop in range(max_hops):
                next_indices = []
                next_attentions = []
                next_relations = []
                
                # Expand each current node
                for i, node_idx in enumerate(current_indices):
                    # Get outgoing relations
                    if not isinstance(node_idx, int):
                        node_idx = int(node_idx)
                    
                    # Get relation IDs for this entity
                    relation_ids = self.storage.relations_by_subject.get(node_idx, [])
                    
                    for relation_id in relation_ids:
                        relation = self.storage.relations.get(relation_id)
                        if relation is None:
                            continue
                            
                        # Get object (neighbor) entity
                        neighbor_idx = relation.object_id
                        relation_type = relation.relation_type.value if hasattr(relation.relation_type, 'value') else relation.relation_type
                        
                        if neighbor_idx not in visited:
                            # Get node values
                            neighbor_value = self.storage.memory_values[neighbor_idx].unsqueeze(0)
                            query = query_keys[b].unsqueeze(0)
                            
                            # Calculate attention
                            if attention_type == "bilinear":
                                attention_score = torch.sigmoid(self.attention_bilinear(query, neighbor_value)).item()
                            elif attention_type == "mlp":
                                concat = torch.cat([query, neighbor_value], dim=1)
                                attention_score = torch.sigmoid(self.attention_mlp(concat)).item()
                            else:  # dot product
                                attention_score = F.cosine_similarity(query, neighbor_value).item()
                                attention_score = (attention_score + 1) / 2  # Normalize to [0, 1]
                            
                            # Consider relation metadata (e.g. confidence)
                            relation_confidence = relation.metadata.confidence
                            
                            # Combine multiple factors
                            combined_score = attention_score * relation_confidence
                            
                            # Only proceed if attention is high enough
                            if combined_score > 0.2:  # Threshold
                                next_indices.append(neighbor_idx)
                                next_attentions.append(combined_score)
                                next_relations.append(relation_type)
                                
                                # Mark as visited to avoid cycles
                                visited.add(neighbor_idx)
                                
                                # Record path, attentions, and relation types
                                paths[neighbor_idx] = paths[node_idx] + [neighbor_idx]
                                path_attentions[neighbor_idx] = path_attentions[node_idx] + [combined_score]
                                path_relation_types[neighbor_idx] = path_relation_types[node_idx] + [relation_type]
                
                # Prepare for next hop
                current_indices = next_indices
                current_similarities = next_attentions
                
                # Break if no new nodes found
                if not current_indices:
                    break
            
            # Collect results for this query
            visited_indices = torch.tensor(list(visited), device=self.device)
            if len(visited_indices) > 0:
                values = self.storage.memory_values[visited_indices]
                
                # Compute final attention weights across all retrieved items
                final_attentions = []
                for idx in visited:
                    # Use mean of path attentions
                    final_attentions.append(sum(path_attentions[idx]) / len(path_attentions[idx]))
                
                # Convert to tensor
                attention_weights = torch.tensor(final_attentions, device=self.device)
                
                # Apply softmax to get final distribution
                attention_weights = F.softmax(attention_weights / 0.1, dim=0)  # Temperature of 0.1
                
                # Weighted combination of values
                weighted_values = torch.sum(values * attention_weights.unsqueeze(1), dim=0)
                all_values.append(weighted_values.unsqueeze(0))
                
                # Store path metadata
                all_paths.append([paths[idx] for idx in visited])
                all_attentions.append([path_attentions[idx] for idx in visited])
                all_path_types.append([path_relation_types[idx] for idx in visited])
            else:
                # Fallback if no paths found
                all_values.append(torch.zeros(1, self.storage.value_dim, device=self.device))
                all_paths.append([])
                all_attentions.append([])
                all_path_types.append([])
        
        # Stack results
        if all(v.size(0) > 0 for v in all_values):
            stacked_values = torch.cat(all_values, dim=0)
        else:
            stacked_values = torch.zeros(batch_size, self.storage.value_dim, device=self.device)
        
        # Create metadata
        metadata = {
            "paths": all_paths,
            "attention_scores": all_attentions,
            "relation_types": all_path_types,
            "traversal_method": "path_attention",
            "attention_type": attention_type,
            "max_hops": max_hops
        }
        
        return stacked_values, metadata
    
    def find_shortest_path(
        self,
        start_idx: int,
        end_idx: int,
        max_path_length: int = 5
    ) -> Optional[List[Tuple[int, int, int]]]:
        """
        Find shortest path between two entities.
        
        Args:
            start_idx: Index of the start entity
            end_idx: Index of the end entity
            max_path_length: Maximum path length to consider
            
        Returns:
            List of (entity_idx, relation_type, next_entity_idx) triples,
            or None if no path found
        """
        # Check if both entities exist
        if not (self.storage.valid_mask[start_idx] and self.storage.valid_mask[end_idx]):
            return None
        
        # Use BFS to find shortest path
        queue = deque([(start_idx, [])])  # (node, path_so_far)
        visited = {start_idx}
        
        while queue:
            node, path = queue.popleft()
            
            # Stop if we've reached the target
            if node == end_idx:
                return path
            
            # Stop if path is too long
            if len(path) >= max_path_length:
                continue
            
            # Expand neighbors
            for neighbor_idx, relation_type in self.storage.get_entity_neighbors(node):
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    new_path = path + [(node, relation_type, neighbor_idx)]
                    queue.append((neighbor_idx, new_path))
        
        # No path found
        return None
    
    def find_paths(
        self,
        start_idx: int,
        end_idx: int,
        max_paths: int = 3,
        max_path_length: int = 5
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Find multiple paths between two entities.
        
        Args:
            start_idx: Index of the start entity
            end_idx: Index of the end entity
            max_paths: Maximum number of paths to find
            max_path_length: Maximum path length to consider
            
        Returns:
            List of paths, where each path is a list of
            (entity_idx, relation_type, next_entity_idx) triples
        """
        # Check if both entities exist
        if not (self.storage.valid_mask[start_idx] and self.storage.valid_mask[end_idx]):
            return []
        
        # Use modified BFS to find multiple paths
        paths = []
        visited = set()
        
        # Helper function for DFS search
        def dfs(node, path, depth=0):
            # Base cases
            if depth > max_path_length:
                return
            
            if node == end_idx:
                paths.append(path[:])
                return
            
            if len(paths) >= max_paths:
                return
            
            # Mark as visited for this path
            visited.add(node)
            
            # Explore neighbors
            for neighbor_idx, relation_type in self.storage.get_entity_neighbors(node):
                if neighbor_idx not in visited:
                    path.append((node, relation_type, neighbor_idx))
                    dfs(neighbor_idx, path, depth + 1)
                    path.pop()  # Backtrack
            
            # Unmark for other paths
            visited.remove(node)
        
        # Start DFS from start node
        dfs(start_idx, [])
        
        return paths
    
    def find_common_neighbors(
        self,
        entity_indices: List[int],
        max_common: int = 10
    ) -> List[int]:
        """
        Find common neighbors of multiple entities.
        
        Args:
            entity_indices: List of entity indices to find common neighbors for
            max_common: Maximum number of common neighbors to return
            
        Returns:
            List of common neighbor indices
        """
        if not entity_indices:
            return []
        
        # Get neighbors for each entity
        neighbor_sets = []
        for idx in entity_indices:
            # Get all neighbors (both outgoing and incoming)
            outgoing = {neighbor for neighbor, _ in self.storage.get_entity_neighbors(idx)}
            incoming = {neighbor for neighbor, _ in self.storage.get_entity_incoming(idx)}
            neighbor_sets.append(outgoing.union(incoming))
        
        # Find intersection of all neighbor sets
        common_neighbors = set.intersection(*neighbor_sets) if neighbor_sets else set()
        
        # Convert to list and limit size
        return list(common_neighbors)[:max_common]
    
    def to(self, device: torch.device) -> 'GraphTraversal':
        """
        Move traversal components to specified device.
        
        Args:
            device: Device to move to
            
        Returns:
            Self for chaining
        """
        self.device = device
        self.attention_bilinear = self.attention_bilinear.to(device)
        self.attention_mlp = self.attention_mlp.to(device)
        
        return self