"""
Automated Knowledge Graph Construction Module.

This module provides capabilities for automatically constructing a knowledge graph
from textual context and model outputs without explicit supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union, Set
import math
import re

from .relation_types import RelationType
from .entity_types import EntityType
from .relation_metadata import Relation, EntityMetadata


class EntityExtractor(nn.Module):
    """
    Extracts entities from text and context.
    
    This module identifies potential entities in text and builds structured
    representations for them, including entity typing.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        max_entities_per_text: int = 20,
        min_entity_score: float = 0.4,
        dropout: float = 0.1
    ):
        """
        Initialize the entity extractor.
        
        Args:
            embedding_dim: Dimension of entity embeddings
            hidden_dim: Dimension of hidden layers
            max_entities_per_text: Maximum number of entities to extract per text
            min_entity_score: Minimum score threshold for entity extraction
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_entities_per_text = max_entities_per_text
        self.min_entity_score = min_entity_score
        
        # Entity candidate scorer
        self.entity_scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Entity type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(EntityType))
        )
        
        # Entity span representation enhancer
        self.span_enhancer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def extract_entities(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[List[Tuple[int, int, float]], torch.Tensor]:
        """
        Extract entity spans from hidden states.
        
        Args:
            hidden_states: Model hidden states [batch_size, seq_len, embedding_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple containing:
                - List of entity spans: (start_idx, end_idx, score)
                - Entity scores: [batch_size, seq_len]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Get token-level entity scores
        entity_token_scores = self.entity_scorer(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            entity_token_scores = entity_token_scores * attention_mask
        
        # Extract entity spans using a simple heuristic
        # In a real implementation, use more sophisticated techniques
        entity_spans = []
        
        for b in range(batch_size):
            scores = entity_token_scores[b]
            mask = attention_mask[b] if attention_mask is not None else torch.ones_like(scores)
            
            # Find spans with consecutive high scores
            in_entity = False
            start_idx = 0
            current_span_scores = []
            
            for i in range(seq_len):
                if mask[i].item() == 0:  # Padding token
                    if in_entity:
                        # End current entity
                        end_idx = i - 1
                        if end_idx >= start_idx:  # Ensure valid span
                            avg_score = sum(current_span_scores) / len(current_span_scores)
                            if avg_score >= self.min_entity_score:
                                entity_spans.append((b, start_idx, end_idx, avg_score))
                        in_entity = False
                    continue
                
                if scores[i].item() >= self.min_entity_score:
                    if not in_entity:
                        # Start new entity
                        start_idx = i
                        current_span_scores = [scores[i].item()]
                        in_entity = True
                    else:
                        # Continue entity
                        current_span_scores.append(scores[i].item())
                elif in_entity:
                    # End current entity
                    end_idx = i - 1
                    if end_idx >= start_idx:  # Ensure valid span
                        avg_score = sum(current_span_scores) / len(current_span_scores)
                        if avg_score >= self.min_entity_score:
                            entity_spans.append((b, start_idx, end_idx, avg_score))
                    in_entity = False
            
            # Handle entity at end of sequence
            if in_entity:
                end_idx = seq_len - 1
                avg_score = sum(current_span_scores) / len(current_span_scores)
                if avg_score >= self.min_entity_score:
                    entity_spans.append((b, start_idx, end_idx, avg_score))
        
        # Sort by score and limit per batch
        filtered_spans = []
        batch_counts = {}
        
        for span in sorted(entity_spans, key=lambda x: x[3], reverse=True):
            batch_idx = span[0]
            if batch_idx not in batch_counts:
                batch_counts[batch_idx] = 0
                
            if batch_counts[batch_idx] < self.max_entities_per_text:
                filtered_spans.append(span)
                batch_counts[batch_idx] += 1
        
        return filtered_spans, entity_token_scores
    
    def create_entity_representations(
        self,
        hidden_states: torch.Tensor,
        entity_spans: List[Tuple[int, int, int, float]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create entity representations from span hidden states.
        
        Args:
            hidden_states: Model hidden states [batch_size, seq_len, embedding_dim]
            entity_spans: List of entity spans (batch_idx, start_idx, end_idx, score)
            
        Returns:
            Tuple containing:
                - Entity representations [num_entities, embedding_dim]
                - Entity type logits [num_entities, num_types]
                - Entity scores [num_entities]
        """
        if not entity_spans:
            device = hidden_states.device
            return (
                torch.zeros(0, self.embedding_dim, device=device),
                torch.zeros(0, len(EntityType), device=device),
                torch.zeros(0, device=device)
            )
        
        # Extract span representations
        span_representations = []
        span_scores = []
        
        for span in entity_spans:
            batch_idx, start_idx, end_idx, score = span
            
            # Extract span hidden states
            span_hidden = hidden_states[batch_idx, start_idx:end_idx+1]
            
            # Create span representation (using mean pooling)
            span_rep = span_hidden.mean(dim=0)
            
            span_representations.append(span_rep)
            span_scores.append(score)
        
        # Stack representations and scores
        entity_representations = torch.stack(span_representations)
        entity_scores = torch.tensor(span_scores, device=entity_representations.device)
        
        # Enhance span representations
        enhanced_representations = self.span_enhancer(entity_representations)
        
        # Classify entity types
        type_logits = self.type_classifier(enhanced_representations)
        
        return enhanced_representations, type_logits, entity_scores
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Extract and represent entities from hidden states.
        
        Args:
            hidden_states: Model hidden states [batch_size, seq_len, embedding_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary containing entity representations and metadata
        """
        # Extract entity spans
        entity_spans, token_scores = self.extract_entities(hidden_states, attention_mask)
        
        # Create entity representations
        entity_representations, type_logits, entity_scores = self.create_entity_representations(
            hidden_states, entity_spans
        )
        
        # Get entity types
        entity_type_probs = F.softmax(type_logits, dim=-1)
        entity_types = torch.argmax(type_logits, dim=-1)
        
        return {
            "entity_spans": entity_spans,
            "entity_representations": entity_representations,
            "entity_scores": entity_scores,
            "entity_types": entity_types,
            "entity_type_probs": entity_type_probs,
            "token_entity_scores": token_scores
        }


class RelationExtractor(nn.Module):
    """
    Extracts relations between entities.
    
    This module identifies potential relations between extracted entities
    and classifies their types.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        relation_threshold: float = 0.5,
        max_relations_per_entity: int = 5,
        dropout: float = 0.1
    ):
        """
        Initialize the relation extractor.
        
        Args:
            embedding_dim: Dimension of entity embeddings
            hidden_dim: Dimension of hidden layers
            relation_threshold: Threshold for relation extraction
            max_relations_per_entity: Maximum relations per entity
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.relation_threshold = relation_threshold
        self.max_relations_per_entity = max_relations_per_entity
        
        # Relation scorer
        self.relation_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Relation type classifier
        self.relation_classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(RelationType))
        )
        
        # Relation direction classifier
        self.direction_classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def extract_relations(
        self,
        entity_representations: torch.Tensor,
        entity_spans: List[Tuple[int, int, int, float]],
        hidden_states: torch.Tensor
    ) -> List[Tuple[int, int, float, int]]:
        """
        Extract relations between entities.
        
        Args:
            entity_representations: Entity representations [num_entities, embedding_dim]
            entity_spans: Entity spans (batch_idx, start_idx, end_idx, score)
            hidden_states: Model hidden states [batch_size, seq_len, embedding_dim]
            
        Returns:
            List of relations: (entity1_idx, entity2_idx, score, relation_type)
        """
        num_entities = len(entity_representations)
        if num_entities <= 1:
            return []
        
        # Compute all pairwise combinations
        relations = []
        
        for i in range(num_entities):
            for j in range(num_entities):
                if i == j:
                    continue
                
                # Get entity spans
                span_i = entity_spans[i]
                span_j = entity_spans[j]
                
                # Only consider entities in the same batch
                if span_i[0] != span_j[0]:
                    continue
                
                # Create pair representation
                entity_pair = torch.cat([
                    entity_representations[i],
                    entity_representations[j]
                ])
                
                # Score relation
                relation_score = self.relation_scorer(entity_pair.unsqueeze(0)).squeeze()
                
                # Determine relation type if score is high enough
                if relation_score.item() >= self.relation_threshold:
                    # Classify relation type
                    relation_logits = self.relation_classifier(entity_pair.unsqueeze(0))
                    relation_type = torch.argmax(relation_logits, dim=-1).item()
                    
                    # Add to relations
                    relations.append((i, j, relation_score.item(), relation_type))
        
        # Sort by score and limit relations per entity
        filtered_relations = []
        entity_relation_counts = {}
        
        for rel in sorted(relations, key=lambda x: x[2], reverse=True):
            src_idx, tgt_idx = rel[0], rel[1]
            
            # Initialize counts
            if src_idx not in entity_relation_counts:
                entity_relation_counts[src_idx] = 0
            if tgt_idx not in entity_relation_counts:
                entity_relation_counts[tgt_idx] = 0
            
            # Check if we've reached the limit
            if (entity_relation_counts[src_idx] < self.max_relations_per_entity and
                entity_relation_counts[tgt_idx] < self.max_relations_per_entity):
                filtered_relations.append(rel)
                entity_relation_counts[src_idx] += 1
                entity_relation_counts[tgt_idx] += 1
        
        return filtered_relations
    
    def forward(
        self,
        entity_representations: torch.Tensor,
        entity_spans: List[Tuple[int, int, int, float]],
        hidden_states: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Extract relations between entities.
        
        Args:
            entity_representations: Entity representations [num_entities, embedding_dim]
            entity_spans: Entity spans (batch_idx, start_idx, end_idx, score)
            hidden_states: Model hidden states [batch_size, seq_len, embedding_dim]
            
        Returns:
            Dictionary containing extracted relations and metadata
        """
        # Extract relations
        relations = self.extract_relations(entity_representations, entity_spans, hidden_states)
        
        # Create tensors for relation properties
        num_relations = len(relations)
        device = entity_representations.device
        
        if num_relations == 0:
            return {
                "relations": relations,
                "relation_scores": torch.zeros(0, device=device),
                "relation_types": torch.zeros(0, dtype=torch.long, device=device)
            }
        
        # Create relation tensors
        relation_indices = torch.tensor([[r[0], r[1]] for r in relations], device=device)
        relation_scores = torch.tensor([r[2] for r in relations], device=device)
        relation_types = torch.tensor([r[3] for r in relations], device=device)
        
        # Classify relation directions
        relation_pairs = torch.cat([
            entity_representations[relation_indices[:, 0]],
            entity_representations[relation_indices[:, 1]]
        ], dim=-1)
        
        direction_scores = self.direction_classifier(relation_pairs).squeeze(-1)
        
        return {
            "relations": relations,
            "relation_indices": relation_indices,
            "relation_scores": relation_scores,
            "relation_types": relation_types,
            "direction_scores": direction_scores
        }


class GraphConstructor(nn.Module):
    """
    Constructs a knowledge graph from extracted entities and relations.
    
    This module assembles a structured knowledge graph from the entities and
    relations extracted from context, creating a coherent knowledge representation.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        entity_type_embedding_dim: int = 64,
        relation_embedding_dim: int = 128,
        confidence_threshold: float = 0.6,
        max_entities: int = 1000,
        dropout: float = 0.1
    ):
        """
        Initialize the graph constructor.
        
        Args:
            embedding_dim: Dimension of entity embeddings
            entity_type_embedding_dim: Dimension of entity type embeddings
            relation_embedding_dim: Dimension of relation embeddings
            confidence_threshold: Confidence threshold for adding to graph
            max_entities: Maximum number of entities to maintain
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.entity_type_embedding_dim = entity_type_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.confidence_threshold = confidence_threshold
        self.max_entities = max_entities
        
        # Entity deduplication model
        self.entity_similarity = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        # Entity type embedding
        self.entity_type_embedding = nn.Embedding(
            len(EntityType),
            entity_type_embedding_dim
        )
        
        # Relation type embedding
        self.relation_type_embedding = nn.Embedding(
            len(RelationType),
            relation_embedding_dim
        )
        
        # Entity update model (for merging duplicates)
        self.entity_merger = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def update_graph(
        self,
        existing_entities: Dict[int, torch.Tensor],
        existing_entity_types: Dict[int, torch.Tensor],
        existing_relations: List[Relation],
        new_entities: torch.Tensor,
        new_entity_spans: List[Tuple[int, int, int, float]],
        new_entity_types: torch.Tensor,
        new_relations: List[Tuple[int, int, float, int]],
        text_spans: Optional[Dict[int, Tuple[int, int, int]]] = None
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], List[Relation]]:
        """
        Update the knowledge graph with new entities and relations.
        
        Args:
            existing_entities: Dictionary of existing entity representations
            existing_entity_types: Dictionary of existing entity types
            existing_relations: List of existing relations
            new_entities: Tensor of new entity representations
            new_entity_spans: Spans for new entities
            new_entity_types: Types for new entities
            new_relations: List of new relations between new entities
            text_spans: Optional mapping of entity IDs to text spans
            
        Returns:
            Tuple containing:
                - Updated entity representations
                - Updated entity types
                - Updated relations
        """
        device = new_entities.device
        
        # Copy existing entities and relations
        updated_entities = existing_entities.copy()
        updated_entity_types = existing_entity_types.copy()
        updated_relations = existing_relations.copy()
        
        # Map from new entity indices to final entity IDs
        entity_id_map = {}
        
        # Generate next entity ID
        next_entity_id = max(existing_entities.keys()) + 1 if existing_entities else 0
        
        # First pass: Handle entity deduplication
        for i, entity_rep in enumerate(new_entities):
            # Check if entity already exists
            is_duplicate = False
            duplicate_id = None
            highest_similarity = 0.0
            
            for entity_id, existing_rep in existing_entities.items():
                # Compute similarity
                pair_rep = torch.cat([entity_rep, existing_rep])
                similarity = self.entity_similarity(pair_rep.unsqueeze(0)).squeeze().item()
                
                if similarity > 0.8 and similarity > highest_similarity:
                    is_duplicate = True
                    duplicate_id = entity_id
                    highest_similarity = similarity
            
            if is_duplicate and duplicate_id is not None:
                # Merge with existing entity
                entity_id_map[i] = duplicate_id
                
                # Update entity representation
                merged_rep = self.entity_merger(
                    torch.cat([entity_rep, existing_entities[duplicate_id]]).unsqueeze(0)
                ).squeeze(0)
                
                updated_entities[duplicate_id] = merged_rep
            else:
                # Add as new entity
                entity_id_map[i] = next_entity_id
                
                updated_entities[next_entity_id] = entity_rep
                updated_entity_types[next_entity_id] = new_entity_types[i]
                
                next_entity_id += 1
        
        # Second pass: Add new relations
        for src_idx, tgt_idx, score, rel_type in new_relations:
            # Skip if confidence too low
            if score < self.confidence_threshold:
                continue
                
            # Map to final entity IDs
            if src_idx not in entity_id_map or tgt_idx not in entity_id_map:
                continue
                
            src_id = entity_id_map[src_idx]
            tgt_id = entity_id_map[tgt_idx]
            
            # Create relation metadata
            metadata = {
                "confidence": score,
                "auto_extracted": True
            }
            
            # Add text span information if available
            if text_spans is not None:
                if src_idx < len(new_entity_spans):
                    src_span = new_entity_spans[src_idx]
                    metadata["source_span"] = (src_span[0], src_span[1], src_span[2])
                    
                if tgt_idx < len(new_entity_spans):
                    tgt_span = new_entity_spans[tgt_idx]
                    metadata["target_span"] = (tgt_span[0], tgt_span[1], tgt_span[2])
            
            # Create relation
            relation = Relation(
                source_id=src_id,
                target_id=tgt_id,
                relation_type=rel_type,
                metadata=metadata
            )
            
            # Check if relation already exists
            is_duplicate = False
            for existing_rel in updated_relations:
                if (existing_rel.source_id == src_id and 
                    existing_rel.target_id == tgt_id and
                    existing_rel.relation_type == rel_type):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                updated_relations.append(relation)
        
        # Limit total number of entities if needed
        if len(updated_entities) > self.max_entities:
            # Sort entities by number of relations
            entity_rel_counts = {}
            for rel in updated_relations:
                src_id = rel.source_id
                tgt_id = rel.target_id
                
                if src_id not in entity_rel_counts:
                    entity_rel_counts[src_id] = 0
                if tgt_id not in entity_rel_counts:
                    entity_rel_counts[tgt_id] = 0
                    
                entity_rel_counts[src_id] += 1
                entity_rel_counts[tgt_id] += 1
            
            # Keep most connected entities
            entities_to_keep = sorted(
                entity_rel_counts.keys(),
                key=lambda x: entity_rel_counts.get(x, 0),
                reverse=True
            )[:self.max_entities]
            
            entities_to_keep = set(entities_to_keep)
            
            # Filter entities and relations
            filtered_entities = {k: v for k, v in updated_entities.items() if k in entities_to_keep}
            filtered_entity_types = {k: v for k, v in updated_entity_types.items() if k in entities_to_keep}
            filtered_relations = [
                rel for rel in updated_relations
                if rel.source_id in entities_to_keep and rel.target_id in entities_to_keep
            ]
            
            return filtered_entities, filtered_entity_types, filtered_relations
        
        return updated_entities, updated_entity_types, updated_relations
    
    def forward(
        self,
        entity_extractor_output: Dict[str, Any],
        relation_extractor_output: Dict[str, Any],
        existing_entities: Dict[int, torch.Tensor],
        existing_entity_types: Dict[int, torch.Tensor],
        existing_relations: List[Relation],
        hidden_states: torch.Tensor
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], List[Relation]]:
        """
        Construct and update the knowledge graph.
        
        Args:
            entity_extractor_output: Output from entity extractor
            relation_extractor_output: Output from relation extractor
            existing_entities: Dictionary of existing entity representations
            existing_entity_types: Dictionary of existing entity types
            existing_relations: List of existing relations
            hidden_states: Model hidden states
            
        Returns:
            Tuple containing:
                - Updated entity representations
                - Updated entity types
                - Updated relations
        """
        # Extract entities and relations
        new_entities = entity_extractor_output["entity_representations"]
        new_entity_spans = entity_extractor_output["entity_spans"]
        new_entity_types = entity_extractor_output["entity_types"]
        
        new_relations = relation_extractor_output["relations"]
        
        # Update the graph
        updated_entities, updated_types, updated_relations = self.update_graph(
            existing_entities=existing_entities,
            existing_entity_types=existing_entity_types,
            existing_relations=existing_relations,
            new_entities=new_entities,
            new_entity_spans=new_entity_spans,
            new_entity_types=new_entity_types,
            new_relations=new_relations
        )
        
        return updated_entities, updated_types, updated_relations


class AutoGraphConstructionModule(nn.Module):
    """
    Module for automated knowledge graph construction from context.
    
    This module automatically builds and maintains a knowledge graph from
    textual context without requiring explicit supervision.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        entity_type_embedding_dim: int = 64,
        relation_embedding_dim: int = 128,
        hidden_dim: int = 256,
        max_entities: int = 1000,
        confidence_threshold: float = 0.6,
        dropout: float = 0.1
    ):
        """
        Initialize the auto graph construction module.
        
        Args:
            embedding_dim: Dimension of entity embeddings
            entity_type_embedding_dim: Dimension of entity type embeddings
            relation_embedding_dim: Dimension of relation embeddings
            hidden_dim: Dimension of hidden layers
            max_entities: Maximum number of entities to maintain
            confidence_threshold: Confidence threshold for adding to graph
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.entity_type_embedding_dim = entity_type_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim
        
        # Entity extraction component
        self.entity_extractor = EntityExtractor(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Relation extraction component
        self.relation_extractor = RelationExtractor(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Graph construction component
        self.graph_constructor = GraphConstructor(
            embedding_dim=embedding_dim,
            entity_type_embedding_dim=entity_type_embedding_dim,
            relation_embedding_dim=relation_embedding_dim,
            confidence_threshold=confidence_threshold,
            max_entities=max_entities,
            dropout=dropout
        )
        
        # Context-aware entity enrichment
        self.entity_enricher = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        existing_entities: Optional[Dict[int, torch.Tensor]] = None,
        existing_entity_types: Optional[Dict[int, torch.Tensor]] = None,
        existing_relations: Optional[List[Relation]] = None,
        skip_extraction: bool = False
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], List[Relation], Dict[str, Any]]:
        """
        Extract and construct a knowledge graph from context.
        
        Args:
            hidden_states: Model hidden states [batch_size, seq_len, embedding_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            existing_entities: Optional dict of existing entity representations
            existing_entity_types: Optional dict of existing entity types
            existing_relations: Optional list of existing relations
            skip_extraction: Whether to skip extraction (for inference)
            
        Returns:
            Tuple containing:
                - Dictionary of entity representations
                - Dictionary of entity types
                - List of relations
                - Dictionary of metadata
        """
        # Initialize empty knowledge graph if not provided
        if existing_entities is None:
            existing_entities = {}
        if existing_entity_types is None:
            existing_entity_types = {}
        if existing_relations is None:
            existing_relations = []
        
        # Skip extraction if requested
        if skip_extraction:
            return existing_entities, existing_entity_types, existing_relations, {}
        
        # Extract entities
        entity_outputs = self.entity_extractor(hidden_states, attention_mask)
        
        # Extract relations between entities
        relation_outputs = self.relation_extractor(
            entity_outputs["entity_representations"],
            entity_outputs["entity_spans"],
            hidden_states
        )
        
        # Construct and update knowledge graph
        updated_entities, updated_types, updated_relations = self.graph_constructor(
            entity_extractor_output=entity_outputs,
            relation_extractor_output=relation_outputs,
            existing_entities=existing_entities,
            existing_entity_types=existing_entity_types,
            existing_relations=existing_relations,
            hidden_states=hidden_states
        )
        
        # Enrich entity representations with context
        if self.training:
            # Compute global context representation
            global_context = hidden_states.mean(dim=1)  # [batch_size, embedding_dim]
            
            # Repeat for each entity (assuming small number of entities per batch)
            for entity_id, entity_rep in updated_entities.items():
                # Find which batch this entity came from
                # For simplicity, we'll just use the first batch
                # In practice, maintain entity-to-batch mapping
                context_rep = global_context[0]
                
                # Enrich with context
                enriched_rep = self.entity_enricher(
                    torch.cat([entity_rep, context_rep]).unsqueeze(0)
                ).squeeze(0)
                
                # Update entity representation
                updated_entities[entity_id] = enriched_rep
        
        # Create metadata
        metadata = {
            "num_extracted_entities": len(entity_outputs["entity_spans"]),
            "num_extracted_relations": len(relation_outputs["relations"]),
            "num_entities": len(updated_entities),
            "num_relations": len(updated_relations)
        }
        
        return updated_entities, updated_types, updated_relations, metadata