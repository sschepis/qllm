"""
Relation Metadata Module.

This module defines metadata classes for relations and entities in the
knowledge graph memory extension.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
import time

from .entity_types import EntityType
from .relation_types import RelationType


@dataclass
class RelationMetadata:
    """
    Metadata for a relation in the knowledge graph.
    
    This class stores additional information about relations beyond
    the core subject-predicate-object structure.
    """
    
    # Confidence score (0.0-1.0)
    confidence: float = 1.0
    
    # Source of this relation
    source: Optional[str] = None
    
    # Timestamp (creation/modification)
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    
    # Temporal scope (when the relation is valid)
    valid_from: Optional[float] = None
    valid_to: Optional[float] = None
    
    # Weight/strength of relation (for weighted graphs)
    weight: float = 1.0
    
    # Custom attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set default timestamps if not provided
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to dictionary for serialization.
        
        Returns:
            Dictionary representation of the metadata
        """
        return {
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "weight": self.weight,
            "attributes": self.attributes.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RelationMetadata':
        """
        Create metadata from dictionary.
        
        Args:
            data: Dictionary containing metadata
            
        Returns:
            RelationMetadata instance
        """
        return cls(
            confidence=data.get("confidence", 1.0),
            source=data.get("source"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            valid_from=data.get("valid_from"),
            valid_to=data.get("valid_to"),
            weight=data.get("weight", 1.0),
            attributes=data.get("attributes", {}).copy()
        )


@dataclass
class EntityMetadata:
    """
    Metadata for an entity in the knowledge graph.
    
    This class stores additional information about entities beyond
    their vector representation.
    """
    
    # Entity type
    entity_type: Union[EntityType, int] = EntityType.GENERIC
    
    # Name or label (for human readability)
    name: Optional[str] = None
    
    # Description
    description: Optional[str] = None
    
    # Confidence score (0.0-1.0)
    confidence: float = 1.0
    
    # Source of this entity
    source: Optional[str] = None
    
    # Timestamp (creation/modification)
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    
    # Importance score (for prioritization)
    importance: float = 0.5
    
    # Custom attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set default timestamps if not provided
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = time.time()
            
        # Convert integer to EntityType if needed
        if isinstance(self.entity_type, int):
            try:
                self.entity_type = EntityType(self.entity_type)
            except ValueError:
                # Use custom type for unknown entity
                self.entity_type = EntityType.CUSTOM
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to dictionary for serialization.
        
        Returns:
            Dictionary representation of the metadata
        """
        return {
            "entity_type": self.entity_type.value if isinstance(self.entity_type, EntityType) else self.entity_type,
            "name": self.name,
            "description": self.description,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "importance": self.importance,
            "attributes": self.attributes.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityMetadata':
        """
        Create metadata from dictionary.
        
        Args:
            data: Dictionary containing metadata
            
        Returns:
            EntityMetadata instance
        """
        return cls(
            entity_type=data.get("entity_type", EntityType.GENERIC),
            name=data.get("name"),
            description=data.get("description"),
            confidence=data.get("confidence", 1.0),
            source=data.get("source"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            importance=data.get("importance", 0.5),
            attributes=data.get("attributes", {}).copy()
        )


@dataclass
class Relation:
    """
    A relation between two entities in the knowledge graph.
    
    This class represents the core triple structure (subject-predicate-object)
    of a knowledge graph relation, with additional metadata.
    """
    
    # Subject entity ID
    subject_id: int
    
    # Object entity ID
    object_id: int
    
    # Relation type
    relation_type: Union[RelationType, int]
    
    # Relation metadata
    metadata: RelationMetadata = field(default_factory=RelationMetadata)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Convert integer to RelationType if needed
        if isinstance(self.relation_type, int):
            try:
                self.relation_type = RelationType(self.relation_type)
            except ValueError:
                # Use custom type for unknown relation
                self.relation_type = RelationType.CUSTOM
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert relation to dictionary for serialization.
        
        Returns:
            Dictionary representation of the relation
        """
        return {
            "subject_id": self.subject_id,
            "object_id": self.object_id,
            "relation_type": self.relation_type.value if isinstance(self.relation_type, RelationType) else self.relation_type,
            "metadata": self.metadata.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relation':
        """
        Create relation from dictionary.
        
        Args:
            data: Dictionary containing relation data
            
        Returns:
            Relation instance
        """
        metadata = RelationMetadata.from_dict(data.get("metadata", {}))
        return cls(
            subject_id=data["subject_id"],
            object_id=data["object_id"],
            relation_type=data["relation_type"],
            metadata=metadata
        )