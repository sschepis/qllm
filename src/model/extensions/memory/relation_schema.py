"""
Relation Schema Module.

This module defines the schema and metadata structure for relations in the
knowledge graph extension of the Semantic Resonance Language Model.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from dataclasses import dataclass, field


class RelationType(Enum):
    """Enumeration of common relation types in the knowledge graph."""
    
    # General semantic relations
    IS_A = 1                # Hypernymy/taxonomy: X is a type of Y
    HAS_PART = 2            # Meronymy: X has Y as a part
    HAS_PROPERTY = 3        # Attribution: X has property Y
    LOCATED_IN = 4          # Physical location: X is located in Y
    TEMPORAL_BEFORE = 5     # Temporal ordering: X occurs before Y
    TEMPORAL_AFTER = 6      # Temporal ordering: X occurs after Y
    CAUSES = 7              # Causality: X causes Y
    USED_FOR = 8            # Purpose: X is used for Y
    SYNONYMY = 9            # Same/similar meaning: X means the same as Y
    ANTONYMY = 10           # Opposite meaning: X means the opposite of Y
    
    # Agent-specific relations
    AGENT_OF = 11           # Agency: X is the agent of action Y
    RECIPIENT_OF = 12       # Recipient: X is the recipient of action Y
    EXPERIENCES = 13        # Experience: X experiences state/event Y
    
    # Domain-specific relations
    AUTHOR_OF = 21          # Authorship: X is the author of Y
    MEMBER_OF = 22          # Membership: X is a member of group Y
    CONTAINS = 23           # Containment: X contains Y
    DERIVED_FROM = 24       # Derivation: X is derived from Y
    INSTANCE_OF = 25        # Instantiation: X is an instance of class Y
    
    # Meta-relations
    CONFIDENCE = 90         # Confidence in another relation
    SOURCE = 91             # Source of information for another relation
    TEMPORAL_SCOPE = 92     # Time period when relation holds
    
    # Custom relation
    CUSTOM = 99             # Custom relation type


@dataclass
class RelationMetadata:
    """Metadata for a relation in the knowledge graph."""
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
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
        """Create metadata from dictionary."""
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
class Relation:
    """A relation between two entities in the knowledge graph."""
    
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
        """Convert relation to dictionary for serialization."""
        return {
            "subject_id": self.subject_id,
            "object_id": self.object_id,
            "relation_type": self.relation_type.value if isinstance(self.relation_type, RelationType) else self.relation_type,
            "metadata": self.metadata.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relation':
        """Create relation from dictionary."""
        metadata = RelationMetadata.from_dict(data.get("metadata", {}))
        return cls(
            subject_id=data["subject_id"],
            object_id=data["object_id"],
            relation_type=data["relation_type"],
            metadata=metadata
        )


@dataclass
class EntityType(Enum):
    """Enumeration of common entity types in the knowledge graph."""
    
    # Basic types
    GENERIC = 0             # Generic/unknown entity
    CONCEPT = 1             # Abstract concept
    OBJECT = 2              # Physical object
    PERSON = 3              # Person or agent
    LOCATION = 4            # Physical location
    EVENT = 5               # Event or happening
    TIME = 6                # Time point or period
    
    # Knowledge domain types
    CATEGORY = 10           # Category or classification
    FACT = 11               # Factual statement
    PROPERTY = 12           # Property or attribute
    ACTION = 13             # Action or process
    STATE = 14              # State or condition
    
    # Domain-specific types
    ORGANIZATION = 20       # Organization or institution
    CREATIVE_WORK = 21      # Creative work (book, movie, etc.)
    BIOLOGICAL_ENTITY = 22  # Biological entity (species, etc.)
    NUMERICAL_VALUE = 23    # Numerical value or quantity
    CODE_ENTITY = 24        # Programming code or function
    
    # Custom type
    CUSTOM = 99             # Custom entity type


@dataclass
class EntityMetadata:
    """Metadata for an entity in the knowledge graph."""
    
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
        # Convert integer to EntityType if needed
        if isinstance(self.entity_type, int):
            try:
                self.entity_type = EntityType(self.entity_type)
            except ValueError:
                # Use custom type for unknown entity
                self.entity_type = EntityType.CUSTOM
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
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
        """Create metadata from dictionary."""
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


class SchemaRegistry:
    """Registry for relation and entity types and their schemas."""
    
    def __init__(self):
        """Initialize the schema registry."""
        self.relation_types = {r.value: r for r in RelationType}
        self.entity_types = {e.value: e for e in EntityType}
        
        # Custom type registries
        self.custom_relation_types = {}
        self.custom_entity_types = {}
    
    def register_relation_type(self, type_id: int, name: str, description: str = "") -> None:
        """
        Register a custom relation type.
        
        Args:
            type_id (int): Numeric ID for the type
            name (str): Name of the relation type
            description (str): Description of the relation type
        """
        if type_id in self.relation_types:
            raise ValueError(f"Relation type ID {type_id} already exists")
        
        self.custom_relation_types[type_id] = {
            "name": name,
            "description": description
        }
    
    def register_entity_type(self, type_id: int, name: str, description: str = "") -> None:
        """
        Register a custom entity type.
        
        Args:
            type_id (int): Numeric ID for the type
            name (str): Name of the entity type
            description (str): Description of the entity type
        """
        if type_id in self.entity_types:
            raise ValueError(f"Entity type ID {type_id} already exists")
        
        self.custom_entity_types[type_id] = {
            "name": name,
            "description": description
        }
    
    def get_relation_name(self, type_id: int) -> str:
        """Get the name of a relation type."""
        if type_id in self.relation_types:
            return self.relation_types[type_id].name
        
        if type_id in self.custom_relation_types:
            return self.custom_relation_types[type_id]["name"]
        
        return f"CUSTOM_RELATION_{type_id}"
    
    def get_entity_name(self, type_id: int) -> str:
        """Get the name of an entity type."""
        if type_id in self.entity_types:
            return self.entity_types[type_id].name
        
        if type_id in self.custom_entity_types:
            return self.custom_entity_types[type_id]["name"]
        
        return f"CUSTOM_ENTITY_{type_id}"


# Global schema registry instance
registry = SchemaRegistry()