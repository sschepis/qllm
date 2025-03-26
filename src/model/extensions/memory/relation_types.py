"""
Relation Types Module.

This module defines the relation type enumerations and related functionality
for the knowledge graph memory extension.
"""

from enum import Enum
from typing import Dict, Any, Optional


class RelationType(Enum):
    """
    Enumeration of common relation types in the knowledge graph.
    
    These types define the semantic relationships between entities in the
    knowledge graph.
    """
    
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


class RelationTypeRegistry:
    """Registry for relation types with support for custom types."""
    
    def __init__(self):
        """Initialize the relation type registry."""
        # Standard relation types from enum
        self.relation_types = {r.value: r for r in RelationType}
        
        # Custom relation types registry
        self.custom_relation_types = {}
    
    def register_relation_type(self, type_id: int, name: str, description: str = "") -> None:
        """
        Register a custom relation type.
        
        Args:
            type_id: Numeric ID for the type
            name: Name of the relation type
            description: Description of the relation type
        
        Raises:
            ValueError: If the relation type ID already exists
        """
        if type_id in self.relation_types:
            raise ValueError(f"Relation type ID {type_id} already exists")
        
        self.custom_relation_types[type_id] = {
            "name": name,
            "description": description
        }
    
    def get_relation_name(self, type_id: int) -> str:
        """
        Get the name of a relation type.
        
        Args:
            type_id: ID of the relation type
            
        Returns:
            Name of the relation type
        """
        if type_id in self.relation_types:
            return self.relation_types[type_id].name
        
        if type_id in self.custom_relation_types:
            return self.custom_relation_types[type_id]["name"]
        
        return f"CUSTOM_RELATION_{type_id}"
    
    def get_relation_description(self, type_id: int) -> Optional[str]:
        """
        Get the description of a relation type.
        
        Args:
            type_id: ID of the relation type
            
        Returns:
            Description of the relation type, or None if not available
        """
        if type_id in self.custom_relation_types:
            return self.custom_relation_types[type_id]["description"]
        
        return None
    
    def get_all_relation_types(self) -> Dict[int, str]:
        """
        Get all registered relation types.
        
        Returns:
            Dictionary mapping type IDs to names
        """
        result = {k: v.name for k, v in self.relation_types.items()}
        result.update({k: v["name"] for k, v in self.custom_relation_types.items()})
        return result


# Global relation type registry instance
relation_registry = RelationTypeRegistry()