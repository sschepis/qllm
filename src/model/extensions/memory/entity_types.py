"""
Entity Types Module.

This module defines the entity type enumerations and related functionality
for the knowledge graph memory extension.
"""

from enum import Enum
from typing import Dict, Any, Optional


class EntityType(Enum):
    """
    Enumeration of common entity types in the knowledge graph.
    
    These types categorize entities within the knowledge graph, allowing for
    type-specific processing and filtering.
    """
    
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


class EntityTypeRegistry:
    """Registry for entity types with support for custom types."""
    
    def __init__(self):
        """Initialize the entity type registry."""
        # Standard entity types from enum
        self.entity_types = {e.value: e for e in EntityType}
        
        # Custom entity types registry
        self.custom_entity_types = {}
    
    def register_entity_type(self, type_id: int, name: str, description: str = "") -> None:
        """
        Register a custom entity type.
        
        Args:
            type_id: Numeric ID for the type
            name: Name of the entity type
            description: Description of the entity type
        
        Raises:
            ValueError: If the entity type ID already exists
        """
        if type_id in self.entity_types:
            raise ValueError(f"Entity type ID {type_id} already exists")
        
        self.custom_entity_types[type_id] = {
            "name": name,
            "description": description
        }
    
    def get_entity_name(self, type_id: int) -> str:
        """
        Get the name of an entity type.
        
        Args:
            type_id: ID of the entity type
            
        Returns:
            Name of the entity type
        """
        if type_id in self.entity_types:
            return self.entity_types[type_id].name
        
        if type_id in self.custom_entity_types:
            return self.custom_entity_types[type_id]["name"]
        
        return f"CUSTOM_ENTITY_{type_id}"
    
    def get_entity_description(self, type_id: int) -> Optional[str]:
        """
        Get the description of an entity type.
        
        Args:
            type_id: ID of the entity type
            
        Returns:
            Description of the entity type, or None if not available
        """
        if type_id in self.custom_entity_types:
            return self.custom_entity_types[type_id]["description"]
        
        return None
    
    def get_all_entity_types(self) -> Dict[int, str]:
        """
        Get all registered entity types.
        
        Returns:
            Dictionary mapping type IDs to names
        """
        result = {k: v.name for k, v in self.entity_types.items()}
        result.update({k: v["name"] for k, v in self.custom_entity_types.items()})
        return result


# Global entity type registry instance
entity_registry = EntityTypeRegistry()