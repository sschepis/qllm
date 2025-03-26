"""
Memory Extension Module.

This module provides extensions for implementing structured knowledge storage
and retrieval in the Semantic Resonance Language Model.

Module Structure:
- Base memory functionality: base_memory_core.py, memory_storage.py
- Simple memory extension: simple_memory_extension.py
- Knowledge graph functionality: knowledge_graph_extension.py, graph_storage.py, graph_traversal.py, graph_query.py
- Schema definitions: entity_types.py, relation_types.py, relation_metadata.py
- Utilities: memory_utils.py, memory_tracking.py, memory_persistence.py, memory_config.py
"""

# Base memory extension classes
from .base_memory_core import BaseMemoryExtension
from .simple_memory_extension import SimpleMemoryExtension
from .knowledge_graph_extension import KnowledgeGraphExtension

# Configuration
from .memory_config import MemoryConfig, GraphMemoryConfig

# Schema and type definitions
from .entity_types import EntityType, entity_registry
from .relation_types import RelationType, relation_registry
from .relation_metadata import EntityMetadata, RelationMetadata, Relation

# Expose storage classes for advanced usage
from .memory_storage import MemoryStorage
from .graph_storage import GraphStorage
from .graph_traversal import GraphTraversal
from .graph_query import GraphQuery

__all__ = [
    # Base extension classes
    'BaseMemoryExtension',
    'SimpleMemoryExtension', 
    'KnowledgeGraphExtension',
    
    # Configuration
    'MemoryConfig',
    'GraphMemoryConfig',
    
    # Schema and type definitions
    'EntityType',
    'RelationType',
    'EntityMetadata',
    'RelationMetadata',
    'Relation',
    'entity_registry',
    'relation_registry',
    
    # Storage classes
    'MemoryStorage',
    'GraphStorage',
    'GraphTraversal',
    'GraphQuery'
]