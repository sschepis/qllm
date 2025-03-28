"""
Core utilities module for QLLM.

This module provides foundational components shared across the codebase
to reduce duplication and improve consistency, including:
- Base model implementations
- Shared text generation utilities
- Unified checkpoint handling
- Consolidated configuration utilities
"""

from src.core.base_model import BaseModel
from src.core.generation import (
    generate_text, 
    apply_temperature, 
    apply_top_k_filtering, 
    apply_top_p_filtering
)
from src.core.checkpoint import (
    save_checkpoint, 
    load_checkpoint, 
    find_latest_checkpoint
)
from src.core.configuration import (
    ConfigurationBase, 
    ConfigurationStrategy, 
    ConfigurationManager
)

__all__ = [
    # Base model
    'BaseModel',
    
    # Generation utilities
    'generate_text',
    'apply_temperature',
    'apply_top_k_filtering',
    'apply_top_p_filtering',
    
    # Checkpoint utilities
    'save_checkpoint',
    'load_checkpoint',
    'find_latest_checkpoint',
    
    # Configuration utilities
    'ConfigurationBase',
    'ConfigurationStrategy',
    'ConfigurationManager'
]