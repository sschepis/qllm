"""
Utility modules for QLLM data processing.

This package provides various utility functions and classes for data
processing, batching, and augmentation in the QLLM system.
"""

# Import utilities
from src.data.utils.tensor_collate import (
    default_collate_fn,
    dialogue_collate_fn,
    function_calling_collate_fn,
    multimodal_collate_fn
)

from src.data.utils.caching import (
    setup_cache_dir,
    load_from_cache,
    save_to_cache
)

# Define what's exposed
tensor_collate = {
    "default_collate_fn": default_collate_fn,
    "dialogue_collate_fn": dialogue_collate_fn,
    "function_calling_collate_fn": function_calling_collate_fn,
    "multimodal_collate_fn": multimodal_collate_fn
}

caching = {
    "setup_cache_dir": setup_cache_dir,
    "load_from_cache": load_from_cache,
    "save_to_cache": save_to_cache
}

__all__ = [
    'tensor_collate',
    'caching',
    'setup_cache_dir',
    'load_from_cache',
    'save_to_cache'
]