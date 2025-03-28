"""
Data module for QLLM.

This module provides datasets, dataloaders, and utilities for handling
data in the QLLM system. It has been refactored to reduce code duplication
and improve maintainability through a hierarchical class structure.
"""

# Import and expose base classes
from src.data.base import BaseDataset, BaseLoader, BaseProcessor

# Import and expose dataset implementations
from src.data.datasets import (
    TextDataset,
    DialogueDataset,
    WikitextDataset,
    FunctionCallingDataset,
    MultimodalDataset
)

# Import and expose loader implementations
from src.data.loaders import (
    WikitextLoader,
    DailyDialogLoader
    # CustomLoader,  # Not implemented yet
    # DummyLoader,   # Not implemented yet
    # RemoteLoader   # Not implemented yet
)

# Import and expose processor implementations
from src.data.processors import (
    TextProcessor
    # Tokenization,    # Not implemented yet
    # Augmentation,    # Not implemented yet
    # Normalization    # Not implemented yet
)

# Import and expose utility functions
from src.data.utils import (
    batch_utils,
    # dataloader_utils,  # Not implemented yet
    tensor_collate,
    # sampling,          # Not implemented yet
    caching
)

# Import and expose dataloader factory functions
from src.data.dataloaders import (
    create_text_dataloader,
    create_dialogue_dataloader,
    create_wikitext_dataloader,
    create_function_calling_dataloader,
    create_multimodal_dataloader,
    create_dataloader_from_config
)

__all__ = [
    # Base classes
    'BaseDataset',
    'BaseLoader',
    'BaseProcessor',
    
    # Datasets
    'TextDataset',
    'DialogueDataset',
    'WikitextDataset',
    'FunctionCallingDataset',
    'MultimodalDataset',
    
    # Loaders
    'WikitextLoader',
    'DailyDialogLoader',
    # 'CustomLoader',  # Not implemented yet
    # 'DummyLoader',   # Not implemented yet
    # 'RemoteLoader',  # Not implemented yet
    
    # Processors
    'TextProcessor',
    # 'Tokenization',    # Not implemented yet
    # 'Augmentation',    # Not implemented yet
    # 'Normalization',   # Not implemented yet
    
    # Dataloader factory functions
    'create_text_dataloader',
    'create_dialogue_dataloader',
    'create_wikitext_dataloader',
    'create_function_calling_dataloader',
    'create_multimodal_dataloader',
    'create_dataloader_from_config'
]