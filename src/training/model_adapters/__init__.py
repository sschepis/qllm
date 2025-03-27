"""
Model adapters for the enhanced training system.

This package contains model adapters that bridge between the training system
and specific model architectures, handling model-specific operations.
"""

from src.training.model_adapters.base_adapter import ModelAdapter
from src.training.model_adapters.standard_adapter import StandardModelAdapter
from src.training.model_adapters.dialogue_adapter import DialogueModelAdapter
from src.training.model_adapters.multimodal_adapter import MultimodalModelAdapter

__all__ = [
    'ModelAdapter',
    'StandardModelAdapter',
    'DialogueModelAdapter',
    'MultimodalModelAdapter',
]

# Model adapter registry
MODEL_ADAPTERS = {
    'standard': StandardModelAdapter,
    'dialogue': DialogueModelAdapter,
    'multimodal': MultimodalModelAdapter,
}

def get_model_adapter(model_type, *args, **kwargs):
    """
    Get the appropriate model adapter for the given model type.
    
    Args:
        model_type: Type of model ('standard', 'dialogue', 'multimodal')
        *args: Arguments to pass to the adapter constructor
        **kwargs: Keyword arguments to pass to the adapter constructor
        
    Returns:
        Initialized model adapter instance
        
    Raises:
        ValueError: If the model type is not supported
    """
    if model_type not in MODEL_ADAPTERS:
        raise ValueError(f"Unsupported model type: {model_type}. "
                         f"Available types: {list(MODEL_ADAPTERS.keys())}")
    
    return MODEL_ADAPTERS[model_type](*args, **kwargs)