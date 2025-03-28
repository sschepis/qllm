"""
Model adapters for the enhanced training system.

This package provides adapters for different model types, handling model-specific
operations like initialization, batch preparation, and loss computation.
"""

import logging
from typing import Any, Optional, Dict, Union, Type

import torch
import torch.nn as nn

from src.training.model_adapters.base_adapter import ModelAdapter
from src.training.model_adapters.standard_adapter import StandardModelAdapter

# Import specialized adapters - will be implemented in separate files
try:
    from src.training.model_adapters.dialogue_adapter import DialogueModelAdapter
except ImportError:
    class DialogueModelAdapter(ModelAdapter):
        """Placeholder for DialogueModelAdapter."""
        pass

try:
    from src.training.model_adapters.multimodal_adapter import MultimodalModelAdapter
except ImportError:
    class MultimodalModelAdapter(ModelAdapter):
        """Placeholder for MultimodalModelAdapter."""
        pass


def get_model_adapter(
    model_type: str,
    model_config: Any,
    training_config: Any,
    device: torch.device,
    logger: Optional[logging.Logger] = None
) -> ModelAdapter:
    """
    Create a model adapter based on model type.
    
    Args:
        model_type: Type of model ('standard', 'dialogue', 'multimodal')
        model_config: Model configuration
        training_config: Training configuration
        device: Device to use
        logger: Logger instance
        
    Returns:
        Initialized model adapter
        
    Raises:
        ValueError: If model_type is not supported
    """
    # Create logger if not provided
    if logger is None:
        logger = logging.getLogger("quantum_resonance")
    
    # Map model type to adapter class
    adapter_map = {
        "standard": StandardModelAdapter,
        "dialogue": DialogueModelAdapter,
        "multimodal": MultimodalModelAdapter
    }
    
    # Normalize model type
    model_type = model_type.lower()
    
    # Check if model type is supported
    if model_type not in adapter_map:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: {', '.join(adapter_map.keys())}"
        )
    
    # Get adapter class
    adapter_class = adapter_map[model_type]
    
    # Create adapter instance
    logger.info(f"Creating model adapter for model type: {model_type}")
    return adapter_class(
        model_config=model_config,
        training_config=training_config,
        device=device,
        logger=logger
    )


__all__ = [
    'ModelAdapter',
    'StandardModelAdapter',
    'DialogueModelAdapter',
    'MultimodalModelAdapter',
    'get_model_adapter',
]