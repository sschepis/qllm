"""
Multimodal Extension Types Module.

This module provides common type definitions for multimodal extensions.
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import torch

# Type aliases
ModalityConfig = Dict[str, Any]
ModalityInput = Union[torch.Tensor, Dict[str, Any]]
IntegrationMethod = str  # "add", "concat", "attention", "cross_attention", "film"
ModalityOutput = Tuple[torch.Tensor, Dict[str, Any]]

# Modality types
MODALITY_VISION = "vision"
MODALITY_AUDIO = "audio"
MODALITY_TEXT = "text"
MODALITY_GENERIC = "generic"

# Integration methods
INTEGRATION_ADD = "add"
INTEGRATION_CONCAT = "concat"
INTEGRATION_ATTENTION = "attention"
INTEGRATION_CROSS_ATTENTION = "cross_attention"
INTEGRATION_FILM = "film"