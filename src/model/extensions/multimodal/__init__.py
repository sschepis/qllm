"""
Multimodal Extension Module.

This module provides extensions for processing multiple modalities such as
images, audio, and text in the Semantic Resonance Language Model.

Module Structure:
- Base functionality: base_multimodal_core.py, multimodal_types.py, multimodal_config.py
- Vision functionality: vision_extension_impl.py, vision_encoder.py, multi_resolution_vision.py
- Integration utilities: multimodal_integration.py

The module supports various modalities through a unified extension interface.
"""

# Core multimodal types and base class
from .base_multimodal_core import BaseMultimodalExtension
from .vision_extension_impl import VisionExtension

# Configuration
from .multimodal_config import MultimodalConfig, VisionConfig

# Integration utilities
from .multimodal_integration import ModalityTextIntegration, DynamicFusionModule

# Vision-specific components
from .vision_encoder import VisionEncoder, PrimeProjectionEncoder
from .multi_resolution_vision import MultiResolutionVisionProcessor

# Type definitions and constants
from .multimodal_types import (
    MODALITY_VISION, MODALITY_AUDIO, MODALITY_TEXT, MODALITY_GENERIC,
    INTEGRATION_ADD, INTEGRATION_CONCAT, INTEGRATION_ATTENTION,
    INTEGRATION_CROSS_ATTENTION, INTEGRATION_FILM
)

# Keep the original exports for backward compatibility
__all__ = [
    # Original exports
    'BaseMultimodalExtension',
    'VisionExtension',
    
    # New exports
    'MultimodalConfig',
    'VisionConfig',
    'ModalityTextIntegration',
    'DynamicFusionModule',
    'VisionEncoder',
    'PrimeProjectionEncoder',
    'MultiResolutionVisionProcessor',
    
    # Constants
    'MODALITY_VISION',
    'MODALITY_AUDIO',
    'MODALITY_TEXT',
    'MODALITY_GENERIC',
    'INTEGRATION_ADD',
    'INTEGRATION_CONCAT', 
    'INTEGRATION_ATTENTION',
    'INTEGRATION_CROSS_ATTENTION',
    'INTEGRATION_FILM'
]