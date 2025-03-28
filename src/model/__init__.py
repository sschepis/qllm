"""
Model module for QLLM.

This module provides model implementations and components for the QLLM system,
with refactored code to reduce duplication by leveraging the core BaseModel class.
"""

from src.model.semantic_resonance_model import SemanticResonanceModel
from src.model.semantic_resonance_model_with_extensions import SemanticResonanceModelWithExtensions
from src.model.homomorphic_wrapper import HomomorphicComputationalWrapper
from src.model.prime_hilbert_encoder import PrimeHilbertEncoder
from src.model.resonance_attention import ResonanceAttention
from src.model.resonance_block import ResonanceBlock
from src.model.pre_manifest_layer import PreManifestLayer
from src.model.fixed_autocast import fixed_autocast

__all__ = [
    # Main model implementations
    'SemanticResonanceModel',
    'SemanticResonanceModelWithExtensions',
    
    # Core components
    'HomomorphicComputationalWrapper',
    'PrimeHilbertEncoder',
    'ResonanceAttention',
    'ResonanceBlock',
    'PreManifestLayer',
    
    # Utilities
    'fixed_autocast'
]