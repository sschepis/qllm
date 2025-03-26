"""
Quantum Group Symmetries Extension Module.

This module provides extensions for implementing quantum-inspired symmetry
operations and structured masking in the Semantic Resonance Language Model.

Module Structure:
- Base quantum functionality: base_quantum_core.py, quantum_types.py, quantum_config.py
- Group operations: group_operations.py
- Mask generators: mask_generators.py, quantum_patterns.py
- Symmetry mask implementation: symmetry_mask_impl.py
"""

# Base quantum extension classes
from .base_quantum_core import BaseQuantumExtension
from .symmetry_mask_impl import SymmetryMaskExtension

# Type definitions and configuration
from .quantum_types import GroupType, MaskType, EvolutionMethod
from .quantum_config import QuantumConfig, SymmetryMaskConfig

# Group operations
from .group_operations import GroupOperations

# Mask generation
from .mask_generators import MaskGenerators
from .quantum_patterns import QuantumPatternGenerator

# For backward compatibility, maintain old imports
# Importing from the original files for legacy code support
from .base_quantum_extension import BaseQuantumExtension as LegacyBaseQuantumExtension
from .symmetry_mask_extension import SymmetryMaskExtension as LegacySymmetryMaskExtension

# Use the new implementations as the default exports
__all__ = [
    # Core extension classes
    'BaseQuantumExtension',
    'SymmetryMaskExtension',
    
    # Type definitions and configuration
    'GroupType',
    'MaskType',
    'EvolutionMethod',
    'QuantumConfig',
    'SymmetryMaskConfig',
    
    # Group operations
    'GroupOperations',
    
    # Mask generation
    'MaskGenerators',
    'QuantumPatternGenerator'
]