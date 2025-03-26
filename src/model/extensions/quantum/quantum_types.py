"""
Quantum Extension Types Module.

This module provides common type definitions for quantum extensions.
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import torch

# Type aliases
GroupType = str  # "cyclic", "permutation", "orthogonal", "lie"
MaskType = str   # "mod", "prime", "adaptive", "quantum"
EvolutionMethod = str  # "gradient_sensitive", "momentum", "resonance"

# Stats dictionary for tracking quantum operations
QuantumStats = Dict[str, Any]

# Function type for quantum operations
GroupOperationFn = Callable[[torch.Tensor, int], torch.Tensor]

# Type for mask generators
MaskGeneratorFn = Callable[[Tuple[int, ...], float], torch.Tensor]