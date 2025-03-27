"""
Model configuration for the Quantum Resonance Language Model.

This module defines the model configuration parameters for building
model architectures with the appropriate settings.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    
    # Architecture parameters
    vocab_size: int = 50257  # GPT-2 default
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    max_seq_length: int = 1024
    
    # Quantum resonance parameters
    primes: List[int] = field(default_factory=lambda: [23, 29, 31, 37, 41, 43, 47])
    max_iterations: int = 10
    entropy_threshold: float = 0.01
    phase_factor: float = 0.5
    
    # Extensions configuration
    extensions: Dict[str, Any] = field(default_factory=dict)
    
    # Extra parameters for flexibility
    extra_model_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create from dictionary representation."""
        filtered_data = {
            k: v for k, v in data.items()
            if hasattr(cls, k) and not k.startswith("_")
        }
        return cls(**filtered_data)