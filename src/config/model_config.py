"""
Model configuration for QLLM.

This module provides a simplified model configuration class that extends
the ConfigurationBase class from the core module, reducing code duplication
and ensuring consistent behavior.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union

from src.core.configuration import ConfigurationBase


@dataclass
class ModelConfig(ConfigurationBase):
    """
    Model configuration for QLLM.
    
    This class defines parameters for model architecture and behavior,
    extending the ConfigurationBase with model-specific parameters.
    
    Note: This is a simplified version that relies on the shared ConfigurationBase
    class to reduce code duplication.
    """
    
    # Basic architecture parameters
    vocab_size: int = 30000
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    max_seq_length: int = 512
    
    # Advanced options
    use_cache: bool = True
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-12
    activation_function: str = "gelu"
    
    # Quantum resonance parameters
    primes: Optional[List[int]] = None
    base_dim: Optional[int] = None
    max_iterations: Optional[int] = None
    entropy_threshold: Optional[float] = None
    use_prime_mask: bool = False
    enable_hcw: bool = False
    
    # Memory parameters
    memory_size: Optional[int] = None
    memory_key_dim: Optional[int] = None
    
    # Extensions
    extensions: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and initialize derived parameters after instance creation."""
        # Set default primes if not provided
        if self.primes is None:
            # Default to equal distribution across layers
            prime_per_layer = max(8, self.hidden_dim // (self.num_layers * 4))
            self.primes = [prime_per_layer] * self.num_layers
        
        # Ensure num_heads evenly divides hidden_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        
        # Set default base_dim if not provided
        if self.base_dim is None:
            self.base_dim = max(32, self.hidden_dim // 4)
        
        # Set default max_iterations if not provided
        if self.max_iterations is None:
            self.max_iterations = 10
        
        # Set default entropy_threshold if not provided
        if self.entropy_threshold is None:
            self.entropy_threshold = 0.2
    
    def validate(self) -> List[str]:
        """
        Validate model configuration values.
        
        Returns:
            List of validation error messages, empty if valid
        """
        errors = super().validate()
        
        # Validate specific model config parameters
        if self.hidden_dim % self.num_heads != 0:
            errors.append(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        
        if self.vocab_size <= 0:
            errors.append(f"vocab_size must be positive, got {self.vocab_size}")
        
        if self.num_layers <= 0:
            errors.append(f"num_layers must be positive, got {self.num_layers}")
        
        if self.num_heads <= 0:
            errors.append(f"num_heads must be positive, got {self.num_heads}")
        
        if self.dropout < 0 or self.dropout > 1:
            errors.append(f"dropout must be between 0 and 1, got {self.dropout}")
        
        if self.max_seq_length <= 0:
            errors.append(f"max_seq_length must be positive, got {self.max_seq_length}")
        
        # Validate primes if provided
        if self.primes is not None:
            if len(self.primes) != self.num_layers:
                errors.append(
                    f"primes list length ({len(self.primes)}) must match "
                    f"num_layers ({self.num_layers})"
                )
            
            for i, prime in enumerate(self.primes):
                if prime <= 0:
                    errors.append(f"prime[{i}] must be positive, got {prime}")
        
        return errors
    
    def get_head_dim(self) -> int:
        """Get the dimension of each attention head."""
        return self.hidden_dim // self.num_heads
    
    def get_extension_config(self, extension_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific extension.
        
        Args:
            extension_name: Name of the extension
            
        Returns:
            Configuration dictionary for the extension
        """
        if self.extensions is None:
            return {}
        
        return self.extensions.get(extension_name, {})
    
    def has_extension(self, extension_name: str) -> bool:
        """
        Check if an extension is enabled in the model config.
        
        Args:
            extension_name: Name of the extension
            
        Returns:
            True if the extension is enabled
        """
        if self.extensions is None:
            return False
        
        extension_config = self.extensions.get(extension_name, {})
        return extension_config.get("enabled", False)