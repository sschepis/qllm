"""
Prime Hilbert Encoder for QLLM.

This module provides a specialized encoder that uses prime number theory
and Hilbert space mathematics to create embeddings with unique properties
for quantum-inspired language models.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union


class PrimeHilbertEncoder(nn.Module):
    """
    Encoder that leverages prime number theory and Hilbert spaces.
    
    This encoder creates specialized embeddings based on prime number theory
    and Hilbert space mathematics, which provide useful properties for
    quantum-inspired language models.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        primes: List[int],
        base_dim: int = 32,
        use_complex: bool = False,
        normalize_output: bool = True
    ):
        """
        Initialize the Prime Hilbert Encoder.
        
        Args:
            hidden_dim: Dimension of the hidden state
            primes: List of prime numbers to use for encoding
            base_dim: Base dimension for prime encoding
            use_complex: Whether to use complex numbers
            normalize_output: Whether to normalize the output
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.primes = primes
        self.base_dim = base_dim
        self.use_complex = use_complex
        self.normalize_output = normalize_output
        
        # Create prime basis matrices
        self.register_buffer("prime_basis", self._create_prime_basis())
        
        # Projections for Hilbert space mapping
        if use_complex:
            # For complex numbers, we need separate real and imaginary projections
            self.real_projection = nn.Parameter(torch.randn(hidden_dim, base_dim) * 0.02)
            self.imag_projection = nn.Parameter(torch.randn(hidden_dim, base_dim) * 0.02)
        else:
            # For real numbers, we just need one projection
            self.projection = nn.Parameter(torch.randn(hidden_dim, base_dim) * 0.02)
    
    def _create_prime_basis(self) -> torch.Tensor:
        """
        Create prime basis matrices for encoding.
        
        Returns:
            Tensor containing prime basis matrices
        """
        # Initialize basis tensor
        basis = torch.zeros(len(self.primes), self.hidden_dim)
        
        # Fill basis tensor with prime-based patterns
        for i, prime in enumerate(self.primes):
            # Create pattern based on prime number
            pattern = torch.zeros(self.hidden_dim)
            
            # Fill pattern based on prime residues
            for j in range(self.hidden_dim):
                residue = j % prime
                angle = 2 * math.pi * residue / prime
                pattern[j] = math.sin(angle)
            
            # Normalize pattern
            pattern = F.normalize(pattern, dim=0)
            
            # Store in basis tensor
            basis[i] = pattern
        
        return basis
    
    def forward(self, input_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate prime Hilbert encoding.
        
        Args:
            input_tensor: Optional input tensor to encode (if None, returns mask)
            
        Returns:
            Encoded tensor or prime mask
        """
        # If no input, just return the prime mask
        if input_tensor is None:
            return self._generate_prime_mask()
        
        # Project input to base dimension
        if self.use_complex:
            # Complex projection
            real_proj = torch.matmul(input_tensor, self.real_projection)
            imag_proj = torch.matmul(input_tensor, self.imag_projection)
            
            # Convert to complex domain
            complex_proj = torch.complex(real_proj, imag_proj)
            
            # Apply prime basis in complex domain
            output = torch.matmul(complex_proj, self.prime_basis.transpose(0, 1))
            
            # Convert back to real domain
            output = torch.abs(output)
        else:
            # Real projection
            proj = torch.matmul(input_tensor, self.projection)
            
            # Apply prime basis
            output = torch.matmul(proj, self.prime_basis.transpose(0, 1))
        
        # Normalize if required
        if self.normalize_output:
            output = F.normalize(output, dim=-1)
        
        return output
    
    def _generate_prime_mask(self) -> torch.Tensor:
        """
        Generate a prime-based mask for attention.
        
        Returns:
            Prime mask tensor
        """
        # Generate a mask based on prime patterns
        mask = torch.ones(self.hidden_dim)
        
        # Apply prime patterns
        for i, prime in enumerate(self.primes):
            # Create pattern based on prime
            for j in range(self.hidden_dim):
                # Only activate on positions that are divisible by prime
                if j % prime == 0:
                    mask[j] *= math.sin(math.pi / prime) + 1.0
        
        # Normalize to [0, 1] range
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
        
        return mask
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            "hidden_dim": self.hidden_dim,
            "primes": self.primes,
            "base_dim": self.base_dim,
            "use_complex": self.use_complex,
            "normalize_output": self.normalize_output
        }