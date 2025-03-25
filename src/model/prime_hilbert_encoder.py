"""
Prime Hilbert Encoder Module.

This module converts tokens and positions into prime-based subspaces as described in the
Semantic Resonance Language Model paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimeHilbertEncoder(nn.Module):
    """
    Prime Hilbert Encoder that maps tokens and positions into prime-based subspaces.
    
    Each token w and position n is mapped as:
    x_w = ⊕_{i=1}^k Proj^(p_i)(baseEmbed(w)) ⊕ PositionEnc(n, p_i)
    
    Where:
    - ⊕ represents concatenation
    - p_i is a prime number defining a subspace
    - Proj^(p_i) is a projection from the base embedding dimension to p_i
    - PositionEnc(n, p_i) is a prime-based positional encoding
    """
    
    def __init__(self, vocab_size, primes=[7, 11, 13, 17, 19], base_dim=768, max_seq_len=512):
        """
        Initialize the Prime Hilbert Encoder.
        
        Args:
            vocab_size (int): Size of the vocabulary
            primes (List[int]): List of prime numbers for subspace decomposition
            base_dim (int): Dimension of the base embedding
            max_seq_len (int): Maximum sequence length for positional encodings
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.primes = primes
        self.base_dim = base_dim
        self.max_seq_len = max_seq_len
        self.embedding_dim = sum(primes)  # Total dimension after concatenation
        
        # Base embedding for the vocabulary
        self.base_embedding = nn.Embedding(vocab_size, base_dim)
        
        # Projections for each prime subspace
        self.prime_projections = nn.ModuleList([
            nn.Linear(base_dim, p) for p in primes
        ])
        
        # Create prime-based positional encodings
        self.register_buffer('position_encodings', self._create_position_encodings())
    
    def _create_position_encodings(self):
        """
        Create prime-based sinusoidal position encodings.
        
        Returns:
            torch.Tensor: Position encoding tensor of shape [max_seq_len, sum(primes)]
        """
        encodings = []
        
        for prime in self.primes:
            # Create position encoding for this prime
            pe = torch.zeros(self.max_seq_len, prime)
            position = torch.arange(0, self.max_seq_len).unsqueeze(1).float()
            
            # Calculate number of dimensions for sin and cos
            # Ensure we handle odd-sized prime dimensions properly
            dim_half = prime // 2
            odd_dim = prime % 2 == 1
            
            # Create frequency terms based on actual dimensions
            sin_dim = dim_half + (1 if odd_dim else 0)  # Add extra dimension for sin if prime is odd
            cos_dim = dim_half
            
            # Generate frequency terms for sin and cos separately
            sin_div_term = torch.exp(torch.arange(0, sin_dim).float() * (-math.log(10000.0) / prime))
            if cos_dim > 0:
                cos_div_term = torch.exp(torch.arange(0, cos_dim).float() * (-math.log(10000.0) / prime))
            
            # Apply sin for first half (plus extra if odd)
            pe_sin = torch.sin(position * sin_div_term + (position % prime) / prime)
            pe[:, :sin_dim] = pe_sin
            
            # Apply cos for second half if there are dimensions left
            if cos_dim > 0:
                pe_cos = torch.cos(position * cos_div_term + (position % prime) / prime)
                pe[:, sin_dim:] = pe_cos
            
            encodings.append(pe)
        
        # Concatenate all prime-based position encodings
        return torch.cat(encodings, dim=1)
    
    def forward(self, input_ids, positions=None):
        """
        Forward pass of the Prime Hilbert Encoder.
        
        Args:
            input_ids (torch.Tensor): Token IDs of shape [batch_size, seq_len]
            positions (torch.Tensor, optional): Position indices. If None, uses default positions.
        
        Returns:
            torch.Tensor: Encoded representation of shape [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = input_ids.shape
        
        # Get base embeddings
        base_embed = self.base_embedding(input_ids)  # [batch_size, seq_len, base_dim]
        
        # Use default positions if not provided
        if positions is None:
            positions = torch.arange(seq_len, device=input_ids.device).expand(batch_size, seq_len)
        
        # Get position encodings
        pos_encodings = self._get_position_encodings(positions)  # [batch_size, seq_len, embedding_dim]
        
        # Project into prime subspaces and add position encodings
        prime_embeds = []
        start_idx = 0
        
        for i, proj in enumerate(self.prime_projections):
            # Project base embedding to prime subspace
            prime_embed = proj(base_embed)  # [batch_size, seq_len, p_i]
            
            # Get position encoding for this prime
            end_idx = start_idx + self.primes[i]
            prime_pos = pos_encodings[:, :, start_idx:end_idx]
            
            # Add position encoding to projected embedding
            prime_embed = prime_embed + prime_pos
            prime_embeds.append(prime_embed)
            
            start_idx = end_idx
        
        # Concatenate all prime-based embeddings
        return torch.cat(prime_embeds, dim=-1)  # [batch_size, seq_len, embedding_dim]
    
    def _get_position_encodings(self, positions):
        """
        Get position encodings for given positions.
        
        Args:
            positions (torch.Tensor): Position indices of shape [batch_size, seq_len]
        
        Returns:
            torch.Tensor: Position encodings of shape [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = positions.shape
        
        # Ensure positions are within range
        positions = torch.clamp(positions, 0, self.max_seq_len - 1)
        
        # Gather position encodings for each position
        pos_encodings = self.position_encodings[positions]  # [batch_size, seq_len, embedding_dim]
        
        return pos_encodings