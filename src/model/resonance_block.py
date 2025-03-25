"""
Resonance Block Module.

This module implements a complete resonance block as described in the Semantic Resonance 
Language Model paper, combining resonance attention, feed-forward layers with 
prime-based masks, and residual connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resonance_attention import ResonanceAttention


def create_prime_resonance_mask(dim, primes=[7, 11, 13, 17, 19]):
    """
    Create a structured mask based on prime resonance conditions.
    
    Args:
        dim (int): Dimension of the weight matrix
        primes (List[int]): List of prime numbers for resonance conditions
    
    Returns:
        torch.Tensor: Boolean mask of shape [dim, dim]
    """
    mask = torch.zeros((dim, dim), dtype=torch.bool)
    
    # Set indices to 1 if they pass the prime resonance condition
    for i in range(dim):
        for j in range(dim):
            # Example condition: (i-j) mod p = 0 for some prime p
            if any((i-j) % p == 0 for p in primes):
                mask[i, j] = True
                
    return mask


class MaskedFeedForward(nn.Module):
    """
    Feed-forward layer with prime resonance mask.
    
    The mask ensures structured sparsity guided by prime-based frequency analysis,
    reducing parameter count while maintaining model capacity.
    """
    
    def __init__(self, hidden_dim, ff_dim, primes=[7, 11, 13, 17, 19], dropout=0.1):
        """
        Initialize the masked feed-forward layer.
        
        Args:
            hidden_dim (int): Hidden dimension size
            ff_dim (int): Feed-forward inner dimension size
            primes (List[int]): Primes used for resonance mask
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.primes = primes
        
        # First linear layer with mask
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        
        # Create and register the prime resonance mask
        mask = create_prime_resonance_mask(hidden_dim, primes)
        self.register_buffer('mask', mask)
        
        # Second linear layer
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize the masked weights
        self._apply_mask()
    
    def _apply_mask(self):
        """Apply the prime resonance mask to the weights of the first linear layer."""
        # Get the weight matrix
        weight = self.linear1.weight  # [ff_dim, hidden_dim]
        
        # Create mask of correct size
        mask = create_prime_resonance_mask(min(self.ff_dim, self.hidden_dim), self.primes)
        
        # Expand mask if needed
        if self.ff_dim > mask.size(0) or self.hidden_dim > mask.size(1):
            expanded_mask = torch.zeros((self.ff_dim, self.hidden_dim), dtype=torch.bool, device=weight.device)
            # Copy the mask to the expanded tensor
            h_dim = min(mask.size(0), self.ff_dim)
            w_dim = min(mask.size(1), self.hidden_dim)
            expanded_mask[:h_dim, :w_dim] = mask[:h_dim, :w_dim]
            mask_tensor = expanded_mask
        else:
            # Use a slice of the mask
            mask_tensor = mask[:self.ff_dim, :self.hidden_dim]
        
        # Register the properly sized mask
        self.register_buffer('applied_mask', mask_tensor)
        
        # Apply the mask
        masked_weight = weight * mask_tensor.float()
        
        # Update the weight
        self.linear1.weight.data = masked_weight
    
    def forward(self, x):
        """
        Forward pass through the masked feed-forward layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_dim]
        """
        # Re-apply mask to ensure it's maintained during training
        if self.training:
            weight = self.linear1.weight
            weight.data = weight.data * self.applied_mask.float()
        
        # Forward pass through the feed-forward network
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        return x


class ResonanceBlock(nn.Module):
    """
    Complete Resonance Block combining attention, feed-forward layers, and residual connections.
    
    Each block can be viewed as a resonance iteration unit that refines the representation
    through entropy-guided attention and structured feed-forward transformations.
    """
    
    def __init__(self, hidden_dim, num_heads, ff_dim, primes=[7, 11, 13, 17, 19], 
                 max_iterations=10, epsilon=0.1, dropout=0.1, layer_norm_eps=1e-12):
        """
        Initialize the Resonance Block.
        
        Args:
            hidden_dim (int): Size of the hidden dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Size of the feed-forward inner dimension
            primes (List[int]): Primes used for resonance mask
            max_iterations (int): Maximum number of refinement iterations
            epsilon (float): Entropy threshold for halting
            dropout (float): Dropout probability
            layer_norm_eps (float): Layer normalization epsilon
        """
        super().__init__()
        
        # Resonance Attention
        self.attention = ResonanceAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_iterations=max_iterations,
            epsilon=epsilon,
            dropout=dropout
        )
        
        # Feed-forward with prime resonance mask
        self.feed_forward = MaskedFeedForward(
            hidden_dim=hidden_dim,
            ff_dim=ff_dim,
            primes=primes,
            dropout=dropout
        )
        
        # Layer normalizations
        self.layer_norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attention_mask=None, return_attention=False):
        """
        Forward pass through the resonance block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim]
            attention_mask (torch.Tensor, optional): Attention mask of shape 
                [batch_size, 1, 1, seq_len]
            return_attention (bool): Whether to return attention metadata
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_dim]
            dict: Metadata including entropy, iterations used, and optionally attention weights
        """
        # Residual connection for attention
        residual = x
        
        # Layer normalization before attention
        x_norm = self.layer_norm1(x)
        
        # Apply resonance attention
        attn_output, attn_metadata = self.attention(
            x_norm, 
            attention_mask=attention_mask,
            return_attn_weights=return_attention
        )
        
        # Apply dropout and residual connection
        x = residual + self.dropout(attn_output)
        
        # Residual connection for feed-forward
        residual = x
        
        # Layer normalization before feed-forward
        x_norm = self.layer_norm2(x)
        
        # Apply masked feed-forward
        ff_output = self.feed_forward(x_norm)
        
        # Apply dropout and residual connection
        x = residual + self.dropout(ff_output)
        
        return x, attn_metadata