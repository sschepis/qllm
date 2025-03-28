"""
Resonance Attention mechanism for QLLM.

This module provides a specialized attention mechanism that incorporates
quantum resonance principles into the standard attention computation,
enabling more dynamic information flow and contextual understanding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union


class ResonanceAttention(nn.Module):
    """
    Attention mechanism enhanced with quantum resonance principles.
    
    This attention mechanism extends standard multi-head attention with
    quantum-inspired computations that allow for more complex interactions
    between tokens based on resonance patterns.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        prime: int = 7,
        max_iterations: int = 10,
        entropy_threshold: float = 0.2,
        layer_idx: int = 0
    ):
        """
        Initialize resonance attention.
        
        Args:
            hidden_dim: Dimension of hidden states
            num_heads: Number of attention heads
            dropout: Dropout rate
            prime: Prime number for resonance patterns
            max_iterations: Maximum number of iterations for resonance
            entropy_threshold: Entropy threshold for early stopping
            layer_idx: Index of the layer (used for scaling)
        """
        super().__init__()
        
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        
        # Configuration
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.prime = prime
        self.max_iterations = max_iterations
        self.entropy_threshold = entropy_threshold
        self.layer_idx = layer_idx
        
        # Initialize linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize resonance components
        self.resonance_gate = nn.Linear(hidden_dim, hidden_dim)
        self.resonance_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Prime-based scaling factors
        self.register_buffer(
            "prime_scale", 
            torch.tensor(1.0 / math.sqrt(prime * self.head_dim))
        )
        
        # Resonance factors
        self.register_buffer(
            "resonance_base",
            self._create_resonance_base()
        )
    
    def _create_resonance_base(self) -> torch.Tensor:
        """
        Create resonance base factors.
        
        Returns:
            Resonance base tensor
        """
        # Create resonance pattern based on prime number
        resonance_base = torch.zeros(self.num_heads, self.head_dim)
        
        for h in range(self.num_heads):
            for d in range(self.head_dim):
                # Create unique pattern for each head based on prime factors
                angle = 2 * math.pi * ((h + 1) * (d + 1) % self.prime) / self.prime
                resonance_base[h, d] = 0.5 + 0.5 * math.sin(angle)
        
        return resonance_base
    
    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split tensor into multiple attention heads.
        
        Args:
            tensor: Input tensor [batch_size, seq_len, hidden_dim]
            
        Returns:
            Tensor with separated heads [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = tensor.shape
        
        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        return tensor.transpose(1, 2)
    
    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Merge attention heads back into a single tensor.
        
        Args:
            tensor: Input tensor [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            Merged tensor [batch_size, seq_len, hidden_dim]
        """
        batch_size, _, seq_len, _ = tensor.shape
        
        # Transpose to [batch_size, seq_len, num_heads, head_dim]
        tensor = tensor.transpose(1, 2)
        
        # Reshape to [batch_size, seq_len, hidden_dim]
        return tensor.reshape(batch_size, seq_len, self.hidden_dim)
    
    def _compute_entropy(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of attention weights.
        
        Args:
            attn_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
            
        Returns:
            Entropy tensor [batch_size, num_heads]
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-6
        
        # Compute entropy for each head
        entropy = -torch.sum(
            attn_weights * torch.log(attn_weights + eps),
            dim=-1
        ).mean(dim=-1)  # Average over sequence length
        
        return entropy
    
    def _compute_resonance(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        iteration: int
    ) -> torch.Tensor:
        """
        Compute resonance-modified values.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len, head_dim]
            value: Value tensor [batch_size, num_heads, seq_len, head_dim]
            iteration: Current iteration number
            
        Returns:
            Resonance-modified value tensor
        """
        # Extract dimensions
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Apply resonance gate based on iteration
        iter_factor = torch.sigmoid(torch.tensor(iteration / self.max_iterations, device=query.device))
        
        # Instead of sequence-dependent resonance factors, use a simple scalar
        # This avoids dimension mismatches entirely
        resonance_scale = 0.5 + 0.5 * iter_factor
        
        # Apply a uniform scaling factor to the values
        # No broadcasting issues since we're multiplying by a scalar
        resonance_values = value * resonance_scale
        
        return resonance_values
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        return_metadata: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, Any]]:
        """
        Forward pass of resonance attention.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            output_attentions: Whether to return attention weights
            return_metadata: Whether to return metadata
            
        Returns:
            Attention output with optional attention weights and metadata
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(hidden_states)  # [batch_size, seq_len, hidden_dim]
        k = self.k_proj(hidden_states)  # [batch_size, seq_len, hidden_dim]
        v = self.v_proj(hidden_states)  # [batch_size, seq_len, hidden_dim]
        
        # Split heads
        q = self._split_heads(q)  # [batch_size, num_heads, seq_len, head_dim]
        k = self._split_heads(k)  # [batch_size, num_heads, seq_len, head_dim]
        v = self._split_heads(v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Scale queries with prime-based scaling
        q = q * self.prime_scale
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask shape to [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # Apply mask
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0,
                float('-inf')
            )
        
        # Initialize metadata if needed
        metadata = {"iterations": 0, "entropy": 0.0} if return_metadata else None
        
        # Start with standard attention
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        
        # Apply resonance iteratively
        current_output = torch.matmul(attention_weights, v)
        iterations_performed = 1
        
        # Iterative refinement with resonance
        for i in range(1, self.max_iterations):
            # Compute entropy of attention weights
            entropy = self._compute_entropy(attention_weights)
            avg_entropy = entropy.mean().item()
            
            # Early stopping based on entropy threshold
            if avg_entropy < self.entropy_threshold:
                break
            
            # Apply resonance to refine values
            resonance_v = self._compute_resonance(q, k, v, i)
            
            # Compute refined attention output
            refined_output = torch.matmul(attention_weights, resonance_v)
            
            # Blend with current output
            blend_factor = torch.sigmoid(torch.tensor(i / self.max_iterations))
            current_output = (1 - blend_factor) * current_output + blend_factor * refined_output
            
            iterations_performed = i + 1
        
        # Record metadata if requested
        if return_metadata:
            metadata["iterations"] = iterations_performed
            metadata["entropy"] = self._compute_entropy(attention_weights).mean().item()
        
        # Merge heads
        attention_output = self._merge_heads(current_output)
        
        # Apply output projection
        attention_output = self.o_proj(attention_output)
        attention_output = self.output_dropout(attention_output)
        
        # Prepare return values
        if output_attentions and return_metadata:
            return attention_output, attention_weights, metadata
        elif output_attentions:
            return attention_output, attention_weights
        elif return_metadata:
            return attention_output, metadata
        else:
            return attention_output