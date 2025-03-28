"""
Resonance Block for QLLM.

This module provides a transformer block enhanced with quantum resonance
principles, replacing the standard transformer block for improved
contextual understanding and information flow.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union

from src.model.resonance_attention import ResonanceAttention
from src.model.pre_manifest_layer import PreManifestLayer


class ResonanceBlock(nn.Module):
    """
    Transformer block enhanced with quantum resonance principles.
    
    This block extends the standard transformer block by incorporating
    resonance-based attention and pre-manifest transformation, allowing for
    more complex interactions between tokens and improved contextualization.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        prime: int = 7,
        base_dim: int = 32,
        max_iterations: int = 10,
        entropy_threshold: float = 0.2,
        use_pre_manifest: bool = True,
        layer_idx: int = 0
    ):
        """
        Initialize resonance block.
        
        Args:
            hidden_dim: Dimension of hidden states
            num_heads: Number of attention heads
            dropout: Dropout rate
            prime: Prime number for resonance patterns
            base_dim: Base dimension for prime encoding
            max_iterations: Maximum number of iterations for resonance
            entropy_threshold: Entropy threshold for early stopping
            use_pre_manifest: Whether to use pre-manifest transformation
            layer_idx: Index of the layer (used for scaling)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.use_pre_manifest = use_pre_manifest
        
        # Resonance attention
        self.attention = ResonanceAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            prime=prime,
            max_iterations=max_iterations,
            entropy_threshold=entropy_threshold,
            layer_idx=layer_idx
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Pre-manifest layer (optional)
        if use_pre_manifest:
            self.pre_manifest = PreManifestLayer(
                hidden_dim=hidden_dim,
                prime=prime,
                base_dim=base_dim
            )
        else:
            self.pre_manifest = None
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        return_metadata: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, Any]]:
        """
        Forward pass of resonance block.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            output_attentions: Whether to return attention weights
            return_metadata: Whether to return metadata
            
        Returns:
            Updated hidden states with optional attention weights and metadata
        """
        # Apply pre-manifest transformation if enabled
        if self.use_pre_manifest and self.pre_manifest is not None:
            hidden_states = self.pre_manifest(hidden_states)
        
        # Save residual connection
        residual = hidden_states
        
        # Apply layer normalization before attention
        hidden_states = self.norm1(hidden_states)
        
        # Apply resonance attention
        attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_metadata=return_metadata
        )
        
        # Process attention outputs
        if output_attentions and return_metadata:
            attention_output, attention_weights, metadata = attention_outputs
        elif output_attentions:
            attention_output, attention_weights = attention_outputs
        elif return_metadata:
            attention_output, metadata = attention_outputs
        else:
            attention_output = attention_outputs
        
        # Apply residual connection
        hidden_states = residual + self.dropout(attention_output)
        
        # Save residual connection
        residual = hidden_states
        
        # Apply layer normalization before feed-forward
        hidden_states = self.norm2(hidden_states)
        
        # Apply feed-forward network
        hidden_states = self.feed_forward(hidden_states)
        
        # Apply residual connection
        hidden_states = residual + self.dropout(hidden_states)
        
        # Prepare return values
        if output_attentions and return_metadata:
            metadata["layer_idx"] = self.layer_idx
            return hidden_states, attention_weights, metadata
        elif output_attentions:
            return hidden_states, attention_weights
        elif return_metadata:
            metadata["layer_idx"] = self.layer_idx
            return hidden_states, metadata
        else:
            return hidden_states
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get block configuration.
        
        Returns:
            Configuration dictionary
        """
        config = {
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "layer_idx": self.layer_idx,
            "use_pre_manifest": self.use_pre_manifest
        }
        
        # Add attention config
        if hasattr(self.attention, "get_config"):
            config["attention"] = self.attention.get_config()
        
        # Add pre-manifest config
        if self.pre_manifest is not None and hasattr(self.pre_manifest, "get_config"):
            config["pre_manifest"] = self.pre_manifest.get_config()
        
        return config