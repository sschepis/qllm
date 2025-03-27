"""
Semantic Resonance Model implementation for QLLM.

This module defines the core language model with quantum resonance
principles incorporated into its architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union

from src.config.model_config import ModelConfig


class SemanticResonanceModel(nn.Module):
    """
    Semantic Resonance Language Model.
    
    This model implements a transformer-based architecture with
    quantum resonance principles for enhanced semantic understanding.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        # Store configuration
        self.config = config
        
        # Setup dimensions
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_dim
        )
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(
            config.max_seq_length, config.hidden_dim
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(
            config.hidden_dim, config.vocab_size, bias=False
        )
        
        # Tie weights with token embeddings
        self.output_projection.weight = self.token_embeddings.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Extensions support
        self.extensions_enabled = False
        if hasattr(config, "extensions"):
            extensions = config.extensions
            if isinstance(extensions, dict) and extensions.get("extensions_enabled", False):
                self.extensions_enabled = True
                
                # Set up extensions
                self._setup_extensions(extensions)
    
    def _init_weights(self, module):
        """Initialize weights for transformer components."""
        if isinstance(module, nn.Linear):
            # Initialize linear layers
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Initialize embeddings
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Initialize layer normalization
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _setup_extensions(self, extensions: Dict[str, Any]):
        """
        Set up model extensions.
        
        Args:
            extensions: Extensions configuration
        """
        # Memory extension
        if extensions.get("enable_memory", False):
            self._setup_memory_extension(extensions.get("memory_config", {}))
        
        # Multimodal extension
        if extensions.get("enable_multimodal", False):
            self._setup_multimodal_extension(extensions.get("multimodal_config", {}))
        
        # Quantum extension
        if extensions.get("enable_quantum", False):
            self._setup_quantum_extension(extensions.get("quantum_config", {}))
    
    def _setup_memory_extension(self, memory_config: Dict[str, Any]):
        """
        Set up memory extension.
        
        Args:
            memory_config: Memory extension configuration
        """
        # This would normally set up the memory extension
        # For this simplified implementation, we just log that it's enabled
        print("Memory extension enabled with config:", memory_config)
    
    def _setup_multimodal_extension(self, multimodal_config: Dict[str, Any]):
        """
        Set up multimodal extension.
        
        Args:
            multimodal_config: Multimodal extension configuration
        """
        # This would normally set up the multimodal extension
        # For this simplified implementation, we just log that it's enabled
        print("Multimodal extension enabled with config:", multimodal_config)
    
    def _setup_quantum_extension(self, quantum_config: Dict[str, Any]):
        """
        Set up quantum extension.
        
        Args:
            quantum_config: Quantum extension configuration
        """
        # This would normally set up the quantum extension
        # For this simplified implementation, we just log that it's enabled
        print("Quantum extension enabled with config:", quantum_config)
    
    def create_position_ids(self, input_shape: torch.Size, device: torch.device) -> torch.Tensor:
        """
        Create position IDs for the model.
        
        Args:
            input_shape: Shape of input tensor
            device: Device to create tensor on
            
        Returns:
            Position IDs tensor
        """
        batch_size, seq_length = input_shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        return position_ids
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            labels: Labels for computing loss
            return_dict: Whether to return output as dictionary
            
        Returns:
            Model outputs
        """
        # Get input dimensions
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = self.create_position_ids(input_ids.shape, device)
        
        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Apply final layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Compute logits
        logits = self.output_projection(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )
        
        # Return outputs
        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": hidden_states
            }
        else:
            return loss if loss is not None else logits
    
    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """
        Resize token embeddings.
        
        Args:
            new_vocab_size: New vocabulary size
        """
        old_embeddings = self.token_embeddings
        self.token_embeddings = nn.Embedding(
            new_vocab_size, self.hidden_dim
        )
        self.token_embeddings.to(old_embeddings.weight.device)
        
        # Copy weights for existing tokens
        if new_vocab_size > self.vocab_size:
            self.token_embeddings.weight.data[:self.vocab_size] = old_embeddings.weight.data
        else:
            self.token_embeddings.weight.data = old_embeddings.weight.data[:new_vocab_size]
        
        # Update output projection
        old_projection = self.output_projection
        self.output_projection = nn.Linear(
            self.hidden_dim, new_vocab_size, bias=False
        )
        self.output_projection.to(old_projection.weight.device)
        
        # Copy weights for existing tokens
        if new_vocab_size > self.vocab_size:
            self.output_projection.weight.data[:self.vocab_size] = old_projection.weight.data
        else:
            self.output_projection.weight.data = old_projection.weight.data[:new_vocab_size]
        
        # Tie weights
        self.output_projection.weight = self.token_embeddings.weight
        
        # Update vocab size
        self.vocab_size = new_vocab_size


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward layers.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
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
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of transformer block.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            
        Returns:
            Updated hidden states
        """
        # Convert mask format for nn.MultiheadAttention
        if attention_mask is not None:
            # Create attention mask (1.0 for tokens to attend to, 0.0 for tokens to ignore)
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask) * -10000.0
            extended_mask = extended_mask.to(dtype=hidden_states.dtype)
        else:
            extended_mask = None
        
        # Apply self-attention with residual connection
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        # Apply self-attention
        attn_output, _ = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=(attention_mask == 0) if attention_mask is not None else None,
            need_weights=False
        )
        hidden_states = residual + self.dropout(attn_output)
        
        # Apply feed-forward network with residual connection
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.feed_forward(hidden_states)
        
        return hidden_states