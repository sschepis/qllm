"""
Base model implementation for QLLM.

This module provides a foundational model class that consolidates common
functionality found across different model implementations to reduce
code duplication and ensure consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union, Callable


class BaseModel(nn.Module):
    """
    Base model class that provides common functionality for all QLLM models.
    
    This class consolidates duplicated code found in multiple model implementations
    including SemanticResonanceModel, EvaluationModel, and other variants.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        tokenizer = None,
        **kwargs
    ):
        """
        Initialize the base model with common parameters.
        
        Args:
            vocab_size: Size of the vocabulary
            hidden_dim: Dimension of hidden layers
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
            tokenizer: Optional tokenizer
            **kwargs: Additional keyword arguments for specific implementations
        """
        super().__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        
        # Store additional configuration as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize common components
        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_dim)
        
        # Create transformer layers
        self.layers = self._create_transformer_layers()
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection for language modeling
        self.output_projection = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie weights between token embeddings and output projection
        self.output_projection.weight = self.token_embeddings.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Extensions support (found in multiple model implementations)
        self.extensions_enabled = False
        self.multimodal_extension = None
        self.memory_extension = None
        self.quantum_extension = None
    
    def _create_transformer_layers(self) -> nn.ModuleList:
        """
        Create transformer layers. Overridable by subclasses.
        
        Returns:
            ModuleList of transformer layers
        """
        return nn.ModuleList([
            TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
    
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
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            labels: Labels for computing loss
            return_dict: Whether to return output as dictionary
            **kwargs: Additional keyword arguments
            
        Returns:
            Model outputs as dictionary or tensor
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
            outputs = {
                "logits": logits,
                "hidden_states": hidden_states
            }
            if loss is not None:
                outputs["loss"] = loss
            return outputs
        else:
            return loss if loss is not None else logits
    
    def generate(
        self,
        input_ids: Union[torch.Tensor, str],
        max_length: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        **kwargs
    ) -> Union[torch.Tensor, str]:
        """
        Generate text based on input_ids.
        
        Args:
            input_ids: Input token ids or text string
            max_length: Maximum length of the generated sequence
            temperature: Temperature for sampling
            do_sample: Whether to sample from the distribution
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            num_return_sequences: Number of sequences to return
            pad_token_id: Token ID to use for padding
            **kwargs: Additional parameters
            
        Returns:
            Generated token ids or text
        """
        device = self.token_embeddings.weight.device
        return_text = False
        
        # Handle string input
        if isinstance(input_ids, str):
            return_text = True
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for text input")
            encoded = self.tokenizer(input_ids, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
        
        # Ensure input_ids is on the correct device
        input_ids = input_ids.to(device)
        batch_size = input_ids.shape[0]
        
        # Initialize generated sequences with input_ids
        generated = input_ids.clone()
        
        # Create attention mask
        attention_mask = torch.ones_like(generated, device=device)
        
        # Continue generating until we reach max_length
        with torch.no_grad():
            for _ in range(max_length - generated.shape[1]):
                # Get model outputs
                outputs = self.forward(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                # Get logits for next token (last token in sequence)
                next_token_logits = outputs["logits"][:, -1, :]
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    # Keep only the top-k values
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    
                    # Create a mask of values to keep
                    filter_mask = torch.zeros_like(next_token_logits, device=device)
                    filter_mask.scatter_(1, top_k_indices, 1.0)
                    
                    # Apply the mask (set remaining values to -inf)
                    next_token_logits = torch.where(
                        filter_mask > 0,
                        next_token_logits,
                        torch.full_like(next_token_logits, float('-inf'))
                    )
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    # Sort logits and corresponding indices
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    
                    # Compute cumulative probabilities
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    
                    # Shift the indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Create a scatter mask and apply it
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = torch.where(
                        indices_to_remove,
                        torch.full_like(next_token_logits, float('-inf')),
                        next_token_logits
                    )
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in generated[i]:
                            if token_id in next_token_logits[i]:
                                next_token_logits[i, token_id] /= repetition_penalty
                
                # Sample from the filtered distribution
                if do_sample:
                    # Apply softmax to convert logits to probabilities
                    probs = torch.softmax(next_token_logits, dim=-1)
                    
                    # Sample from the distribution
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    # Take the token with the highest probability
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Check for EOS token if pad_token_id is provided
                if pad_token_id is not None and (next_tokens == pad_token_id).any():
                    # For tokens that are EOS, replace with pad_token_id
                    # This allows us to keep generating but recognize which sequences are done
                    is_eos = next_tokens == pad_token_id
                    next_tokens = torch.where(
                        is_eos,
                        torch.full_like(next_tokens, pad_token_id),
                        next_tokens
                    )
                    
                    # If all sequences have generated EOS, we can stop
                    if (next_tokens == pad_token_id).all():
                        break
                
                # Append the new token to the sequence
                generated = torch.cat([generated, next_tokens], dim=1)
                
                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=device)
                ], dim=1)
        
        # Decode if returning text
        if return_text and self.tokenizer is not None:
            return self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        return generated
    
    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and its configuration.
        
        Args:
            save_directory: Directory to save the model to
        """
        import os
        import json
        
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save configuration
        config = {
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "max_seq_length": self.max_seq_length,
        }
        
        # Add any additional attributes
        for key, value in self.__dict__.items():
            if isinstance(value, (int, float, str, bool, list, dict)) and key not in config:
                config[key] = value
        
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # Save tokenizer if available
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(save_directory)
    
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
    
    This class is used by the BaseModel and consolidates the common transformer
    block implementations found across the codebase.
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
        # Convert attention mask to the format expected by nn.MultiheadAttention
        if attention_mask is not None:
            # Attention mask has shape [batch_size, seq_length]
            # Convert to [batch_size, seq_length, seq_length] where 1 = attend, 0 = don't attend
            attention_mask = attention_mask.unsqueeze(1).expand(
                -1, attention_mask.shape[1], -1
            )
            # Flip values to match PyTorch convention (True = don't attend, False = attend)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_output, _ = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attn_mask=attention_mask
        )
        hidden_states = residual + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.dropout(self.feed_forward(hidden_states))
        
        return hidden_states