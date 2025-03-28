"""
Semantic Resonance Model implementation for QLLM.

This module provides the core language model with quantum resonance
principles incorporated into its architecture. It has been refactored
to extend the BaseModel class to reduce code duplication.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union

from src.core.base_model import BaseModel
from src.config.model_config import ModelConfig
from src.model.resonance_block import ResonanceBlock
from src.model.prime_hilbert_encoder import PrimeHilbertEncoder


class SemanticResonanceModel(BaseModel):
    """
    Semantic Resonance Language Model.
    
    This model implements a transformer-based architecture with
    quantum resonance principles for enhanced semantic understanding.
    It extends the BaseModel class to reduce code duplication.
    """
    
    def __init__(self, config: Union[ModelConfig, Dict[str, Any]]):
        """
        Initialize the semantic resonance model.
        
        Args:
            config: Model configuration as ModelConfig object or dictionary
        """
        # Convert dict to ModelConfig if needed
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
            
        # Call parent constructor with basic parameters
        super().__init__(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_seq_length=config.max_seq_length
        )
        
        # Store full configuration
        self.config = config
        
        # Set quantum resonance specific attributes
        self.primes = config.primes
        self.base_dim = config.base_dim
        self.max_iterations = config.max_iterations
        self.entropy_threshold = config.entropy_threshold
        self.use_prime_mask = config.use_prime_mask
        self.enable_hcw = config.enable_hcw
        
        # Override the transformer layers with resonance blocks
        self.layers = self._create_resonance_layers()
        
        # Initialize prime Hilbert encoder if using prime mask
        if self.use_prime_mask:
            self.prime_encoder = PrimeHilbertEncoder(
                hidden_dim=self.hidden_dim,
                primes=self.primes,
                base_dim=self.base_dim
            )
        else:
            self.prime_encoder = None
    
    def _create_resonance_layers(self) -> nn.ModuleList:
        """
        Create resonance transformer layers instead of standard transformer blocks.
        
        Returns:
            ModuleList of resonance transformer layers
        """
        return nn.ModuleList([
            ResonanceBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                prime=self.primes[i] if i < len(self.primes) else 8,
                base_dim=self.base_dim,
                max_iterations=self.max_iterations,
                entropy_threshold=self.entropy_threshold,
                layer_idx=i
            )
            for i in range(self.num_layers)
        ])
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_metadata: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the semantic resonance model.
        
        This overrides BaseModel's forward to add quantum resonance features.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            labels: Labels for computing loss
            return_dict: Whether to return output as dictionary
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_metadata: Whether to return metadata from layers
            
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
        
        # Apply prime mask if enabled
        if self.use_prime_mask and self.prime_encoder is not None:
            prime_mask = self.prime_encoder()
            hidden_states = hidden_states * prime_mask
        
        # Initialize lists for storing outputs if needed
        all_hidden_states = [hidden_states] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        metadata = [] if return_metadata else None
        
        # Apply resonance layers
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                return_metadata=return_metadata
            )
            
            # Extract outputs
            if return_metadata:
                hidden_states, layer_metadata = layer_outputs
                if metadata is not None:
                    layer_metadata["layer"] = i
                    metadata.append(layer_metadata)
            elif output_attentions:
                hidden_states, attention_weights = layer_outputs
                if all_attentions is not None:
                    all_attentions.append(attention_weights)
            else:
                hidden_states = layer_outputs
            
            # Save hidden states if needed
            if output_hidden_states and all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
        
        # Apply final layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Add final hidden states if needed
        if output_hidden_states and all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        
        # Add final layer metadata if needed
        if return_metadata and metadata is not None:
            metadata.append({
                "layer": "final",
                "hidden_states_norm": torch.norm(hidden_states, dim=-1).mean().item()
            })
        
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
                "hidden_states": hidden_states,
            }
            
            if loss is not None:
                outputs["loss"] = loss
                
            if output_hidden_states and all_hidden_states is not None:
                outputs["all_hidden_states"] = all_hidden_states
                
            if output_attentions and all_attentions is not None:
                outputs["attentions"] = all_attentions
                
            if return_metadata and metadata is not None:
                outputs["metadata"] = metadata
                
            return outputs
        else:
            return loss if loss is not None else logits
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the model configuration.
        
        Returns:
            Model configuration dictionary
        """
        if hasattr(self.config, "to_dict"):
            return self.config.to_dict()
        elif isinstance(self.config, dict):
            return self.config.copy()
        else:
            # Fall back to manually constructed config
            return {
                "vocab_size": self.vocab_size,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "max_seq_length": self.max_seq_length,
                "primes": self.primes,
                "base_dim": self.base_dim,
                "max_iterations": self.max_iterations,
                "entropy_threshold": self.entropy_threshold,
                "use_prime_mask": self.use_prime_mask,
                "enable_hcw": self.enable_hcw
            }
    
    @classmethod
    def from_pretrained(cls, model_path: str, device: Optional[torch.device] = None) -> 'SemanticResonanceModel':
        """
        Load a model from a pretrained checkpoint.
        
        Args:
            model_path: Path to the checkpoint directory or file
            device: Device to load the model to
            
        Returns:
            Loaded model instance
        """
        import os
        import json
        
        # Determine if path is a directory or file
        if os.path.isdir(model_path):
            # Try to load config.json
            config_path = os.path.join(model_path, "config.json")
            model_path = os.path.join(model_path, "pytorch_model.bin")
            
            # Load config
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
            else:
                raise FileNotFoundError(f"Config file not found: {config_path}")
        else:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location="cpu")
            
            # Extract config
            if "model_config" in checkpoint:
                config = checkpoint["model_config"]
            elif "config" in checkpoint:
                config = checkpoint["config"]
            else:
                raise ValueError(f"No config found in checkpoint: {model_path}")
        
        # Create model
        model = cls(config)
        
        # Move to device if specified
        if device is not None:
            model.to(device)
        
        # Load state dict
        if os.path.isfile(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            
            # Extract model state dict if wrapped in a dictionary
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            
            # Load state dict
            model.load_state_dict(state_dict)
        
        return model