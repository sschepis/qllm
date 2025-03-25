"""
Semantic Resonance Language Model.

This module implements the complete model architecture described in the
Semantic Resonance Language Model paper, integrating all components into
a cohesive next-generation language model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prime_hilbert_encoder import PrimeHilbertEncoder
from .resonance_block import ResonanceBlock
from .homomorphic_wrapper import HomomorphicComputationalWrapper
from .pre_manifest_layer import PreManifestResonanceLayer


class SemanticResonanceModel(nn.Module):
    """
    Complete Semantic Resonance Language Model integrating all components:
    
    1. Prime Hilbert Encoder: Converts tokens and positions into prime-based subspaces
    2. Stack of Resonance Blocks: Processes with iterative, entropy-driven attention
    3. Self-Evolving Memory (HCW): Enables continuous adaptation
    4. Pre-Manifest Resonance Layer: Refines outputs before final distribution
    """
    
    def __init__(self, config):
        """
        Initialize the Semantic Resonance Language Model.
        
        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        
        self.config = config
        
        # Prime Hilbert Encoder
        self.encoder = PrimeHilbertEncoder(
            vocab_size=config.vocab_size,
            primes=config.primes,
            base_dim=config.base_dim,
            max_seq_len=config.max_seq_length
        )
        
        # Stack of Resonance Blocks
        self.layers = nn.ModuleList([
            ResonanceBlock(
                hidden_dim=config.embedding_dim,
                num_heads=config.num_heads,
                ff_dim=config.hidden_dim * 4,  # Standard multiplier for feed-forward
                primes=config.primes,
                max_iterations=config.max_iterations,
                epsilon=config.entropy_threshold,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        # Self-Evolving Memory (HCW)
        if config.enable_hcw:
            self.hcw = HomomorphicComputationalWrapper(
                hidden_dim=config.embedding_dim,
                memory_size=config.memory_size,
                key_dim=config.memory_key_dim,
                dropout=config.dropout
            )
        else:
            self.hcw = None
        
        # Pre-Manifest Resonance Layer
        self.pre_manifest = PreManifestResonanceLayer(
            hidden_dim=config.embedding_dim,
            vocab_size=config.vocab_size,
            max_iterations=config.pre_manifest_iterations,
            epsilon=config.pre_manifest_entropy_threshold,
            dropout=config.dropout
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize weights for the model.
        
        Args:
            module: Module to initialize
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self):
        """
        Get the model's input embeddings.
        
        Returns:
            nn.Module: Input embeddings module
        """
        return self.encoder.base_embedding
    
    def _prepare_attention_mask(self, input_ids, attention_mask=None):
        """
        Prepare attention mask for the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask of shape [batch_size, seq_len]
        
        Returns:
            torch.Tensor: Prepared attention mask
        """
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
        
        # If attention_mask is provided, combine with causal mask
        if attention_mask is not None:
            # Convert from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Combine masks (1.0 for tokens to attend to, 0.0 for tokens to ignore)
            combined_mask = causal_mask * attention_mask
        else:
            combined_mask = causal_mask
        
        # Convert to additive mask (0 for tokens to attend to, large negative for tokens to ignore)
        additive_mask = (1.0 - combined_mask) * -10000.0
        
        return additive_mask
    
    def forward(self, input_ids, attention_mask=None, positions=None, labels=None, return_dict=True):
        """
        Forward pass of the Semantic Resonance Model.
        
        Args:
            input_ids (torch.Tensor): Token IDs of shape [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask of shape [batch_size, seq_len]
            positions (torch.Tensor, optional): Position indices. If None, uses default positions.
            labels (torch.Tensor, optional): Target token IDs for language modeling
            return_dict (bool): Whether to return outputs as a dictionary
        
        Returns:
            Union[torch.Tensor, Dict]: Model outputs, either as logits tensor or as dictionary
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Prepare attention mask
        attn_mask = self._prepare_attention_mask(input_ids, attention_mask)
        
        # Encode inputs using Prime Hilbert Encoder
        hidden_states = self.encoder(input_ids, positions)  # [batch_size, seq_len, embedding_dim]
        
        # Initialize dictionary to collect block metadata
        all_block_metadata = []
        
        # Process through resonance blocks
        for i, layer in enumerate(self.layers):
            # Apply self-evolving memory (HCW) updates if enabled
            if self.hcw is not None and i % 2 == 0:  # Apply HCW every other layer
                # Generate weight deltas based on current context
                hidden_states, hcw_metadata = self.hcw(hidden_states)
                all_block_metadata.append({"layer": i, "type": "hcw", "metadata": hcw_metadata})
            
            # Process through resonance block
            hidden_states, block_metadata = layer(hidden_states, attn_mask)
            all_block_metadata.append({"layer": i, "type": "resonance", "metadata": block_metadata})
        
        # Final pre-manifest resonance layer to compute logits
        logits, final_metadata = self.pre_manifest(hidden_states, attention_mask)
        all_block_metadata.append({"layer": "final", "type": "pre_manifest", "metadata": final_metadata})
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the shifted tensors
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # Compute cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)
            
            # Add entropy regularization if specified
            if hasattr(self.config, 'entropy_regularization_weight') and self.config.entropy_regularization_weight > 0:
                # Get entropy values from block metadata
                entropy_values = []
                for metadata in all_block_metadata:
                    if "entropy" in metadata["metadata"]:
                        entropy = metadata["metadata"]["entropy"]
                        if isinstance(entropy, torch.Tensor):
                            entropy_values.append(entropy.mean())
                
                if entropy_values:
                    # Average entropy across all blocks
                    avg_entropy = torch.stack(entropy_values).mean()
                    # Add regularization term
                    entropy_reg = self.config.entropy_regularization_weight * avg_entropy
                    loss = loss + entropy_reg
        
        if not return_dict:
            return (loss, logits) if loss is not None else logits
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states,
            "metadata": all_block_metadata
        }
    
    def generate(self, input_ids, max_length=20, temperature=1.0, do_sample=True, 
                top_k=50, top_p=0.95, repetition_penalty=1.0, **kwargs):
        """
        Generate text using the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len]
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature
            do_sample (bool): Whether to sample or take the most likely token
            top_k (int): Number of highest probability tokens to keep for top-k filtering
            top_p (float): Cumulative probability for nucleus sampling
            repetition_penalty (float): Penalty for repeating tokens
            
        Returns:
            torch.Tensor: Generated token IDs
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Store current evaluation mode
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            # Initialize generation with input_ids
            generated = input_ids.clone()
            
            # Generate tokens up to max_length
            for _ in range(max_length):
                # Create attention mask for generated tokens
                attention_mask = torch.ones_like(generated)
                
                # Get model predictions
                outputs = self.forward(generated, attention_mask, return_dict=True)
                next_token_logits = outputs["logits"][:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            if token_id in self.config.vocab_size:
                                next_token_logits[i, token_id] /= repetition_penalty
                
                # Filter with top-k
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Filter with top-p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for batch_idx in range(batch_size):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = -float('Inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append next token to generated
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check if all sequences have reached the end
                # Use default EOS token ID (50256 for GPT-2) if not specified in config
                eos_token_id = getattr(self.config, 'eos_token_id', 50256)
                if (next_token == eos_token_id).all():
                    break
        
        # Restore training mode
        self.train(was_training)
        
        return generated
    
    def save_pretrained(self, save_directory):
        """
        Save the model to the specified directory.
        
        Args:
            save_directory (str): Directory to save the model
        """
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "model.pt"))
        
        # Save config
        config_dict = {k: v for k, v in self.config.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        with open(os.path.join(save_directory, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, load_directory):
        """
        Load the model from the specified directory.
        
        Args:
            load_directory (str): Directory to load the model from
            
        Returns:
            SemanticResonanceModel: Loaded model
        """
        import os
        import json
        from dataclasses import fields
        from src.config import ModelConfig
        
        # Load config
        with open(os.path.join(load_directory, "config.json"), 'r') as f:
            config_dict = json.load(f)
        
        # Create config object
        config = ModelConfig()
        
        # Set attributes from loaded config
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Initialize model with loaded config
        model = cls(config)
        
        # Load model weights
        model.load_state_dict(torch.load(os.path.join(load_directory, "model.pt")))
        
        return model