"""
Enhanced Semantic Resonance Language Model with Extensions.

This module extends the original Semantic Resonance Model with the extension
framework, supporting multimodal processing, extended memory, and quantum
group symmetries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union

from .semantic_resonance_model import SemanticResonanceModel
from .extensions.extension_config import ExtensionConfig
from .extensions.extension_manager import ExtensionManager
from .extensions.multimodal.vision_extension import VisionExtension
from .extensions.memory.knowledge_graph_extension import KnowledgeGraphExtension
from .extensions.quantum.symmetry_mask_extension import SymmetryMaskExtension


class EnhancedSemanticResonanceModel(SemanticResonanceModel):
    """
    Enhanced Semantic Resonance Language Model with extensions.
    
    This class extends the original Semantic Resonance Model to incorporate
    the extension framework, enabling multimodal processing, extended memory,
    and quantum group symmetries.
    """
    
    def __init__(self, config, extension_config=None):
        """
        Initialize the Enhanced Semantic Resonance Model.
        
        Args:
            config: Configuration object with model parameters
            extension_config (ExtensionConfig, optional): Configuration for extensions
        """
        # Initialize the base model
        super().__init__(config)
        
        # Initialize extension framework
        self.initialize_extensions(extension_config)
    
    def initialize_extensions(self, extension_config=None):
        """
        Initialize the extension framework.
        
        Args:
            extension_config (ExtensionConfig, optional): Configuration for extensions
        """
        # Create default extension config if not provided
        if extension_config is None:
            extension_config = ExtensionConfig()
        
        # Store extension config
        self.extension_config = extension_config
        
        # Create extension manager
        self.extension_manager = ExtensionManager(extension_config, self)
        
        # Register default extensions if enabled
        if extension_config.extensions_enabled:
            self.register_default_extensions()
    
    def register_default_extensions(self):
        """Register the default extensions based on configuration."""
        # Register multimodal extension if enabled
        if self.extension_config.multimodal.enabled:
            vision_extension = VisionExtension(
                name="vision",
                config=self.extension_config.multimodal.__dict__
            )
            self.extension_manager.register_extension(vision_extension)
        
        # Register memory extension if enabled
        if self.extension_config.memory.enabled:
            memory_extension = KnowledgeGraphExtension(
                name="knowledge_graph",
                config=self.extension_config.memory.__dict__
            )
            self.extension_manager.register_extension(memory_extension)
        
        # Register quantum extension if enabled
        if self.extension_config.quantum.enabled:
            quantum_extension = SymmetryMaskExtension(
                name="symmetry_mask",
                config=self.extension_config.quantum.__dict__
            )
            self.extension_manager.register_extension(quantum_extension)
    
    def register_extension(self, extension):
        """
        Register a custom extension with the model.
        
        Args:
            extension: Extension to register
        """
        self.extension_manager.register_extension(extension)
    
    def set_extension_order(self, order):
        """
        Set the execution order for extensions.
        
        Args:
            order: List of (type, name) tuples defining execution order
        """
        self.extension_manager.set_extension_order(order)
    
    def enable_extension(self, ext_type, ext_name):
        """
        Enable a specific extension.
        
        Args:
            ext_type: Type of the extension
            ext_name: Name of the extension
        """
        self.extension_manager.enable_extension(ext_type, ext_name)
    
    def disable_extension(self, ext_type, ext_name):
        """
        Disable a specific extension.
        
        Args:
            ext_type: Type of the extension
            ext_name: Name of the extension
        """
        self.extension_manager.disable_extension(ext_type, ext_name)
    
    def get_extension(self, ext_type, ext_name):
        """
        Get a registered extension by type and name.
        
        Args:
            ext_type: Type of the extension
            ext_name: Name of the extension
            
        Returns:
            The extension if found, None otherwise
        """
        return self.extension_manager.get_extension(ext_type, ext_name)
    
    def forward(self, 
                input_ids, 
                attention_mask=None, 
                positions=None, 
                labels=None, 
                return_dict=True,
                **extension_inputs):
        """
        Forward pass with extension support.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            positions: Position indices. If None, uses default positions.
            labels: Target token IDs for language modeling
            return_dict: Whether to return outputs as a dictionary
            **extension_inputs: Additional inputs for extensions
            
        Returns:
            Model outputs, either as logits tensor or as dictionary
        """
        # Create a dictionary to store model outputs
        model_outputs = {}
        
        # Include extension inputs in model outputs
        for key, value in extension_inputs.items():
            model_outputs[key] = value
        
        # Run the base model's forward pass
        base_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            positions=positions,
            labels=labels,
            return_dict=True
        )
        
        # Add base outputs to model outputs
        for key, value in base_outputs.items():
            model_outputs[key] = value
        
        # Process through extensions if enabled
        if hasattr(self, 'extension_manager') and self.extension_config.extensions_enabled:
            # Get the hidden states
            hidden_states = base_outputs["hidden_states"]
            
            # Process through extensions
            enhanced_hidden_states, extension_outputs = self.extension_manager.process_extensions(
                hidden_states, model_outputs
            )
            
            # Update hidden states and add extension outputs
            model_outputs["hidden_states"] = enhanced_hidden_states
            model_outputs["extension_outputs"] = extension_outputs
            
            # If the hidden states were modified and logits should be updated
            if not torch.equal(enhanced_hidden_states, hidden_states):
                # Recompute logits using the pre-manifest layer
                new_logits, final_metadata = self.pre_manifest(enhanced_hidden_states, attention_mask)
                model_outputs["logits"] = new_logits
                model_outputs["metadata"]["pre_manifest_recomputed"] = final_metadata
                
                # Recompute loss if labels are provided
                if labels is not None:
                    # Shift logits and labels for next token prediction
                    shift_logits = new_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    # Flatten the shifted tensors
                    shift_logits = shift_logits.view(-1, self.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    
                    # Compute cross entropy loss
                    loss_fct = nn.CrossEntropyLoss()
                    new_loss = loss_fct(shift_logits, shift_labels)
                    model_outputs["loss"] = new_loss
        
        # Return appropriate output format
        if not return_dict:
            if "loss" in model_outputs:
                return (model_outputs["loss"], model_outputs["logits"])
            return model_outputs["logits"]
        
        return model_outputs
    
    def generate(self, 
                input_ids, 
                max_length=20, 
                temperature=1.0, 
                do_sample=True,
                top_k=50, 
                top_p=0.95, 
                repetition_penalty=1.0, 
                **extension_inputs):
        """
        Generate text with extension support.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to sample or take the most likely token
            top_k: Number of highest probability tokens for top-k filtering
            top_p: Cumulative probability for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            **extension_inputs: Additional inputs for extensions
            
        Returns:
            Generated token IDs
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
                
                # Get model predictions with extension support
                outputs = self.forward(
                    generated, 
                    attention_mask,
                    return_dict=True,
                    **extension_inputs
                )
                next_token_logits = outputs["logits"][:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            if token_id < self.config.vocab_size:
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
                
                # Update extension inputs if needed
                for ext_type, ext_dict in outputs.get("extension_outputs", {}).items():
                    for ext_name, ext_output in ext_dict.items():
                        # Pass relevant extension outputs as inputs to the next iteration
                        if isinstance(ext_output, dict) and "state" in ext_output:
                            extension_inputs[f"{ext_type}.{ext_name}.state"] = ext_output["state"]
                
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
        Save the model and extensions to the specified directory.
        
        Args:
            save_directory: Directory to save the model
        """
        import os
        import json
        
        # Save base model
        super().save_pretrained(save_directory)
        
        # Save extension configuration
        if hasattr(self, 'extension_config'):
            ext_config_path = os.path.join(save_directory, "extension_config.json")
            with open(ext_config_path, 'w') as f:
                json.dump(self.extension_config.to_dict(), f, indent=2)
        
        # Save extension states if available
        if hasattr(self, 'extension_manager'):
            ext_states_dir = os.path.join(save_directory, "extension_states")
            os.makedirs(ext_states_dir, exist_ok=True)
            self.extension_manager.save_state(os.path.join(ext_states_dir, "extensions.pt"))
    
    @classmethod
    def from_pretrained(cls, load_directory):
        """
        Load the model and extensions from the specified directory.
        
        Args:
            load_directory: Directory to load the model from
            
        Returns:
            EnhancedSemanticResonanceModel: Loaded model
        """
        import os
        import json
        from .extensions.extension_config import ExtensionConfig
        
        # Load base model
        model = super().from_pretrained(load_directory)
        
        # Convert to enhanced model if needed
        if not isinstance(model, cls):
            # Create enhanced model from base model
            enhanced_model = cls(model.config)
            enhanced_model.load_state_dict(model.state_dict())
            model = enhanced_model
        
        # Load extension configuration if available
        ext_config_path = os.path.join(load_directory, "extension_config.json")
        if os.path.exists(ext_config_path):
            with open(ext_config_path, 'r') as f:
                ext_config_dict = json.load(f)
            ext_config = ExtensionConfig.from_dict(ext_config_dict)
            model.initialize_extensions(ext_config)
        
        # Load extension states if available
        ext_states_path = os.path.join(load_directory, "extension_states", "extensions.pt")
        if os.path.exists(ext_states_path) and hasattr(model, 'extension_manager'):
            model.extension_manager.load_state(ext_states_path)
        
        return model

# Alias for backward compatibility
SemanticResonanceModelWithExtensions = EnhancedSemanticResonanceModel