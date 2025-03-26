"""
Base Multimodal Extension Module.

This module defines the base class for multimodal extensions in the
Semantic Resonance Language Model.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import abc

import torch
import torch.nn as nn

from ..base_extension import BaseExtension


class BaseMultimodalExtension(BaseExtension):
    """
    Base class for all multimodal extensions.
    
    This class extends the BaseExtension to provide common functionality
    specific to multimodal processing, such as handling different input types
    and cross-modal attention.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the multimodal extension.
        
        Args:
            name (str): Unique name for this extension instance
            config (Dict[str, Any]): Configuration dictionary for the extension
        """
        super().__init__(name, config)
        
        # Register modality information
        self.modality_type = config.get("modality_type", "generic")
        self.modality_embedding_dim = config.get("modality_embedding_dim", 768)
        self.prime_mapping = config.get("prime_mapping", [])
        
        # For tracking multimodal inputs
        self.supports_multiple_inputs = config.get("supports_multiple_inputs", False)
        self.input_embedding_cache = {}
    
    def get_extension_type(self) -> str:
        """
        Get the type of this extension.
        
        Returns:
            str: Extension type
        """
        return "multimodal"
    
    def get_modality_type(self) -> str:
        """
        Get the modality type for this extension.
        
        Returns:
            str: Modality type (e.g., "vision", "audio")
        """
        return self.modality_type
    
    @abc.abstractmethod
    def encode_modality(self, modality_input: Any) -> torch.Tensor:
        """
        Encode the modality-specific input into the model's embedding space.
        
        Args:
            modality_input (Any): Modality-specific input (e.g., image tensor, audio tensor)
            
        Returns:
            torch.Tensor: Encoded representation
        """
        raise NotImplementedError
    
    def align_with_text_embedding(self, 
                                 modality_embedding: torch.Tensor, 
                                 text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Align the modality embedding with the text embedding space.
        
        Args:
            modality_embedding (torch.Tensor): Embedding from the modality encoder
            text_embedding (torch.Tensor): Text embedding to align with
            
        Returns:
            torch.Tensor: Aligned modality embedding
        """
        # Default implementation just ensures shape compatibility
        # More sophisticated alignment would be implemented in subclasses
        
        # Check if dimensions match
        if modality_embedding.shape[-1] != text_embedding.shape[-1]:
            # Simple projection to match dimensions
            if not hasattr(self, 'alignment_projection'):
                self.alignment_projection = nn.Linear(
                    modality_embedding.shape[-1], 
                    text_embedding.shape[-1]
                ).to(modality_embedding.device)
            
            aligned_embedding = self.alignment_projection(modality_embedding)
        else:
            aligned_embedding = modality_embedding
        
        return aligned_embedding
    
    def clear_cache(self) -> None:
        """Clear the input embedding cache."""
        self.input_embedding_cache = {}
    
    def integrate_with_text(self, 
                           modality_embedding: torch.Tensor, 
                           text_embedding: torch.Tensor,
                           integration_method: str = "concat") -> torch.Tensor:
        """
        Integrate modality embedding with text embedding.
        
        Args:
            modality_embedding (torch.Tensor): Embedding from the modality encoder
            text_embedding (torch.Tensor): Text embedding to integrate with
            integration_method (str): Integration method ("concat", "add", "attention")
            
        Returns:
            torch.Tensor: Integrated embedding
        """
        # Align dimensions first
        aligned_modality = self.align_with_text_embedding(modality_embedding, text_embedding)
        
        # Different integration methods
        if integration_method == "add":
            # Simple addition
            return text_embedding + aligned_modality
        
        elif integration_method == "concat":
            # Concatenate along sequence dimension (assume modality is a single "token")
            if len(aligned_modality.shape) == 3:  # [batch, seq, dim]
                return torch.cat([aligned_modality, text_embedding], dim=1)
            else:  # [batch, dim]
                aligned_modality = aligned_modality.unsqueeze(1)  # Add sequence dimension
                return torch.cat([aligned_modality, text_embedding], dim=1)
        
        elif integration_method == "attention":
            # Use cross-attention to integrate
            if not hasattr(self, 'cross_attention'):
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=text_embedding.shape[-1],
                    num_heads=8,
                    batch_first=True
                ).to(text_embedding.device)
            
            # Reshape if needed
            if len(aligned_modality.shape) == 2:  # [batch, dim]
                aligned_modality = aligned_modality.unsqueeze(1)  # Add sequence dimension
            
            # Cross-attention: text attends to modality
            integrated, _ = self.cross_attention(
                query=text_embedding,
                key=aligned_modality,
                value=aligned_modality
            )
            
            return integrated
        
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")
    
    def forward(self, 
               x: torch.Tensor,
               model_outputs: Optional[Dict[str, Any]] = None, 
               extension_outputs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass of the multimodal extension.
        
        Args:
            x (torch.Tensor): Input tensor (typically text embedding)
            model_outputs (Dict[str, Any], optional): Outputs from the main model
            extension_outputs (Dict[str, Any], optional): Outputs from other extensions
            
        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Modified tensor and extension metadata
        """
        # Default implementation for when no modality input is provided
        # Subclasses should override for actual modality processing
        metadata = {
            "modality_type": self.modality_type,
            "modality_embedding_dim": self.modality_embedding_dim,
            "has_modality_input": False
        }
        
        return x, metadata