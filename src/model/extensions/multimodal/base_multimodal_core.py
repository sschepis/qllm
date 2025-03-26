"""
Base Multimodal Core Module.

This module defines the base class for multimodal extensions in the
Semantic Resonance Language Model.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import abc

import torch
import torch.nn as nn

from ..base_extension import BaseExtension
from .multimodal_config import MultimodalConfig
from .multimodal_integration import ModalityTextIntegration


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
            name: Unique name for this extension instance
            config: Configuration dictionary for the extension
        """
        super().__init__(name, config)
        
        # Create multimodal configuration
        self.config = MultimodalConfig.from_dict(config)
        
        # Register modality information
        self.modality_type = self.config.modality_type
        self.modality_embedding_dim = self.config.modality_embedding_dim
        self.prime_mapping = self.config.prime_mapping
        
        # For tracking multimodal inputs
        self.supports_multiple_inputs = self.config.supports_multiple_inputs
        self.input_embedding_cache = {}
        
        # Create text integration module
        self.text_integration = ModalityTextIntegration(
            embedding_dim=self.modality_embedding_dim
        )
    
    def get_extension_type(self) -> str:
        """
        Get the type of this extension.
        
        Returns:
            Extension type
        """
        return "multimodal"
    
    def get_modality_type(self) -> str:
        """
        Get the modality type for this extension.
        
        Returns:
            Modality type (e.g., "vision", "audio")
        """
        return self.modality_type
    
    @abc.abstractmethod
    def encode_modality(self, modality_input: Any) -> torch.Tensor:
        """
        Encode the modality-specific input into the model's embedding space.
        
        Args:
            modality_input: Modality-specific input (e.g., image tensor, audio tensor)
            
        Returns:
            Encoded representation
        """
        raise NotImplementedError
    
    def align_with_text_embedding(self, 
                                 modality_embedding: torch.Tensor, 
                                 text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Align the modality embedding with the text embedding space.
        
        Args:
            modality_embedding: Embedding from the modality encoder
            text_embedding: Text embedding to align with
            
        Returns:
            Aligned modality embedding
        """
        return self.text_integration.align_dimensions(modality_embedding, text_embedding)
    
    def clear_cache(self) -> None:
        """Clear the input embedding cache."""
        self.input_embedding_cache = {}
    
    def integrate_with_text(self, 
                          modality_embedding: torch.Tensor, 
                          text_embedding: torch.Tensor,
                          integration_method: str = "attention") -> torch.Tensor:
        """
        Integrate modality embedding with text embedding.
        
        Args:
            modality_embedding: Embedding from the modality encoder
            text_embedding: Text embedding to integrate with
            integration_method: Integration method ("concat", "add", "attention", etc.)
            
        Returns:
            Integrated embedding
        """
        return self.text_integration.integrate(
            modality_embedding, 
            text_embedding, 
            method=integration_method
        )
    
    def forward(self, 
               x: torch.Tensor,
               model_outputs: Optional[Dict[str, Any]] = None, 
               extension_outputs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass of the multimodal extension.
        
        Args:
            x: Input tensor (typically text embedding)
            model_outputs: Outputs from the main model
            extension_outputs: Outputs from other extensions
            
        Returns:
            Tuple of (modified tensor, extension metadata)
        """
        # Default implementation for when no modality input is provided
        # Subclasses should override for actual modality processing
        metadata = {
            "modality_type": self.modality_type,
            "modality_embedding_dim": self.modality_embedding_dim,
            "has_modality_input": False
        }
        
        return x, metadata