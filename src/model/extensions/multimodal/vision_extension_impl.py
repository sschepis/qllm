"""
Vision Extension Implementation Module.

This module defines the implementation for processing visual inputs in the
Semantic Resonance Language Model.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_multimodal_core import BaseMultimodalExtension
from .multimodal_config import VisionConfig
from .vision_encoder import VisionEncoder, PrimeProjectionEncoder
from .multi_resolution_vision import MultiResolutionVisionProcessor
from .multimodal_integration import DynamicFusionModule


class VisionExtension(BaseMultimodalExtension):
    """
    Extension for processing visual inputs (images).
    
    This extension integrates visual information into the language model
    using a prime-based projection approach that maintains the quantum-inspired
    design of the main model. It supports various pre-trained models including
    ResNet, Vision Transformers, and EfficientNet.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the vision extension.
        
        Args:
            name: Unique name for this extension instance
            config: Configuration dictionary for the extension
        """
        # Set default modality type
        config["modality_type"] = "vision"
        
        super().__init__(name, config)
        
        # Create vision-specific configuration
        self.vision_config = VisionConfig.from_dict(config)
        
        # Extract key configuration parameters for convenience
        self.encoder_type = self.vision_config.vision_encoder_type
        self.encoder_model = self.vision_config.vision_encoder_model
        self.embedding_dim = self.vision_config.vision_embedding_dim
        self.vision_primes = self.vision_config.vision_primes
        self.use_spatial_features = self.vision_config.use_spatial_features
        self.use_multi_resolution = self.vision_config.use_multi_resolution
        self.use_dynamic_fusion = self.vision_config.use_dynamic_fusion
        self.image_size = self.vision_config.image_size
        
        # Initialize image preprocessing
        self.register_buffer("image_mean", torch.tensor(self.vision_config.image_mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(self.vision_config.image_std).view(1, 3, 1, 1))
        
        # Initialize vision encoder components
        self._initialize_vision_components()
    
    def _initialize_vision_components(self):
        """Initialize vision encoder and processing components."""
        # Create base vision encoder
        self.vision_encoder = VisionEncoder(
            encoder_type=self.encoder_type,
            encoder_model=self.encoder_model,
            embedding_dim=self.embedding_dim,
            pretrained=self.vision_config.pretrained,
            freeze_backbone=self.vision_config.freeze_backbone,
            use_spatial_features=self.use_spatial_features,
            image_size=self.image_size
        )
        
        # Initialize prime projection if needed
        if not self.use_spatial_features:
            self.prime_projector = PrimeProjectionEncoder(
                embedding_dim=self.embedding_dim,
                prime_dims=self.vision_primes
            )
        
        # Initialize multi-resolution processor if enabled
        if self.use_multi_resolution:
            self.multi_res_processor = MultiResolutionVisionProcessor(
                embedding_dim=self.embedding_dim,
                base_processor=self.vision_encoder,
                use_spatial_features=self.use_spatial_features
            )
        
        # Initialize dynamic fusion module if enabled
        if self.use_dynamic_fusion:
            self.dynamic_fusion = DynamicFusionModule(
                embedding_dim=self.embedding_dim if not self.use_spatial_features else self.embedding_dim,
                high_threshold=0.8,
                low_threshold=0.3
            )
    
    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess an image for the vision encoder.
        
        Args:
            image: Image tensor of shape [B, C, H, W]
            
        Returns:
            Preprocessed image
        """
        # Resize if needed
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Normalize
        if hasattr(self, "image_mean") and hasattr(self, "image_std"):
            image = (image - self.image_mean) / self.image_std
        
        return image
    
    def encode_modality(self, modality_input: torch.Tensor) -> torch.Tensor:
        """
        Encode the image input into the model's embedding space.
        
        Args:
            modality_input: Image tensor of shape [B, C, H, W]
            
        Returns:
            Encoded representation
        """
        # Basic validation
        if modality_input.dim() != 4:
            raise ValueError(f"Expected 4D image tensor, got shape: {modality_input.shape}")
        
        # Preprocess image
        image = self.preprocess_image(modality_input)
        
        # Process the image - use multi-resolution features if enabled
        if self.use_multi_resolution:
            features = self.multi_res_processor(image)
        else:
            # Use base vision encoder
            features = self.vision_encoder(image)
        
        # Apply prime-based projections for non-spatial features
        if not self.use_spatial_features and hasattr(self, "prime_projector"):
            features = self.prime_projector(features)
        
        return features
    
    def initialize(self, model: nn.Module) -> None:
        """
        Initialize the vision extension with the main model.
        
        Args:
            model: The main model instance
        """
        self.model = model
        
        # Add any hooks or connections to the main model
        for name, module in model.named_modules():
            if "encoder" in name and isinstance(module, nn.Module):
                # Hook to integrate vision features after encoding
                self.register_forward_hook(model, name, self._vision_hook)
                break
        
        self.initialized = True
    
    def _vision_hook(self, module, input, output):
        """Hook function for integrating vision features."""
        # This would be called during the forward pass of the main model
        # We're just defining the placeholder for now
        return output
    
    def forward(self,
               x: torch.Tensor,
               model_outputs: Optional[Dict[str, Any]] = None,
               extension_outputs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass of the vision extension.
        
        Args:
            x: Input tensor (text embedding)
            model_outputs: Outputs from the main model
            extension_outputs: Outputs from other extensions
            
        Returns:
            Tuple of (modified tensor, extension metadata)
        """
        # Initialize metadata
        metadata = {
            "modality_type": "vision",
            "modality_embedding_dim": self.embedding_dim,
            "has_modality_input": False,
            "vision_primes": self.vision_primes,
            "model": self.encoder_model,
            "spatial_features": self.use_spatial_features
        }
        
        # Check if we have image input in model_outputs
        if model_outputs is None or "image_input" not in model_outputs:
            # No image input, return original tensor
            return x, metadata
        
        # Get image input
        image_input = model_outputs["image_input"]
        metadata["has_modality_input"] = True
        metadata["image_shape"] = list(image_input.shape)
        
        # Check if we already computed embeddings for this image
        # Only hash if the image is not too large to avoid memory issues
        image_hash = str(hash(image_input.cpu().numpy().tobytes())) if image_input.numel() < 10000 else None
        
        if image_hash and image_hash in self.input_embedding_cache:
            vision_embedding = self.input_embedding_cache[image_hash]
        else:
            # Encode the image
            vision_embedding = self.encode_modality(image_input)
            
            # Cache the embedding if hash is available
            if image_hash:
                self.input_embedding_cache[image_hash] = vision_embedding
        
        # Choose integration method based on configuration or explicit request
        integration_method = model_outputs.get("integration_method", self.vision_config.fusion_type)
        
        # Use dynamic fusion if enabled, otherwise use regular integration
        if self.use_dynamic_fusion:
            # Add dynamic fusion information to metadata
            metadata["using_dynamic_fusion"] = True
            
            # Get relevance scores if available
            relevance_scores = model_outputs.get("relevance_scores", None)
            
            # Apply dynamic fusion
            integrated_embedding = self.dynamic_fusion(
                vision_embedding,
                x,
                relevance_scores=relevance_scores
            )
        else:
            # Use standard integration method
            integrated_embedding = self.integrate_with_text(
                vision_embedding,
                x,
                integration_method=integration_method
            )
        
        # Update metadata
        metadata["vision_embedding_shape"] = list(vision_embedding.shape)
        metadata["integrated_embedding_shape"] = list(integrated_embedding.shape)
        metadata["integration_method"] = integration_method
        
        return integrated_embedding, metadata