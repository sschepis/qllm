"""
Vision Extension Module.

This module defines the extension for processing visual inputs in the
Semantic Resonance Language Model.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_multimodal_extension import BaseMultimodalExtension


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
            name (str): Unique name for this extension instance
            config (Dict[str, Any]): Configuration dictionary for the extension
        """
        # Set default modality type
        config["modality_type"] = "vision"
        
        super().__init__(name, config)
        
        # Vision model configuration
        self.encoder_type = config.get("vision_encoder_type", "resnet")
        self.encoder_model = config.get("vision_encoder_model", "resnet50")
        self.embedding_dim = config.get("vision_embedding_dim", 768)
        self.vision_primes = config.get("vision_primes", [11, 13, 17, 19, 23])
        
        # Advanced configuration for pre-trained models
        self.pretrained = config.get("pretrained", True)
        self.freeze_backbone = config.get("freeze_backbone", True)
        self.feature_extraction_layer = config.get("feature_extraction_layer", "penultimate")
        self.use_spatial_features = config.get("use_spatial_features", False)
        
        # Multi-resolution and dynamic fusion settings
        self.use_multi_resolution = config.get("use_multi_resolution", False)
        self.use_dynamic_fusion = config.get("use_dynamic_fusion", False)
        self.relevance_threshold = config.get("relevance_threshold", 0.5)
        
        # Input image configuration
        self.image_size = config.get("image_size", 224)
        self.use_patch_tokens = config.get("use_patch_tokens", False)
        
        # Initialize image preprocessing
        mean = config.get("image_mean", [0.485, 0.456, 0.406])
        std = config.get("image_std", [0.229, 0.224, 0.225])
        
        self.register_buffer("image_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(std).view(1, 3, 1, 1))
        
        # Initialize the vision encoder
        self._initialize_vision_encoder()
        
        # Prime-based projections
        self._initialize_prime_projections()
        
        # Initialize vision-text fusion module
        self.fusion_type = config.get("fusion_type", "attention")
        self.fusion_heads = config.get("fusion_heads", 8)
        self.fusion_iterations = config.get("fusion_iterations", 5)
        self.fusion_entropy_threshold = config.get("fusion_entropy_threshold", 0.2)
        
        # Initialize any additional fusion components
        self._initialize_fusion_components()
    
    def _initialize_vision_encoder(self):
        """Initialize the vision encoder with a pre-trained model."""
        import torch.nn as nn
        import torchvision.models as models
        
        model_name = self.encoder_model
        
        # Create model based on configuration
        if model_name.startswith("resnet"):
            if model_name == "resnet18":
                base_model = models.resnet18(pretrained=self.pretrained)
                feature_dim = 512
            elif model_name == "resnet34":
                base_model = models.resnet34(pretrained=self.pretrained)
                feature_dim = 512
            elif model_name == "resnet50":
                base_model = models.resnet50(pretrained=self.pretrained)
                feature_dim = 2048
            elif model_name == "resnet101":
                base_model = models.resnet101(pretrained=self.pretrained)
                feature_dim = 2048
            elif model_name == "resnet152":
                base_model = models.resnet152(pretrained=self.pretrained)
                feature_dim = 2048
            else:
                # Default to resnet50
                base_model = models.resnet50(pretrained=self.pretrained)
                feature_dim = 2048
            
            # Remove classification head
            self.base_model = nn.Sequential(*list(base_model.children())[:-2 if self.use_spatial_features else -1])
            
        elif model_name.startswith("vit"):
            # Use Vision Transformer models
            if model_name == "vit_b_16":
                base_model = models.vit_b_16(pretrained=self.pretrained)
                feature_dim = 768
            elif model_name == "vit_b_32":
                base_model = models.vit_b_32(pretrained=self.pretrained)
                feature_dim = 768
            elif model_name == "vit_l_16":
                base_model = models.vit_l_16(pretrained=self.pretrained)
                feature_dim = 1024
            elif model_name == "vit_l_32":
                base_model = models.vit_l_32(pretrained=self.pretrained)
                feature_dim = 1024
            else:
                # Default to ViT-B/16
                base_model = models.vit_b_16(pretrained=self.pretrained)
                feature_dim = 768
            
            # Special handling for ViT models
            if not self.use_spatial_features:
                # Use only the classifier token
                self.base_model = base_model
                self.extract_cls_token = True
            else:
                # Use all patch features
                class VitFeatureExtractor(nn.Module):
                    def __init__(self, vit_model):
                        super().__init__()
                        self.vit_model = vit_model
                    
                    def forward(self, x):
                        # Get patch embeddings before pooling
                        return self.vit_model._process_input(x)
                
                self.base_model = VitFeatureExtractor(base_model)
                self.extract_cls_token = False
                
        elif model_name.startswith("efficientnet"):
            # Use EfficientNet models
            if model_name == "efficientnet_b0":
                base_model = models.efficientnet_b0(pretrained=self.pretrained)
                feature_dim = 1280
            elif model_name == "efficientnet_b1":
                base_model = models.efficientnet_b1(pretrained=self.pretrained)
                feature_dim = 1280
            elif model_name == "efficientnet_b2":
                base_model = models.efficientnet_b2(pretrained=self.pretrained)
                feature_dim = 1408
            elif model_name == "efficientnet_b3":
                base_model = models.efficientnet_b3(pretrained=self.pretrained)
                feature_dim = 1536
            else:
                # Default to EfficientNet-B0
                base_model = models.efficientnet_b0(pretrained=self.pretrained)
                feature_dim = 1280
            
            # Remove classification head
            self.base_model = nn.Sequential(*list(base_model.children())[:-1])
            
        else:
            # Fallback to a simple CNN for demonstration
            self.base_model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            feature_dim = 128
        
        # Freeze backbone if specified
        if self.freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Create feature projection to embedding dimension
        self.feature_dim = feature_dim
        self.feature_projection = nn.Linear(feature_dim, self.embedding_dim)
        
        # Spatial feature handling
        if self.use_spatial_features:
            self.spatial_pool = nn.AdaptiveAvgPool2d((7, 7))  # Fixed spatial size
            self.spatial_projection = nn.Conv2d(feature_dim, self.embedding_dim, kernel_size=1)
        
        # Update model name for logging
        print(f"Initialized VisionExtension with {model_name} backbone")
        print(f"Feature dimension: {feature_dim}")
        print(f"Using spatial features: {self.use_spatial_features}")
        print(f"Backbone frozen: {self.freeze_backbone}")
    
    def _initialize_prime_projections(self):
        """Initialize prime-based projections for the vision embeddings."""
        # Prime projections for vision features
        self.prime_projections = nn.ModuleList([
            nn.Linear(self.embedding_dim, prime) for prime in self.vision_primes
        ])
        
        # Layer norm for prime projections
        self.prime_norm = nn.LayerNorm(sum(self.vision_primes))
    
    def _initialize_fusion_components(self):
        """Initialize components for fusing vision and text."""
        # Output dimension of prime projections
        prime_sum = sum(self.vision_primes)
        
        if self.fusion_type == "attention":
            # Vision-text cross-attention
            self.vision_text_attention = nn.MultiheadAttention(
                embed_dim=prime_sum,
                num_heads=self.fusion_heads,
                batch_first=True
            )
        elif self.fusion_type == "cross_attention":
            # Advanced cross-attention implementation
            self.cross_attention_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=self.embedding_dim,
                    num_heads=self.fusion_heads,
                    batch_first=True
                ) for _ in range(2)  # 2 layers of cross-attention
            ])
            self.cross_layer_norm = nn.LayerNorm(self.embedding_dim)
        elif self.fusion_type == "film":
            # Feature-wise Linear Modulation
            self.film_generator = nn.Sequential(
                nn.Linear(prime_sum, self.embedding_dim * 2),
                nn.LayerNorm(self.embedding_dim * 2)
            )
        
        # Final projection to model dimension
        self.output_projection = nn.Linear(prime_sum, self.embedding_dim)
    
    def initialize(self, model: nn.Module) -> None:
        """
        Initialize the vision extension with the main model.
        
        Args:
            model (nn.Module): The main model instance
        """
        self.model = model
        
        # Add any hooks or connections to the main model
        # For vision, we might hook after the token embedding layer
        
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
    
    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess an image for the vision encoder.
        
        Args:
            image (torch.Tensor): Image tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Preprocessed image
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
    
    def extract_multi_resolution_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract features at multiple resolutions to capture details at different scales.
        
        Args:
            image (torch.Tensor): Image tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Multi-resolution features
        """
        # Original image size
        orig_size = image.shape[-2:]
        batch_size = image.shape[0]
        device = image.device
        
        # Define scales for multi-resolution processing
        scales = [1.0, 0.75, 0.5]  # 100%, 75%, and 50% of original size
        
        # Collect features at each resolution
        multi_scale_features = []
        
        for scale in scales:
            # Skip if scale is 1.0 (original resolution)
            if scale == 1.0:
                # Process at original resolution
                features = self._process_image(image)
                multi_scale_features.append(features)
                continue
                
            # Resize image to current scale
            scaled_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
            scaled_image = F.interpolate(
                image,
                size=scaled_size,
                mode='bilinear',
                align_corners=False
            )
            
            # Process scaled image
            scaled_features = self._process_image(scaled_image)
            
            # Ensure all features have same sequence length if spatial
            if self.use_spatial_features and scaled_features.dim() > 2:
                # Resize feature map to match original feature sequence length
                target_len = multi_scale_features[0].size(1)
                current_len = scaled_features.size(1)
                
                if current_len != target_len:
                    # Reshape, resize, then reshape back
                    h = int(math.sqrt(current_len))
                    scaled_features = scaled_features.transpose(1, 2).view(batch_size, -1, h, h)
                    target_h = int(math.sqrt(target_len))
                    scaled_features = F.interpolate(
                        scaled_features,
                        size=(target_h, target_h),
                        mode='bilinear',
                        align_corners=False
                    )
                    scaled_features = scaled_features.view(batch_size, -1, target_len).transpose(1, 2)
            
            # Add to collection
            multi_scale_features.append(scaled_features)
        
        # Combine features from different resolutions
        if self.use_spatial_features and multi_scale_features[0].dim() > 2:
            # For spatial features, use attention to combine
            combined_features = self._combine_spatial_resolutions(multi_scale_features)
        else:
            # For global features, concatenate and project
            if not hasattr(self, "multi_res_projection"):
                # Create projection layer for combining resolutions
                input_dim = self.embedding_dim * len(scales)
                self.multi_res_projection = nn.Sequential(
                    nn.Linear(input_dim, self.embedding_dim),
                    nn.LayerNorm(self.embedding_dim)
                ).to(device)
            
            # Concatenate global features
            if multi_scale_features[0].dim() > 2:
                # Average spatial dimensions first
                global_features = [f.mean(dim=1) for f in multi_scale_features]
                concat_features = torch.cat(global_features, dim=1)
            else:
                concat_features = torch.cat(multi_scale_features, dim=1)
                
            # Project back to embedding dimension
            combined_features = self.multi_res_projection(concat_features)
            
            # Add sequence dimension if needed
            if multi_scale_features[0].dim() > 2:
                combined_features = combined_features.unsqueeze(1)
        
        return combined_features
    
    def _combine_spatial_resolutions(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine spatial features from multiple resolutions using attention.
        
        Args:
            multi_scale_features (List[torch.Tensor]): List of features from different resolutions
            
        Returns:
            torch.Tensor: Combined multi-resolution features
        """
        batch_size = multi_scale_features[0].shape[0]
        seq_len = multi_scale_features[0].shape[1]
        device = multi_scale_features[0].device
        
        # Create attention mechanisms if not already defined
        if not hasattr(self, "resolution_attention"):
            self.resolution_attention = nn.MultiheadAttention(
                embed_dim=self.embedding_dim,
                num_heads=4,
                batch_first=True
            ).to(device)
            
            self.resolution_fusion = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim)
            ).to(device)
        
        # Stack all resolutions
        # [batch, scales, seq_len, dim]
        stacked_features = torch.stack(multi_scale_features, dim=1)
        
        # Reshape to [batch*seq_len, scales, dim] for attention
        reshaped = stacked_features.view(batch_size * seq_len, len(multi_scale_features), self.embedding_dim)
        
        # Apply self-attention across resolutions
        attended, _ = self.resolution_attention(
            query=reshaped,
            key=reshaped,
            value=reshaped
        )
        
        # Apply fusion layer
        fused = self.resolution_fusion(attended)
        
        # Take weighted average across resolution dimension
        combined = fused.mean(dim=1)
        
        # Reshape back to [batch, seq_len, dim]
        combined = combined.view(batch_size, seq_len, self.embedding_dim)
        
        return combined
        
    def _process_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Process an image through the vision encoder.
        
        Args:
            image (torch.Tensor): Image tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Encoded image features
        """
        # Basic validation
        if image.dim() != 4:
            raise ValueError(f"Expected 4D image tensor, got shape: {image.shape}")
        
        # Preprocess image
        image = self.preprocess_image(image)
        
        # Extract features using the base model
        with torch.set_grad_enabled(not self.freeze_backbone):
            features = self.base_model(image)
        
        # Process features based on model type and configuration
        if hasattr(self, "extract_cls_token") and self.extract_cls_token:
            # For ViT models, extract CLS token
            if isinstance(features, tuple):
                # Some ViT implementations return multiple outputs
                features = features[0][:, 0]  # Get CLS token
            else:
                features = features[:, 0]  # Get CLS token
            
            # Project to embedding dimension
            features = self.feature_projection(features)
            
        elif self.use_spatial_features:
            # Process spatial features (feature maps)
            if features.dim() == 4:
                # Handle feature maps (B, C, H, W)
                batch_size = features.shape[0]
                
                # Apply spatial projection to reduce channels
                features = self.spatial_projection(features)  # [B, embedding_dim, H, W]
                
                # Adaptive pooling to fixed spatial dimensions
                features = self.spatial_pool(features)  # [B, embedding_dim, 7, 7]
                
                # Flatten spatial dimensions
                features = features.view(batch_size, self.embedding_dim, -1)  # [B, embedding_dim, 49]
                
                # Transpose to get sequence-like format
                features = features.transpose(1, 2)  # [B, 49, embedding_dim]
                
            else:
                # Handle unexpected format with fallback to global pooling
                batch_size = features.shape[0]
                features = features.view(batch_size, -1)  # Flatten all dimensions
                features = self.feature_projection(features)  # Project to embedding dim
                features = features.unsqueeze(1)  # Add sequence dimension [B, 1, embedding_dim]
                
        else:
            # Global pooling for feature maps
            if features.dim() == 4:
                # Global average pooling
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            
            # Project to embedding dimension
            features = self.feature_projection(features)
        
        return features
    
    def _apply_prime_projections(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply prime-based projections to the vision features.
        
        Args:
            features (torch.Tensor): Vision features
            
        Returns:
            torch.Tensor: Prime-projected features
        """
        batch_size = features.shape[0]
        
        # Project features to each prime subspace
        prime_embeddings = []
        for proj in self.prime_projections:
            prime_embed = proj(features)  # [B, prime_i]
            prime_embeddings.append(prime_embed)
        
        # Concatenate along the last dimension
        prime_concat = torch.cat(prime_embeddings, dim=-1)  # [B, sum(primes)]
        
        # Apply layer normalization
        prime_concat = self.prime_norm(prime_concat)
        
        return prime_concat
    
    def dynamic_fusion(self,
                      modality_embedding: torch.Tensor,
                      text_embedding: torch.Tensor,
                      relevance_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform dynamic fusion based on content relevance.
        
        Args:
            modality_embedding (torch.Tensor): Vision embeddings
            text_embedding (torch.Tensor): Text embeddings to integrate with
            relevance_scores (torch.Tensor, optional): Pre-computed relevance scores
            
        Returns:
            torch.Tensor: Dynamically fused embeddings
        """
        device = text_embedding.device
        batch_size = text_embedding.shape[0]
        
        # Compute relevance scores if not provided
        if relevance_scores is None:
            # Normalize embeddings
            if len(modality_embedding.shape) == 3:  # [batch, seq, dim]
                # Average over sequence dimension if present
                modality_embedding_flat = modality_embedding.mean(dim=1)  # [batch, dim]
            else:
                modality_embedding_flat = modality_embedding
                
            # Get last token representation for text
            text_repr = text_embedding[:, -1, :] if text_embedding.dim() > 2 else text_embedding
            
            # Compute relevance as cosine similarity
            text_norm = F.normalize(text_repr, p=2, dim=-1)
            mod_norm = F.normalize(modality_embedding_flat, p=2, dim=-1)
            relevance_scores = torch.bmm(
                text_norm.unsqueeze(1),
                mod_norm.unsqueeze(2)
            ).squeeze()  # [batch]
            
            # Apply softmax to get attention weights
            relevance_scores = torch.sigmoid(relevance_scores * 5.0)  # Scale for sharper distribution
        
        # Dynamic routing based on relevance
        # 1. High relevance: Use cross-attention integration
        # 2. Medium relevance: Use addition with scaling
        # 3. Low relevance: Minimal integration
        
        # Define relevance thresholds
        high_thresh = 0.8
        low_thresh = 0.3
        
        # Prepare output tensor
        integrated = text_embedding.clone()
        
        # Process each item in batch according to its relevance
        for i in range(batch_size):
            relevance = relevance_scores[i].item()
            
            if relevance > high_thresh:
                # High relevance: Use cross-attention for deep integration
                item_text = text_embedding[i:i+1]
                item_mod = modality_embedding[i:i+1] if modality_embedding.dim() > 1 else modality_embedding.unsqueeze(0)
                
                # Ensure proper dimensions
                if item_mod.dim() == 2:
                    item_mod = item_mod.unsqueeze(1)  # Add sequence dimension [1, 1, dim]
                
                # Use cross-attention integration
                attended, _ = self.integrate_with_text(
                    item_mod,
                    item_text,
                    integration_method="cross_attention"
                )
                integrated[i:i+1] = attended
                
            elif relevance > low_thresh:
                # Medium relevance: Weighted addition
                scale = (relevance - low_thresh) / (high_thresh - low_thresh)
                
                # Prepare modality embedding
                item_mod = modality_embedding[i] if modality_embedding.dim() > 1 else modality_embedding
                if item_mod.dim() == 2:  # [seq, dim]
                    item_mod = item_mod.mean(dim=0)  # Average over sequence
                
                # Apply FiLM-like integration with dynamic scaling
                if not hasattr(self, "dynamic_film"):
                    self.dynamic_film = nn.Sequential(
                        nn.Linear(item_mod.shape[-1], text_embedding.shape[-1] * 2),
                        nn.LayerNorm(text_embedding.shape[-1] * 2)
                    ).to(device)
                
                # Generate scale and shift parameters based on modality and relevance
                film_params = self.dynamic_film(item_mod)
                scale_param, shift_param = torch.chunk(film_params, 2)
                
                # Apply scaled transformation
                # Higher relevance = stronger effect
                effective_scale = 1.0 + scale * 0.5 * scale_param  # Max 50% scaling
                effective_shift = scale * 0.3 * shift_param  # Max 30% shift
                
                integrated[i] = effective_scale * text_embedding[i] + effective_shift
                
            else:
                # Low relevance: Minimal integration (keep text mostly as is)
                # Maybe just add a small bias
                item_mod = modality_embedding[i] if modality_embedding.dim() > 1 else modality_embedding
                if item_mod.dim() == 2:  # [seq, dim]
                    item_mod = item_mod.mean(dim=0)
                
                # Apply minimal shift
                shift = relevance * 0.1 * F.normalize(item_mod, dim=-1)  # Max 10% effect
                integrated[i] = text_embedding[i] + shift.unsqueeze(0) if text_embedding[i].dim() > 1 else text_embedding[i] + shift
        
        return integrated

    def integrate_with_text(self,
                           modality_embedding: torch.Tensor,
                           text_embedding: torch.Tensor,
                           integration_method: str = "attention") -> torch.Tensor:
        """
        Integrate modality embedding with text embedding using advanced methods.
        
        Args:
            modality_embedding (torch.Tensor): Embedding from the modality encoder
            text_embedding (torch.Tensor): Text embedding to integrate with
            integration_method (str): Integration method
            
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
            if not hasattr(self, 'vision_text_attention'):
                self.vision_text_attention = nn.MultiheadAttention(
                    embed_dim=text_embedding.shape[-1],
                    num_heads=self.fusion_heads,
                    batch_first=True
                ).to(text_embedding.device)
            
            # Reshape if needed
            if len(aligned_modality.shape) == 2:  # [batch, dim]
                aligned_modality = aligned_modality.unsqueeze(1)  # Add sequence dimension
            
            # Cross-attention: text attends to modality
            integrated, _ = self.vision_text_attention(
                query=text_embedding,
                key=aligned_modality,
                value=aligned_modality
            )
            
            return integrated
            
        elif integration_method == "cross_attention":
            if not hasattr(self, "cross_attention_layers"):
                # Create multi-layer cross attention
                self.cross_attention_layers = nn.ModuleList([
                    nn.MultiheadAttention(
                        embed_dim=text_embedding.shape[-1],
                        num_heads=self.fusion_heads,
                        batch_first=True
                    ).to(text_embedding.device) for _ in range(2)  # 2 layers of cross-attention
                ])
                self.cross_layer_norm = nn.LayerNorm(text_embedding.shape[-1]).to(text_embedding.device)
            
            # Ensure modality embedding has sequence dimension
            if len(aligned_modality.shape) == 2:
                aligned_modality = aligned_modality.unsqueeze(1)
            
            # Apply cross-attention layers
            x = text_embedding
            for layer in self.cross_attention_layers:
                # Cross-attention: text attends to modality
                attn_output, _ = layer(
                    query=x,
                    key=aligned_modality,
                    value=aligned_modality
                )
                # Residual connection and layer norm
                x = self.cross_layer_norm(x + attn_output)
            
            return x
            
        elif integration_method == "film":
            # Feature-wise Linear Modulation (FiLM)
            if not hasattr(self, "film_generator"):
                # Create FiLM parameter generator
                self.film_generator = nn.Sequential(
                    nn.Linear(aligned_modality.shape[-1], text_embedding.shape[-1] * 2),
                    nn.LayerNorm(text_embedding.shape[-1] * 2)
                ).to(text_embedding.device)
            
            # Generate scale and shift parameters
            if len(aligned_modality.shape) == 3:
                # Average over sequence dimension if present
                aligned_modality = aligned_modality.mean(dim=1)
            
            film_params = self.film_generator(aligned_modality)  # [batch_size, hidden_dim*2]
            scale, shift = torch.chunk(film_params, 2, dim=-1)
            
            # Apply FiLM transformation: scale * x + shift
            scale = scale.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            shift = shift.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            return scale * text_embedding + shift
        
        else:
            # Default to simple attention if unknown method
            return super().integrate_with_text(modality_embedding, text_embedding, "attention")
    
    def encode_modality(self, modality_input: torch.Tensor) -> torch.Tensor:
        """
        Encode the image input into the model's embedding space.
        
        Args:
            modality_input (torch.Tensor): Image tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Encoded representation
        """
        # Process the image - use multi-resolution features if enabled
        if self.use_multi_resolution:
            features = self.extract_multi_resolution_features(modality_input)
        else:
            features = self._process_image(modality_input)
        
        # If using spatial features, we might not want prime projections
        if self.use_spatial_features and features.dim() > 2:
            return features
        
        # Apply prime-based projections for non-spatial features
        prime_features = self._apply_prime_projections(features)
        
        return prime_features
    
    def forward(self,
               x: torch.Tensor,
               model_outputs: Optional[Dict[str, Any]] = None,
               extension_outputs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass of the vision extension.
        
        Args:
            x (torch.Tensor): Input tensor (text embedding)
            model_outputs (Dict[str, Any], optional): Outputs from the main model
            extension_outputs (Dict[str, Any], optional): Outputs from other extensions
            
        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Modified tensor and extension metadata
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
        integration_method = model_outputs.get("integration_method", self.fusion_type)
        
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