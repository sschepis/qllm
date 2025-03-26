"""
Vision Encoder Module.

This module provides encoders for processing visual inputs in the
Semantic Resonance Language Model.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    """
    Base class for vision encoders.
    
    This class provides common functionality for different vision encoder types,
    handling pretrained models, feature extraction, and projection.
    """
    
    def __init__(
        self,
        encoder_type: str = "resnet",
        encoder_model: str = "resnet50",
        embedding_dim: int = 768,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        use_spatial_features: bool = False,
        image_size: int = 224
    ):
        """
        Initialize the vision encoder.
        
        Args:
            encoder_type: Type of encoder ("resnet", "vit", "efficientnet")
            encoder_model: Specific model name
            embedding_dim: Dimension of output embeddings
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone
            use_spatial_features: Whether to return spatial features
            image_size: Input image size
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.encoder_model = encoder_model
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.use_spatial_features = use_spatial_features
        self.image_size = image_size
        
        # Initialize the vision encoder
        self._initialize_vision_encoder()
    
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
        print(f"Initialized VisionEncoder with {model_name} backbone")
        print(f"Feature dimension: {feature_dim}")
        print(f"Using spatial features: {self.use_spatial_features}")
        print(f"Backbone frozen: {self.freeze_backbone}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the vision encoder.
        
        Args:
            x: Input image tensor [batch_size, channels, height, width]
            
        Returns:
            Encoded features
        """
        # Extract features using the base model
        with torch.set_grad_enabled(not self.freeze_backbone):
            features = self.base_model(x)
        
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


class PrimeProjectionEncoder(nn.Module):
    """
    Prime-based projection encoder for vision features.
    
    This class projects vision features into a set of prime-dimensional subspaces
    to maintain the quantum-inspired design of the main model.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        prime_dims: List[int] = [11, 13, 17, 19, 23]
    ):
        """
        Initialize prime projection encoder.
        
        Args:
            embedding_dim: Input embedding dimension
            prime_dims: List of prime dimensions for projections
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.prime_dims = prime_dims
        
        # Prime projections for vision features
        self.prime_projections = nn.ModuleList([
            nn.Linear(embedding_dim, prime) for prime in prime_dims
        ])
        
        # Layer norm for prime projections
        self.prime_norm = nn.LayerNorm(sum(prime_dims))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through prime projection encoder.
        
        Args:
            x: Input features [batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
            
        Returns:
            Prime-projected features
        """
        # Store original shape
        original_shape = x.shape
        
        # Reshape to [batch_size*seq_len, embedding_dim] if needed
        if len(original_shape) == 3:
            batch_size, seq_len, _ = original_shape
            x = x.reshape(batch_size * seq_len, self.embedding_dim)
        
        # Project features to each prime subspace
        prime_embeddings = []
        for proj in self.prime_projections:
            prime_embed = proj(x)
            prime_embeddings.append(prime_embed)
        
        # Concatenate along the last dimension
        prime_concat = torch.cat(prime_embeddings, dim=-1)  # [batch*seq, sum(primes)]
        
        # Apply layer normalization
        prime_concat = self.prime_norm(prime_concat)
        
        # Reshape back to original shape if needed
        if len(original_shape) == 3:
            prime_concat = prime_concat.reshape(batch_size, seq_len, -1)
        
        return prime_concat