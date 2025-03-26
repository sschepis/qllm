"""
Multi-Resolution Vision Module.

This module provides functionality for processing images at multiple resolutions
to capture details at different scales.
"""

import math
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionVisionProcessor(nn.Module):
    """
    Processor for extracting and combining features at multiple resolutions.
    
    This class handles scaling images to different resolutions, extracting features
    at each scale, and combining them into a unified representation.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        base_processor: nn.Module,
        use_spatial_features: bool = False
    ):
        """
        Initialize multi-resolution processor.
        
        Args:
            embedding_dim: Embedding dimension
            base_processor: Base vision processor module
            use_spatial_features: Whether to use spatial features
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.base_processor = base_processor
        self.use_spatial_features = use_spatial_features
        
        # Projection for combining multi-resolution features
        self.multi_res_projection = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),  # Default 3 scales
            nn.LayerNorm(embedding_dim)
        )
        
        # Attention for combining spatial resolutions
        self.resolution_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            batch_first=True
        )
        
        self.resolution_fusion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def extract_features_at_resolution(
        self,
        image: torch.Tensor,
        scale: float
    ) -> torch.Tensor:
        """
        Extract features at a specific resolution scale.
        
        Args:
            image: Input image tensor [B, C, H, W]
            scale: Scale factor (1.0 = original size)
            
        Returns:
            Features at the specified resolution
        """
        # Skip resizing if scale is 1.0 (original resolution)
        if scale == 1.0:
            # Process at original resolution
            return self.base_processor(image)
        
        # Original image size
        orig_size = image.shape[-2:]
        
        # Resize image to current scale
        scaled_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
        scaled_image = F.interpolate(
            image,
            size=scaled_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Process scaled image
        return self.base_processor(scaled_image)
    
    def combine_spatial_resolutions(
        self,
        multi_scale_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Combine spatial features from multiple resolutions using attention.
        
        Args:
            multi_scale_features: List of features from different resolutions
            
        Returns:
            Combined multi-resolution features
        """
        batch_size = multi_scale_features[0].shape[0]
        seq_len = multi_scale_features[0].shape[1]
        
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
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract and combine features at multiple resolutions.
        
        Args:
            image: Input image tensor [B, C, H, W]
            
        Returns:
            Combined multi-resolution features
        """
        # Define scales for multi-resolution processing
        scales = [1.0, 0.75, 0.5]  # 100%, 75%, and 50% of original size
        batch_size = image.shape[0]
        device = image.device
        
        # Collect features at each resolution
        multi_scale_features = []
        
        for scale in scales:
            # Extract features at current scale
            scaled_features = self.extract_features_at_resolution(image, scale)
            
            # Ensure all features have same sequence length if spatial
            if self.use_spatial_features and scaled_features.dim() > 2 and multi_scale_features:
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
            combined_features = self.combine_spatial_resolutions(multi_scale_features)
        else:
            # For global features, concatenate and project
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