"""
Multimodal Integration Module.

This module provides functions and classes for integrating multimodal
embeddings with text embeddings.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_types import IntegrationMethod


class ModalityTextIntegration(nn.Module):
    """
    Module for integrating modality embeddings with text embeddings.
    
    This class provides various methods for combining different modality
    embeddings with text embeddings, including addition, concatenation,
    and various attention mechanisms.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        fusion_heads: int = 8,
        fusion_iterations: int = 2
    ):
        """
        Initialize integration module.
        
        Args:
            embedding_dim: Dimension of embeddings
            fusion_heads: Number of attention heads for fusion
            fusion_iterations: Number of iterations for iterative fusion
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fusion_heads = fusion_heads
        self.fusion_iterations = fusion_iterations
        
        # Initialize various integration components
        self._initialize_integration_components()
    
    def _initialize_integration_components(self):
        """Initialize components for different integration methods."""
        # Alignment projection for dimension matching
        self.alignment_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        # Attention for integration
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.fusion_heads,
            batch_first=True
        )
        
        # Multi-layer cross attention
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.embedding_dim,
                num_heads=self.fusion_heads,
                batch_first=True
            ) for _ in range(2)  # 2 layers of cross-attention
        ])
        self.cross_layer_norm = nn.LayerNorm(self.embedding_dim)
        
        # FiLM (Feature-wise Linear Modulation)
        self.film_generator = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.LayerNorm(self.embedding_dim * 2)
        )
    
    def align_dimensions(
        self,
        modality_embedding: torch.Tensor,
        text_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Align the modality embedding dimensions with the text embedding.
        
        Args:
            modality_embedding: Embedding from the modality encoder
            text_embedding: Text embedding to align with
            
        Returns:
            Aligned modality embedding
        """
        # Check if dimensions match
        if modality_embedding.shape[-1] != text_embedding.shape[-1]:
            # Use projection to match dimensions
            aligned_embedding = self.alignment_projection(modality_embedding)
        else:
            aligned_embedding = modality_embedding
        
        return aligned_embedding
    
    def add_integration(
        self,
        modality_embedding: torch.Tensor,
        text_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate by adding modality embedding to text embedding.
        
        Args:
            modality_embedding: Embedding from the modality encoder
            text_embedding: Text embedding to integrate with
            
        Returns:
            Integrated embedding
        """
        # Align dimensions first
        aligned_modality = self.align_dimensions(modality_embedding, text_embedding)
        
        # Simple addition
        return text_embedding + aligned_modality
    
    def concat_integration(
        self,
        modality_embedding: torch.Tensor,
        text_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate by concatenating modality embedding with text embedding.
        
        Args:
            modality_embedding: Embedding from the modality encoder
            text_embedding: Text embedding to integrate with
            
        Returns:
            Integrated embedding
        """
        # Align dimensions first
        aligned_modality = self.align_dimensions(modality_embedding, text_embedding)
        
        # Concatenate along sequence dimension
        if len(aligned_modality.shape) == 3:  # [batch, seq, dim]
            return torch.cat([aligned_modality, text_embedding], dim=1)
        else:  # [batch, dim]
            aligned_modality = aligned_modality.unsqueeze(1)  # Add sequence dimension
            return torch.cat([aligned_modality, text_embedding], dim=1)
    
    def attention_integration(
        self,
        modality_embedding: torch.Tensor,
        text_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate using cross-attention.
        
        Args:
            modality_embedding: Embedding from the modality encoder
            text_embedding: Text embedding to integrate with
            
        Returns:
            Integrated embedding
        """
        # Align dimensions first
        aligned_modality = self.align_dimensions(modality_embedding, text_embedding)
        
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
    
    def cross_attention_integration(
        self,
        modality_embedding: torch.Tensor,
        text_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate using multi-layer cross-attention.
        
        Args:
            modality_embedding: Embedding from the modality encoder
            text_embedding: Text embedding to integrate with
            
        Returns:
            Integrated embedding
        """
        # Align dimensions first
        aligned_modality = self.align_dimensions(modality_embedding, text_embedding)
        
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
    
    def film_integration(
        self,
        modality_embedding: torch.Tensor,
        text_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate using Feature-wise Linear Modulation (FiLM).
        
        Args:
            modality_embedding: Embedding from the modality encoder
            text_embedding: Text embedding to integrate with
            
        Returns:
            Integrated embedding
        """
        # Align dimensions first
        aligned_modality = self.align_dimensions(modality_embedding, text_embedding)
        
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
    
    def integrate(
        self,
        modality_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        method: IntegrationMethod = "attention"
    ) -> torch.Tensor:
        """
        Integrate modality embedding with text embedding.
        
        Args:
            modality_embedding: Embedding from the modality encoder
            text_embedding: Text embedding to integrate with
            method: Integration method
            
        Returns:
            Integrated embedding
        """
        if method == "add":
            return self.add_integration(modality_embedding, text_embedding)
        elif method == "concat":
            return self.concat_integration(modality_embedding, text_embedding)
        elif method == "attention":
            return self.attention_integration(modality_embedding, text_embedding)
        elif method == "cross_attention":
            return self.cross_attention_integration(modality_embedding, text_embedding)
        elif method == "film":
            return self.film_integration(modality_embedding, text_embedding)
        else:
            raise ValueError(f"Unknown integration method: {method}")


class DynamicFusionModule(nn.Module):
    """
    Module for dynamically fusing modality and text embeddings based on content relevance.
    
    This class implements adaptive integration where the level of integration
    depends on the estimated relevance between the modality and text.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        high_threshold: float = 0.8,
        low_threshold: float = 0.3
    ):
        """
        Initialize dynamic fusion module.
        
        Args:
            embedding_dim: Dimension of embeddings
            high_threshold: Threshold for high relevance
            low_threshold: Threshold for low relevance
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        
        # Integration components
        self.integration = ModalityTextIntegration(embedding_dim)
        
        # FiLM generator for medium relevance
        self.dynamic_film = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2)
        )
    
    def compute_relevance_score(
        self,
        modality_embedding: torch.Tensor,
        text_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relevance scores between modality and text.
        
        Args:
            modality_embedding: Modality embedding
            text_embedding: Text embedding
            
        Returns:
            Tensor of relevance scores [batch_size]
        """
        # Flatten modality embedding if it has sequence dimension
        if len(modality_embedding.shape) == 3:  # [batch, seq, dim]
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
        
        # Apply sigmoid to get attention weights
        relevance_scores = torch.sigmoid(relevance_scores * 5.0)  # Scale for sharper distribution
        
        return relevance_scores
    
    def forward(
        self,
        modality_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        relevance_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Dynamically fuse modality and text embeddings.
        
        Args:
            modality_embedding: Modality embedding
            text_embedding: Text embedding
            relevance_scores: Optional pre-computed relevance scores
            
        Returns:
            Fused embedding
        """
        device = text_embedding.device
        batch_size = text_embedding.shape[0]
        
        # Compute relevance scores if not provided
        if relevance_scores is None:
            relevance_scores = self.compute_relevance_score(modality_embedding, text_embedding)
        
        # Prepare output tensor
        integrated = text_embedding.clone()
        
        # Process each item in batch according to its relevance
        for i in range(batch_size):
            relevance = relevance_scores[i].item()
            
            if relevance > self.high_threshold:
                # High relevance: Use cross-attention for deep integration
                item_text = text_embedding[i:i+1]
                item_mod = modality_embedding[i:i+1] if modality_embedding.dim() > 1 else modality_embedding.unsqueeze(0)
                
                # Ensure proper dimensions
                if item_mod.dim() == 2:
                    item_mod = item_mod.unsqueeze(1)  # Add sequence dimension [1, 1, dim]
                
                # Use cross-attention integration
                integrated[i:i+1] = self.integration.cross_attention_integration(item_mod, item_text)
                
            elif relevance > self.low_threshold:
                # Medium relevance: Weighted addition
                scale = (relevance - self.low_threshold) / (self.high_threshold - self.low_threshold)
                
                # Prepare modality embedding
                item_mod = modality_embedding[i] if modality_embedding.dim() > 1 else modality_embedding
                if item_mod.dim() == 2:  # [seq, dim]
                    item_mod = item_mod.mean(dim=0)  # Average over sequence
                
                # Generate scale and shift parameters
                film_params = self.dynamic_film(item_mod)
                scale_param, shift_param = torch.chunk(film_params, 2)
                
                # Apply scaled transformation
                # Higher relevance = stronger effect
                effective_scale = 1.0 + scale * 0.5 * scale_param  # Max 50% scaling
                effective_shift = scale * 0.3 * shift_param  # Max 30% shift
                
                integrated[i] = effective_scale * text_embedding[i] + effective_shift
                
            else:
                # Low relevance: Minimal integration (keep text mostly as is)
                item_mod = modality_embedding[i] if modality_embedding.dim() > 1 else modality_embedding
                if item_mod.dim() == 2:  # [seq, dim]
                    item_mod = item_mod.mean(dim=0)
                
                # Apply minimal shift
                shift = relevance * 0.1 * F.normalize(item_mod, dim=-1)  # Max 10% effect
                integrated[i] = text_embedding[i] + shift.unsqueeze(0) if text_embedding[i].dim() > 1 else text_embedding[i] + shift
        
        return integrated