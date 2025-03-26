"""
Cross-Modal Transfer Learning Module.

This module enables transfer of knowledge between different modalities,
allowing for improved representations through cross-modal learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union


class CrossModalAlignmentLayer(nn.Module):
    """
    Aligns representations across different modalities to enable transfer learning.
    
    This layer uses contrastive learning techniques to align embeddings from
    different modalities in a shared semantic space.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        temperature: float = 0.07,
        projection_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize the cross-modal alignment layer.
        
        Args:
            embedding_dim: Dimension of input embeddings
            temperature: Temperature parameter for contrastive learning
            projection_dim: Dimension of the projection space
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.projection_dim = projection_dim
        
        # Projection networks for each modality
        self.visual_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Optional projections for other modalities
        self.audio_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        return_projections: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute aligned cross-modal representations and contrastive loss.
        
        Args:
            visual_features: Visual features [batch_size, visual_len, embedding_dim]
            text_features: Text features [batch_size, text_len, embedding_dim]
            audio_features: Optional audio features [batch_size, audio_len, embedding_dim]
            return_projections: Whether to return projected features
            
        Returns:
            Dictionary containing aligned representations and contrastive loss
        """
        # Global pooling to get sequence-level representations
        if visual_features.dim() > 2:
            visual_global = visual_features.mean(dim=1)  # [batch_size, embedding_dim]
        else:
            visual_global = visual_features
            
        if text_features.dim() > 2:
            text_global = text_features.mean(dim=1)  # [batch_size, embedding_dim]
        else:
            text_global = text_features
        
        # Project features to alignment space
        visual_proj = self.visual_projection(visual_global)  # [batch_size, projection_dim]
        text_proj = self.text_projection(text_global)  # [batch_size, projection_dim]
        
        # Normalize projections
        visual_proj = F.normalize(visual_proj, dim=-1)
        text_proj = F.normalize(text_proj, dim=-1)
        
        # Include audio if provided
        audio_proj = None
        if audio_features is not None:
            if audio_features.dim() > 2:
                audio_global = audio_features.mean(dim=1)
            else:
                audio_global = audio_features
                
            audio_proj = self.audio_projection(audio_global)
            audio_proj = F.normalize(audio_proj, dim=-1)
        
        # Compute similarity matrix
        batch_size = visual_proj.shape[0]
        similarity_vt = torch.matmul(visual_proj, text_proj.transpose(0, 1)) / self.temperature
        
        # Contrastive loss with in-batch negatives
        labels = torch.arange(batch_size, device=visual_proj.device)
        loss_vt = F.cross_entropy(similarity_vt, labels)
        loss_tv = F.cross_entropy(similarity_vt.transpose(0, 1), labels)
        contrastive_loss = (loss_vt + loss_tv) / 2.0
        
        results = {
            "contrastive_loss": contrastive_loss,
            "similarity_matrix": similarity_vt
        }
        
        if return_projections:
            results["visual_projection"] = visual_proj
            results["text_projection"] = text_proj
            if audio_proj is not None:
                results["audio_projection"] = audio_proj
        
        return results


class CrossModalTransferModule(nn.Module):
    """
    Enables knowledge transfer between modalities for improved representations.
    
    This module implements cross-modal transfer learning techniques to enhance
    representations of each modality using knowledge from other modalities.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        modalities: List[str] = ["vision", "text", "audio"]
    ):
        """
        Initialize the cross-modal transfer module.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            modalities: List of supported modalities
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.modalities = modalities
        
        # Cross-modal alignment layer
        self.alignment_layer = CrossModalAlignmentLayer(
            embedding_dim=embedding_dim,
            projection_dim=embedding_dim // 2,
            dropout=dropout
        )
        
        # Cross-modal transformer layers
        self.cross_attn_layers = nn.ModuleDict()
        for source in modalities:
            for target in modalities:
                if source != target:
                    key = f"{source}_to_{target}"
                    self.cross_attn_layers[key] = nn.MultiheadAttention(
                        embed_dim=embedding_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_first=True
                    )
        
        # Feature enhancement layers
        self.enhancement_layers = nn.ModuleDict()
        for modality in modalities:
            self.enhancement_layers[modality] = nn.Sequential(
                nn.Linear(embedding_dim * len(modalities), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
    
    def forward(
        self,
        features_dict: Dict[str, torch.Tensor],
        attention_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Apply cross-modal transfer learning to enhance representations.
        
        Args:
            features_dict: Dictionary of features for each modality
                {modality_name: features_tensor}
            attention_masks: Optional dictionary of attention masks
                {modality_name: attention_mask}
                
        Returns:
            Tuple containing:
                - Dictionary of enhanced features for each modality
                - Dictionary of metadata including alignment metrics
        """
        batch_size = next(iter(features_dict.values())).shape[0]
        enhanced_features = {}
        cross_modal_attn_outputs = {}
        
        # Compute cross-modal attention for all modality pairs
        for source in self.modalities:
            if source not in features_dict:
                continue
                
            source_features = features_dict[source]
            source_mask = None if attention_masks is None else attention_masks.get(source)
            
            # Store attended representations from other modalities
            other_modality_contexts = []
            other_modality_contexts.append(source_features)  # Include original features
            
            for target in self.modalities:
                if target == source or target not in features_dict:
                    continue
                    
                target_features = features_dict[target]
                target_mask = None if attention_masks is None else attention_masks.get(target)
                
                # Apply cross-attention from source to target
                key = f"{source}_to_{target}"
                if key in self.cross_attn_layers:
                    attended_features, attn_weights = self.cross_attn_layers[key](
                        query=source_features,
                        key=target_features,
                        value=target_features,
                        key_padding_mask=target_mask
                    )
                    
                    cross_modal_attn_outputs[key] = attended_features
                    other_modality_contexts.append(attended_features)
            
            # Combine information from all modalities
            if len(other_modality_contexts) > 1:
                combined = torch.cat(other_modality_contexts, dim=-1)
                enhanced = self.enhancement_layers[source](combined)
                enhanced_features[source] = enhanced
            else:
                enhanced_features[source] = source_features
        
        # Compute alignment metrics if multiple modalities are present
        alignment_metrics = {}
        if "vision" in features_dict and "text" in features_dict:
            visual_features = features_dict["vision"]
            text_features = features_dict["text"]
            audio_features = features_dict.get("audio")
            
            alignment_results = self.alignment_layer(
                visual_features=visual_features,
                text_features=text_features,
                audio_features=audio_features,
                return_projections=True
            )
            
            alignment_metrics.update(alignment_results)
        
        return enhanced_features, alignment_metrics