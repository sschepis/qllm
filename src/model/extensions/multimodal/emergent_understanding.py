"""
Emergent Cross-Modal Understanding Module.

This module enables the model to develop emergent understanding of relationships
between different modalities without explicit supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union
import math


class EmergentRepresentationLayer(nn.Module):
    """
    Learns emergent representations across modalities through 
    self-organization and unsupervised alignment.
    
    This module enables the model to discover latent relationships and 
    patterns across modalities even when explicit supervision is not available.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        latent_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 0.07
    ):
        """
        Initialize the emergent representation layer.
        
        Args:
            embedding_dim: Dimension of input embeddings
            latent_dim: Dimension of latent representations
            num_heads: Number of attention heads
            dropout: Dropout probability
            temperature: Temperature for contrast learning
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.temperature = temperature
        
        # Latent projections for each modality
        self.modality_projections = nn.ModuleDict({
            "vision": nn.Sequential(
                nn.Linear(embedding_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim)
            ),
            "text": nn.Sequential(
                nn.Linear(embedding_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim)
            ),
            "audio": nn.Sequential(
                nn.Linear(embedding_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim)
            )
        })
        
        # Self-organizing latent space with cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Latent space transformation
        self.latent_transform = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Prototype vectors for emergent concepts
        self.num_prototypes = 64
        self.prototype_vectors = nn.Parameter(
            torch.randn(self.num_prototypes, latent_dim)
        )
        nn.init.orthogonal_(self.prototype_vectors)
        
        # Back-projection to original embedding dimensions
        self.back_projections = nn.ModuleDict({
            "vision": nn.Linear(latent_dim, embedding_dim),
            "text": nn.Linear(latent_dim, embedding_dim),
            "audio": nn.Linear(latent_dim, embedding_dim)
        })
    
    def compute_prototype_assignments(self, latent_vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute soft assignments of latent vectors to prototypes.
        
        Args:
            latent_vectors: Latent representations [batch_size, seq_len, latent_dim]
            
        Returns:
            Assignment probabilities [batch_size, seq_len, num_prototypes]
        """
        # Normalize prototypes and latent vectors
        normalized_prototypes = F.normalize(self.prototype_vectors, dim=-1)
        normalized_latents = F.normalize(latent_vectors, dim=-1)
        
        # Compute cosine similarity
        logits = torch.matmul(normalized_latents, normalized_prototypes.transpose(0, 1))
        logits = logits / self.temperature
        
        # Compute soft assignments
        assignments = F.softmax(logits, dim=-1)
        
        return assignments
    
    def forward(
        self,
        features_dict: Dict[str, torch.Tensor],
        attention_masks: Optional[Dict[str, torch.Tensor]] = None,
        return_latents: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Process features to develop emergent cross-modal understanding.
        
        Args:
            features_dict: Dictionary of features for each modality
                {modality_name: features_tensor}
            attention_masks: Optional dictionary of attention masks
                {modality_name: attention_mask}
            return_latents: Whether to return latent representations
                
        Returns:
            Tuple containing:
                - Dictionary of enhanced features with emergent understanding
                - Dictionary of metadata including prototype assignments
        """
        batch_size = next(iter(features_dict.values())).shape[0]
        
        # Project each modality to latent space
        latent_dict = {}
        for modality, features in features_dict.items():
            if modality in self.modality_projections:
                # Handle sequence vs non-sequence input
                if features.dim() > 2:
                    orig_shape = features.shape
                    # Reshape to [batch_size * seq_len, embedding_dim]
                    reshaped = features.view(-1, features.shape[-1])
                    # Project to latent space
                    latent = self.modality_projections[modality](reshaped)
                    # Reshape back to [batch_size, seq_len, latent_dim]
                    latent = latent.view(orig_shape[0], orig_shape[1], self.latent_dim)
                else:
                    latent = self.modality_projections[modality](features)
                
                latent_dict[modality] = latent
        
        if not latent_dict:
            return features_dict, {}
        
        # Concatenate all modality latent representations
        all_modalities = list(latent_dict.keys())
        all_latents = torch.cat([latent_dict[m] for m in all_modalities], dim=1)
        all_latents_lens = [latent_dict[m].shape[1] for m in all_modalities]
        
        # Create a modality tracking tensor
        modality_indices = []
        for i, modality in enumerate(all_modalities):
            modality_indices.extend([i] * latent_dict[modality].shape[1])
        modality_indices = torch.tensor(modality_indices, device=all_latents.device)
        
        # Apply cross-modal attention for self-organization
        attended_latents, attn_weights = self.cross_modal_attention(
            query=all_latents,
            key=all_latents,
            value=all_latents
        )
        
        # Apply latent transformation
        transformed_latents = self.latent_transform(attended_latents)
        
        # Compute prototype assignments
        prototype_assignments = self.compute_prototype_assignments(transformed_latents)
        
        # Get top-k prototypes for interpretation
        k = min(5, self.num_prototypes)
        top_k_values, top_k_indices = torch.topk(prototype_assignments, k, dim=-1)
        
        # Split back to individual modalities
        transformed_latent_dict = {}
        enhanced_features_dict = {}
        start_idx = 0
        
        for i, modality in enumerate(all_modalities):
            seq_len = all_latents_lens[i]
            modality_latents = transformed_latents[:, start_idx:start_idx+seq_len, :]
            transformed_latent_dict[modality] = modality_latents
            
            # Project back to original embedding space
            if modality in self.back_projections:
                enhanced_features = self.back_projections[modality](modality_latents)
                # Add residual connection with original features
                enhanced_features = enhanced_features + features_dict[modality]
                enhanced_features_dict[modality] = enhanced_features
            
            start_idx += seq_len
        
        # Prepare metadata including emergent properties
        metadata = {
            "prototype_assignments": prototype_assignments,
            "top_prototypes": {
                "values": top_k_values,
                "indices": top_k_indices
            },
            "cross_modal_attention": attn_weights
        }
        
        if return_latents:
            metadata["latent_representations"] = transformed_latent_dict
        
        return enhanced_features_dict, metadata


class CrossModalClusteringModule(nn.Module):
    """
    Module for discovering and leveraging emergent cross-modal clusters.
    
    This module implements unsupervised discovery of conceptual clusters
    that exist across modalities, enabling deeper semantic understanding.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_clusters: int = 32,
        latent_dim: int = 256,
        cluster_temp: float = 0.1,
        update_momentum: float = 0.99
    ):
        """
        Initialize the cross-modal clustering module.
        
        Args:
            embedding_dim: Dimension of input embeddings
            num_clusters: Number of cross-modal clusters
            latent_dim: Dimension of latent space
            cluster_temp: Temperature for soft cluster assignments
            update_momentum: Momentum coefficient for updating cluster centroids
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim
        self.cluster_temp = cluster_temp
        self.update_momentum = update_momentum
        
        # Projection to latent space
        self.latent_projection = nn.Sequential(
            nn.Linear(embedding_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Cluster centroids in latent space
        self.register_buffer(
            "cluster_centroids", 
            torch.randn(num_clusters, latent_dim)
        )
        self.cluster_centroids = F.normalize(self.cluster_centroids, dim=1)
        
        # Cluster assignment statistics for interpretation
        self.register_buffer(
            "cluster_counts", 
            torch.zeros(num_clusters)
        )
        
        # Feature enhancement based on cluster assignments
        self.cluster_enhancement = nn.Sequential(
            nn.Linear(latent_dim + num_clusters, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def compute_cluster_assignments(self, latent_vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute soft assignments to clusters.
        
        Args:
            latent_vectors: Latent representations [batch_size, seq_len, latent_dim]
            
        Returns:
            Cluster assignment probabilities [batch_size, seq_len, num_clusters]
        """
        # Normalize vectors for cosine similarity
        normalized_vectors = F.normalize(latent_vectors, dim=-1)
        normalized_centroids = F.normalize(self.cluster_centroids, dim=-1)
        
        # Compute cosine similarity with centroids
        similarity = torch.matmul(normalized_vectors, normalized_centroids.transpose(0, 1))
        similarity = similarity / self.cluster_temp
        
        # Compute soft assignments
        assignments = F.softmax(similarity, dim=-1)
        
        return assignments
    
    def update_centroids(self, latent_vectors: torch.Tensor, assignments: torch.Tensor) -> None:
        """
        Update cluster centroids based on new assignments (with momentum).
        
        Args:
            latent_vectors: Latent representations [batch_size, seq_len, latent_dim]
            assignments: Cluster assignments [batch_size, seq_len, num_clusters]
        """
        if not self.training:
            return
            
        batch_size, seq_len, _ = latent_vectors.shape
        flattened_vectors = latent_vectors.reshape(-1, self.latent_dim)
        flattened_assignments = assignments.reshape(-1, self.num_clusters)
        
        # Sum assignments per cluster
        assignment_sum = flattened_assignments.sum(0)
        self.cluster_counts = self.update_momentum * self.cluster_counts + (1 - self.update_momentum) * assignment_sum
        
        # Compute new centroids with weighted sum
        assignment_weighted_sum = torch.matmul(
            flattened_assignments.transpose(0, 1),
            flattened_vectors
        )
        
        # Update with momentum
        new_centroids = assignment_weighted_sum / (assignment_sum.unsqueeze(1) + 1e-8)
        self.cluster_centroids = self.update_momentum * self.cluster_centroids + (1 - self.update_momentum) * new_centroids
    
    def forward(
        self,
        features_dict: Dict[str, torch.Tensor],
        attention_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Process features through cross-modal clustering.
        
        Args:
            features_dict: Dictionary of features for each modality
                {modality_name: features_tensor}
            attention_masks: Optional dictionary of attention masks
                {modality_name: attention_mask}
                
        Returns:
            Tuple containing:
                - Dictionary of enhanced features
                - Dictionary of metadata including cluster assignments
        """
        enhanced_features_dict = {}
        all_cluster_assignments = {}
        all_latent_vectors = {}
        
        # Process each modality
        for modality, features in features_dict.items():
            # Project to latent space
            if features.dim() > 2:
                batch_size, seq_len, _ = features.shape
                latent_vectors = self.latent_projection(features)
            else:
                batch_size = features.shape[0]
                latent_vectors = self.latent_projection(features.unsqueeze(1))
                seq_len = 1
            
            # Compute cluster assignments
            cluster_assignments = self.compute_cluster_assignments(latent_vectors)
            all_cluster_assignments[modality] = cluster_assignments
            all_latent_vectors[modality] = latent_vectors
            
            # Update centroids during training
            if self.training:
                self.update_centroids(latent_vectors, cluster_assignments)
            
            # Enhance features with cluster information
            concat_features = torch.cat([latent_vectors, cluster_assignments], dim=-1)
            enhanced = self.cluster_enhancement(concat_features)
            
            # Add residual connection
            if seq_len == 1 and features.dim() == 2:
                enhanced = enhanced.squeeze(1) + features
                enhanced_features_dict[modality] = enhanced
            else:
                enhanced = enhanced + features
                enhanced_features_dict[modality] = enhanced
        
        # Find cross-modal cluster correspondences
        cross_modal_correspondences = {}
        modalities = list(all_cluster_assignments.keys())
        
        if len(modalities) > 1:
            for i in range(len(modalities)):
                for j in range(i+1, len(modalities)):
                    mod_i, mod_j = modalities[i], modalities[j]
                    
                    # Compute correlation between cluster assignments
                    assign_i = all_cluster_assignments[mod_i].mean(dim=1)  # [batch, clusters]
                    assign_j = all_cluster_assignments[mod_j].mean(dim=1)  # [batch, clusters]
                    
                    # Compute cluster correspondences as correlation matrix
                    # [clusters_i, clusters_j]
                    correspondence = torch.matmul(assign_i.transpose(0, 1), assign_j)
                    correspondence = correspondence / (torch.norm(correspondence, dim=0, keepdim=True) + 1e-8)
                    correspondence = correspondence / (torch.norm(correspondence, dim=1, keepdim=True) + 1e-8)
                    
                    cross_modal_correspondences[f"{mod_i}_{mod_j}"] = correspondence
        
        # Prepare metadata
        metadata = {
            "cluster_assignments": all_cluster_assignments,
            "cross_modal_correspondences": cross_modal_correspondences,
            "cluster_counts": self.cluster_counts
        }
        
        return enhanced_features_dict, metadata


class EmergentUnderstandingModule(nn.Module):
    """
    Module for emergent cross-modal understanding.
    
    This module develops emergent, higher-level understanding of relationships
    between different modalities without explicit supervision.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        modalities: List[str] = ["vision", "text", "audio"],
        latent_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize the emergent understanding module.
        
        Args:
            embedding_dim: Dimension of input embeddings
            modalities: List of supported modalities
            latent_dim: Dimension of latent space
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.modalities = modalities
        self.latent_dim = latent_dim
        
        # Emergent representation learning
        self.emergent_representation = EmergentRepresentationLayer(
            embedding_dim=embedding_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cross-modal clustering
        self.cross_modal_clustering = CrossModalClusteringModule(
            embedding_dim=embedding_dim,
            latent_dim=latent_dim,
            num_clusters=32
        )
        
        # Final integration layer
        self.integration_layer = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim)
            ) for modality in modalities
        })
    
    def forward(
        self,
        features_dict: Dict[str, torch.Tensor],
        attention_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Process features through emergent understanding.
        
        Args:
            features_dict: Dictionary of features for each modality
                {modality_name: features_tensor}
            attention_masks: Optional dictionary of attention masks
                {modality_name: attention_mask}
                
        Returns:
            Tuple containing:
                - Dictionary of enhanced features with emergent understanding
                - Dictionary of metadata including emergent properties
        """
        # Apply emergent representation learning
        emergent_features, emergent_metadata = self.emergent_representation(
            features_dict, 
            attention_masks,
            return_latents=True
        )
        
        # Apply cross-modal clustering
        clustered_features, cluster_metadata = self.cross_modal_clustering(
            features_dict, 
            attention_masks
        )
        
        # Combine and integrate the two approaches
        integrated_features = {}
        
        for modality in features_dict.keys():
            if modality not in self.integration_layer:
                continue
                
            # Concatenate features from both approaches
            emergent_feat = emergent_features.get(modality, features_dict[modality])
            clustered_feat = clustered_features.get(modality, features_dict[modality])
            
            # Integrate features
            concat_features = torch.cat([emergent_feat, clustered_feat], dim=-1)
            integrated = self.integration_layer[modality](concat_features)
            
            # Add residual connection
            integrated = integrated + features_dict[modality]
            integrated_features[modality] = integrated
        
        # Combine metadata
        metadata = {}
        metadata.update({f"emergent_{k}": v for k, v in emergent_metadata.items()})
        metadata.update({f"cluster_{k}": v for k, v in cluster_metadata.items()})
        
        # Add integration information
        metadata["integrated_modalities"] = list(integrated_features.keys())
        
        return integrated_features, metadata