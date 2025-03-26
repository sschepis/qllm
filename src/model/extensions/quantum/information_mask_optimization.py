"""
Information-Theoretic Mask Optimization Module.

This module enables information-theoretic optimization of quantum masks,
improving efficiency and performance based on information content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union, Set
import math
import numpy as np

from .quantum_types import MaskType, PatternType
from .mask_generators import apply_mask_to_parameter
from .quantum_patterns import generate_quantum_pattern


class InformationContentAnalyzer(nn.Module):
    """
    Analyzes information content in neural network parameters.
    
    This module quantifies the information content in parameters to guide
    mask optimization based on information theory principles.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        num_quantization_levels: int = 16,
        use_fisher_information: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize the information content analyzer.
        
        Args:
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden layers
            num_quantization_levels: Number of levels for quantization
            use_fisher_information: Whether to use Fisher information
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_quantization_levels = num_quantization_levels
        self.use_fisher_information = use_fisher_information
        
        # Information density predictor
        self.density_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Entropy estimation network
        self.entropy_estimator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.ReLU()
        )
        
        # Fisher information approximation
        if use_fisher_information:
            self.fisher_approximator = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()  # Ensure positive output
            )
    
    def compute_entropy(
        self,
        parameter: torch.Tensor
    ) -> float:
        """
        Compute entropy of a parameter tensor.
        
        Args:
            parameter: Parameter tensor
            
        Returns:
            Entropy value
        """
        # Flatten parameter
        flat_param = parameter.reshape(-1)
        
        # Quantize to estimate probability distribution
        if flat_param.shape[0] > 1:
            min_val = flat_param.min().item()
            max_val = flat_param.max().item()
            
            # Handle case where min equals max
            if min_val == max_val:
                return 0.0  # Zero entropy for constant values
                
            # Create quantization bins
            bins = torch.linspace(
                min_val, max_val, self.num_quantization_levels + 1,
                device=flat_param.device
            )
            
            # Quantize values
            quantized = torch.bucketize(flat_param, bins) - 1
            quantized = torch.clamp(quantized, 0, self.num_quantization_levels - 1)
            
            # Compute histogram
            counts = torch.bincount(
                quantized,
                minlength=self.num_quantization_levels
            )
            
            # Compute probabilities
            probabilities = counts.float() / counts.sum()
            
            # Remove zeros
            probabilities = probabilities[probabilities > 0]
            
            # Compute entropy
            entropy = -torch.sum(probabilities * torch.log2(probabilities)).item()
            
            # Normalize by maximum possible entropy
            max_entropy = math.log2(self.num_quantization_levels)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            # Single element has zero entropy
            normalized_entropy = 0.0
        
        return normalized_entropy
    
    def compute_fisher_information(
        self,
        parameter: torch.Tensor,
        gradient: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Approximate Fisher information for a parameter.
        
        Args:
            parameter: Parameter tensor
            gradient: Optional gradient tensor
            
        Returns:
            Fisher information tensor
        """
        # Flatten parameter
        flat_param = parameter.reshape(-1)
        
        # Extract parameter statistics as features
        if flat_param.shape[0] > 1:
            mean = flat_param.mean()
            std = flat_param.std()
            percentiles = torch.tensor([
                torch.quantile(flat_param, 0.1),
                torch.quantile(flat_param, 0.5),
                torch.quantile(flat_param, 0.9)
            ], device=flat_param.device)
            max_val = flat_param.max()
            min_val = flat_param.min()
            
            # Create feature vector
            features = torch.cat([
                torch.tensor([mean, std, max_val, min_val], device=flat_param.device),
                percentiles
            ])
        else:
            # Single element case
            features = torch.cat([
                flat_param,
                torch.zeros(6, device=flat_param.device)
            ])
        
        # Pad to embedding dimension
        if features.shape[0] < self.embedding_dim:
            padding = torch.zeros(
                self.embedding_dim - features.shape[0],
                device=features.device
            )
            features = torch.cat([features, padding])
        else:
            features = features[:self.embedding_dim]
        
        # Use gradient information if available
        if gradient is not None:
            flat_grad = gradient.reshape(-1)
            
            # Compute Fisher approximation
            if flat_grad.shape[0] == flat_param.shape[0]:
                fisher_diag = flat_grad ** 2
                
                # Take statistics of Fisher diagonal
                fisher_mean = fisher_diag.mean().unsqueeze(0)
                fisher_std = fisher_diag.std().unsqueeze(0)
                
                features = torch.cat([
                    features[:self.embedding_dim-2],
                    fisher_mean,
                    fisher_std
                ])
        
        # Get fisher information from approximator
        fisher_info = self.fisher_approximator(features.unsqueeze(0)).squeeze()
        
        return fisher_info
    
    def estimate_information_density(
        self,
        parameter: torch.Tensor,
        gradient: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Estimate information density of a parameter.
        
        Args:
            parameter: Parameter tensor
            gradient: Optional gradient tensor
            
        Returns:
            Information density tensor
        """
        # Flatten parameter
        flat_param = parameter.reshape(-1)
        
        # Extract parameter statistics as features
        if flat_param.shape[0] > 1:
            mean = flat_param.mean()
            std = flat_param.std()
            percentiles = torch.tensor([
                torch.quantile(flat_param, 0.1),
                torch.quantile(flat_param, 0.5),
                torch.quantile(flat_param, 0.9)
            ], device=flat_param.device)
            max_val = flat_param.max()
            min_val = flat_param.min()
            
            # Create feature vector
            features = torch.cat([
                torch.tensor([mean, std, max_val, min_val], device=flat_param.device),
                percentiles
            ])
        else:
            # Single element case
            features = torch.cat([
                flat_param,
                torch.zeros(6, device=flat_param.device)
            ])
        
        # Compute entropy
        entropy = self.compute_entropy(parameter)
        entropy_tensor = torch.tensor([entropy], device=features.device)
        
        # Add entropy to features
        features = torch.cat([features, entropy_tensor])
        
        # Pad to embedding dimension
        if features.shape[0] < self.embedding_dim:
            padding = torch.zeros(
                self.embedding_dim - features.shape[0],
                device=features.device
            )
            features = torch.cat([features, padding])
        else:
            features = features[:self.embedding_dim]
        
        # Predict information density
        density = self.density_predictor(features.unsqueeze(0)).squeeze()
        
        return density
    
    def forward(
        self,
        parameters: Dict[str, torch.Tensor],
        gradients: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Analyze information content in parameters.
        
        Args:
            parameters: Dictionary of parameters
            gradients: Optional dictionary of gradients
            
        Returns:
            Tuple containing:
                - Dictionary of information densities
                - Dictionary of analysis metadata
        """
        information_densities = {}
        metadata = {
            "entropy": {},
            "fisher_information": {} if self.use_fisher_information else None
        }
        
        for name, param in parameters.items():
            # Skip non-tensor parameters
            if not isinstance(param, torch.Tensor) or param.dim() == 0:
                continue
                
            # Get gradient if available
            gradient = None
            if gradients is not None and name in gradients:
                gradient = gradients[name]
            
            # Compute entropy
            entropy = self.compute_entropy(param)
            metadata["entropy"][name] = entropy
            
            # Compute fisher information if enabled and gradient available
            if self.use_fisher_information:
                fisher_info = self.compute_fisher_information(param, gradient)
                metadata["fisher_information"][name] = fisher_info.item()
            
            # Estimate information density
            density = self.estimate_information_density(param, gradient)
            information_densities[name] = density
        
        return information_densities, metadata


class InformationGuidedMaskGenerator(nn.Module):
    """
    Generates masks guided by information content.
    
    This module creates quantum-inspired masks that are optimized based on
    the information content of parameters, preserving important information.
    """
    
    def __init__(
        self,
        base_sparsity: float = 0.8,
        mask_type: MaskType = MaskType.BINARY,
        adapt_to_information: bool = True,
        sparsity_range: Tuple[float, float] = (0.5, 0.95),
        pattern_selection: str = "adaptive"
    ):
        """
        Initialize the information-guided mask generator.
        
        Args:
            base_sparsity: Base sparsity level
            mask_type: Type of mask (binary or continuous)
            adapt_to_information: Whether to adapt sparsity to information
            sparsity_range: Range for adaptive sparsity
            pattern_selection: Method for selecting patterns
                Options: "adaptive", "fixed", "mixed"
        """
        super().__init__()
        self.base_sparsity = base_sparsity
        self.mask_type = mask_type
        self.adapt_to_information = adapt_to_information
        self.sparsity_range = sparsity_range
        self.pattern_selection = pattern_selection
        
        # Patterns to choose from
        self.patterns = [
            PatternType.HARMONIC,
            PatternType.HILBERT,
            PatternType.CYCLIC,
            PatternType.PRIME,
            PatternType.ORTHOGONAL
        ]
        
        # For fixed pattern selection
        if pattern_selection == "fixed":
            self.default_pattern = PatternType.HARMONIC
    
    def compute_adaptive_sparsity(
        self,
        information_density: float
    ) -> float:
        """
        Compute adaptive sparsity based on information density.
        
        Args:
            information_density: Information density value
            
        Returns:
            Adaptive sparsity value
        """
        if not self.adapt_to_information:
            return self.base_sparsity
            
        # Scale sparsity inversely with information density
        # Higher density -> lower sparsity (keep more information)
        min_sparsity, max_sparsity = self.sparsity_range
        
        # Linear interpolation
        sparsity = max_sparsity - information_density * (max_sparsity - min_sparsity)
        
        return sparsity
    
    def select_pattern_type(
        self,
        information_density: float,
        parameter_shape: Tuple[int, ...]
    ) -> PatternType:
        """
        Select appropriate pattern type based on information characteristics.
        
        Args:
            information_density: Information density value
            parameter_shape: Shape of the parameter tensor
            
        Returns:
            Selected pattern type
        """
        if self.pattern_selection == "fixed":
            return self.default_pattern
            
        # Adaptive pattern selection
        if self.pattern_selection == "adaptive":
            # Use different patterns based on information density
            if information_density < 0.2:
                # Low information: Use simpler patterns
                return PatternType.CYCLIC
            elif information_density < 0.5:
                # Medium-low information: Use harmonic patterns
                return PatternType.HARMONIC
            elif information_density < 0.8:
                # Medium-high information: Use Hilbert patterns
                return PatternType.HILBERT
            else:
                # High information: Use more complex patterns
                return PatternType.PRIME
        
        # Mixed pattern selection: random choice weighted by information
        pattern_weights = [
            0.3,  # HARMONIC
            0.2 + information_density * 0.3,  # HILBERT (higher for high info)
            0.2 - information_density * 0.1,  # CYCLIC (lower for high info)
            0.2 + information_density * 0.2,  # PRIME (higher for high info)
            0.1 - information_density * 0.05   # ORTHOGONAL (lower for high info)
        ]
        
        # Ensure valid probability distribution
        pattern_weights = [max(0.01, w) for w in pattern_weights]
        total = sum(pattern_weights)
        pattern_weights = [w / total for w in pattern_weights]
        
        # Weighted random choice
        choice = np.random.choice(len(self.patterns), p=pattern_weights)
        
        return self.patterns[choice]
    
    def generate_information_guided_mask(
        self,
        parameter: torch.Tensor,
        information_density: float
    ) -> torch.Tensor:
        """
        Generate a mask optimized for information preservation.
        
        Args:
            parameter: Parameter tensor
            information_density: Information density value
            
        Returns:
            Optimized mask tensor
        """
        # Ensure parameter has at least 2 dimensions
        if parameter.dim() < 2:
            # Convert to 2D
            original_shape = parameter.shape
            parameter = parameter.reshape(1, -1)
        else:
            original_shape = parameter.shape
        
        # Compute adaptive sparsity
        sparsity = self.compute_adaptive_sparsity(information_density)
        
        # Select pattern type
        pattern_type = self.select_pattern_type(information_density, parameter.shape)
        
        # Get pattern parameters
        pattern_params = {
            "frequency_factor": 1.0 + information_density * 9.0,  # [1, 10]
            "phase_shift": information_density * math.pi * 2,  # [0, 2π]
            "amplitude_factor": 0.5 + information_density * 0.5,  # [0.5, 1.0]
            "decay_rate": max(0.1, 0.5 - information_density * 0.4),  # [0.1, 0.5], inverse to info
            "sparsity": sparsity,
            "symmetry_factor": 0.5 + information_density * 0.5  # [0.5, 1.0]
        }
        
        # Generate mask
        mask = generate_quantum_pattern(
            parameter.shape,
            pattern_type,
            self.mask_type,
            **pattern_params
        )
        
        # Ensure mask has same type as parameter
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, device=parameter.device, dtype=parameter.dtype)
        
        # Reshape to original shape if needed
        if mask.shape != original_shape:
            mask = mask.reshape(original_shape)
        
        return mask
    
    def forward(
        self,
        parameters: Dict[str, torch.Tensor],
        information_densities: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Generate information-guided masks for parameters.
        
        Args:
            parameters: Dictionary of parameters
            information_densities: Dictionary of information densities
            
        Returns:
            Tuple containing:
                - Dictionary of optimized masks
                - Dictionary of mask generation metadata
        """
        optimized_masks = {}
        metadata = {
            "sparsity_levels": {},
            "pattern_types": {},
            "information_densities": {}
        }
        
        for name, param in parameters.items():
            # Skip non-tensor parameters
            if not isinstance(param, torch.Tensor) or param.dim() == 0:
                continue
                
            # Get information density
            if name not in information_densities:
                density = 0.5  # Default
            else:
                density = information_densities[name].item()
            
            # Generate mask
            mask = self.generate_information_guided_mask(param, density)
            
            # Store mask
            optimized_masks[name] = mask
            
            # Compute actual sparsity
            sparsity = 1.0 - (mask > 0).float().mean().item()
            
            # Store metadata
            metadata["sparsity_levels"][name] = sparsity
            metadata["pattern_types"][name] = self.select_pattern_type(density, param.shape)
            metadata["information_densities"][name] = density
        
        return optimized_masks, metadata


class MutualInformationOptimizer(nn.Module):
    """
    Optimizes masks based on mutual information between parameters.
    
    This module enhances mask optimization by considering mutual information
    between parameters, preserving important parameter relationships.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        num_clusters: int = 8,
        min_mutual_information: float = 0.2,
        dropout: float = 0.1
    ):
        """
        Initialize the mutual information optimizer.
        
        Args:
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden layers
            num_clusters: Number of parameter clusters
            min_mutual_information: Minimum mutual information threshold
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.min_mutual_information = min_mutual_information
        
        # Parameter embedding network
        self.param_embedding = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Mutual information estimator
        self.mi_estimator = nn.Bilinear(
            embedding_dim, embedding_dim, 1
        )
        
        # Cluster assignment network
        self.cluster_assignment = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_clusters),
            nn.Softmax(dim=-1)
        )
        
        # Mask coordination network
        self.mask_coordinator = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def compute_parameter_embedding(
        self,
        parameter: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute embedding for a parameter tensor.
        
        Args:
            parameter: Parameter tensor
            
        Returns:
            Parameter embedding
        """
        # Flatten parameter
        flat_param = parameter.reshape(-1)
        
        # Extract statistics
        if flat_param.shape[0] > 1:
            mean = flat_param.mean()
            std = flat_param.std()
            percentiles = torch.tensor([
                torch.quantile(flat_param, 0.1),
                torch.quantile(flat_param, 0.5),
                torch.quantile(flat_param, 0.9)
            ], device=flat_param.device)
            max_val = flat_param.max()
            min_val = flat_param.min()
            
            # Create feature vector
            features = torch.cat([
                torch.tensor([mean, std, max_val, min_val], device=flat_param.device),
                percentiles
            ])
        else:
            # Single element case
            features = torch.cat([
                flat_param,
                torch.zeros(6, device=flat_param.device)
            ])
        
        # Pad to embedding dimension
        if features.shape[0] < self.embedding_dim:
            padding = torch.zeros(
                self.embedding_dim - features.shape[0],
                device=features.device
            )
            features = torch.cat([features, padding])
        else:
            features = features[:self.embedding_dim]
        
        # Apply embedding network
        embedding = self.param_embedding(features.unsqueeze(0)).squeeze(0)
        
        return embedding
    
    def estimate_mutual_information(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> float:
        """
        Estimate mutual information between parameter embeddings.
        
        Args:
            embedding1: First parameter embedding
            embedding2: Second parameter embedding
            
        Returns:
            Estimated mutual information
        """
        # Use the bilinear MI estimator
        mi = self.mi_estimator(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0)
        ).squeeze().item()
        
        # Scale to [0, 1]
        mi = torch.sigmoid(torch.tensor(mi)).item()
        
        return mi
    
    def cluster_parameters(
        self,
        parameter_embeddings: Dict[str, torch.Tensor]
    ) -> Dict[int, List[str]]:
        """
        Cluster parameters based on embeddings.
        
        Args:
            parameter_embeddings: Dictionary of parameter embeddings
            
        Returns:
            Dictionary mapping cluster IDs to parameter names
        """
        clusters = {i: [] for i in range(self.num_clusters)}
        
        for name, embedding in parameter_embeddings.items():
            # Get cluster assignments
            assignments = self.cluster_assignment(embedding.unsqueeze(0)).squeeze(0)
            
            # Assign to highest probability cluster
            cluster_id = torch.argmax(assignments).item()
            clusters[cluster_id].append(name)
        
        return clusters
    
    def optimize_masks_for_mutual_information(
        self,
        parameter_embeddings: Dict[str, torch.Tensor],
        initial_masks: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Optimize masks to preserve mutual information.
        
        Args:
            parameter_embeddings: Dictionary of parameter embeddings
            initial_masks: Dictionary of initial masks
            
        Returns:
            Tuple containing:
                - Dictionary of optimized masks
                - Dictionary of optimization metadata
        """
        # Cluster parameters
        clusters = self.cluster_parameters(parameter_embeddings)
        
        # Compute pairwise mutual information within clusters
        mutual_information = {}
        parameter_pairs = []
        
        for cluster_id, param_names in clusters.items():
            for i, name1 in enumerate(param_names):
                for j, name2 in enumerate(param_names[i+1:], i+1):
                    # Compute mutual information
                    mi = self.estimate_mutual_information(
                        parameter_embeddings[name1],
                        parameter_embeddings[name2]
                    )
                    
                    # Store if above threshold
                    if mi >= self.min_mutual_information:
                        pair_key = (name1, name2)
                        mutual_information[pair_key] = mi
                        parameter_pairs.append(pair_key)
        
        # Coordinate masks for high-MI parameter pairs
        optimized_masks = {k: v.clone() for k, v in initial_masks.items()}
        
        for name1, name2 in parameter_pairs:
            if name1 not in initial_masks or name2 not in initial_masks:
                continue
                
            # Get original masks
            mask1 = initial_masks[name1]
            mask2 = initial_masks[name2]
            
            # Compute coordination factor
            combined = torch.cat([
                parameter_embeddings[name1],
                parameter_embeddings[name2]
            ])
            
            coord_factor = self.mask_coordinator(combined.unsqueeze(0)).squeeze().item()
            
            # Skip if coordination factor is too low
            if coord_factor < 0.5:
                continue
                
            # Get mask shapes
            shape1 = mask1.shape
            shape2 = mask2.shape
            
            # Flatten masks
            flat_mask1 = mask1.reshape(-1)
            flat_mask2 = mask2.reshape(-1)
            
            # Determine shorter mask
            min_len = min(flat_mask1.shape[0], flat_mask2.shape[0])
            
            # Coordinate mask patterns
            # Simple approach: align mask values for a portion of the masks
            coord_portion = int(min_len * coord_factor)
            
            if coord_portion > 0:
                # Create index mapping for aligned portion
                if flat_mask1.shape[0] > flat_mask2.shape[0]:
                    # Mask1 is larger
                    ratio = flat_mask1.shape[0] / flat_mask2.shape[0]
                    indices = (torch.arange(coord_portion, device=mask1.device) * ratio).long()
                    
                    # Ensure indices are within bounds
                    indices = torch.clamp(indices, 0, flat_mask1.shape[0] - 1)
                    
                    # Align mask1 to mask2
                    flat_mask1_new = flat_mask1.clone()
                    flat_mask1_new[:coord_portion] = flat_mask2[:coord_portion]
                    
                    # Update mask
                    optimized_masks[name1] = flat_mask1_new.reshape(shape1)
                else:
                    # Mask2 is larger or equal
                    ratio = flat_mask2.shape[0] / flat_mask1.shape[0]
                    indices = (torch.arange(coord_portion, device=mask1.device) * ratio).long()
                    
                    # Ensure indices are within bounds
                    indices = torch.clamp(indices, 0, flat_mask2.shape[0] - 1)
                    
                    # Align mask2 to mask1
                    flat_mask2_new = flat_mask2.clone()
                    flat_mask2_new[:coord_portion] = flat_mask1[:coord_portion]
                    
                    # Update mask
                    optimized_masks[name2] = flat_mask2_new.reshape(shape2)
        
        # Prepare metadata
        metadata = {
            "clusters": {cluster_id: names for cluster_id, names in clusters.items()},
            "mutual_information": {f"{k[0]}_{k[1]}": v for k, v in mutual_information.items()},
            "num_coordinated_pairs": len(parameter_pairs)
        }
        
        return optimized_masks, metadata


class InformationMaskOptimizationModule(nn.Module):
    """
    Module for information-theoretic optimization of quantum masks.
    
    This module combines information theory principles to optimize quantum masks,
    enhancing model efficiency while preserving important information.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        base_sparsity: float = 0.8,
        mask_type: MaskType = MaskType.BINARY,
        update_interval: int = 100,
        use_fisher_information: bool = True,
        min_mutual_information: float = 0.2,
        adapt_to_information: bool = True
    ):
        """
        Initialize the information mask optimization module.
        
        Args:
            embedding_dim: Dimension of embeddings
            base_sparsity: Base sparsity level
            mask_type: Type of mask (binary or continuous)
            update_interval: Interval for updating masks
            use_fisher_information: Whether to use Fisher information
            min_mutual_information: Minimum mutual information threshold
            adapt_to_information: Whether to adapt sparsity to information
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.update_interval = update_interval
        
        # Step counter for update interval
        self.register_buffer("step_counter", torch.tensor(0))
        
        # Information content analyzer
        self.information_analyzer = InformationContentAnalyzer(
            embedding_dim=embedding_dim,
            use_fisher_information=use_fisher_information
        )
        
        # Information-guided mask generator
        self.mask_generator = InformationGuidedMaskGenerator(
            base_sparsity=base_sparsity,
            mask_type=mask_type,
            adapt_to_information=adapt_to_information
        )
        
        # Mutual information optimizer
        self.mi_optimizer = MutualInformationOptimizer(
            embedding_dim=embedding_dim,
            min_mutual_information=min_mutual_information
        )
        
        # Store optimized masks
        self.optimized_masks = {}
        
        # Track optimization quality
        self.optimization_quality = {}
    
    def should_update_masks(self) -> bool:
        """
        Determine if masks should be updated.
        
        Returns:
            Boolean indicating whether to update masks
        """
        if not self.training:
            return False
            
        # Increment counter
        self.step_counter += 1
        
        # Check if update interval reached
        if self.step_counter >= self.update_interval:
            self.step_counter.zero_()
            return True
            
        return False
    
    def compute_parameter_embeddings(
        self,
        parameters: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute embeddings for all parameters.
        
        Args:
            parameters: Dictionary of parameters
            
        Returns:
            Dictionary of parameter embeddings
        """
        embeddings = {}
        
        for name, param in parameters.items():
            if not isinstance(param, torch.Tensor) or param.dim() == 0:
                continue
                
            embedding = self.mi_optimizer.compute_parameter_embedding(param)
            embeddings[name] = embedding
        
        return embeddings
    
    def measure_optimization_quality(
        self,
        parameters: Dict[str, torch.Tensor],
        information_densities: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor]
    ) -> float:
        """
        Measure the quality of the mask optimization.
        
        Args:
            parameters: Dictionary of parameters
            information_densities: Dictionary of information densities
            masks: Dictionary of masks
            
        Returns:
            Optimization quality score
        """
        # Weighted information preservation score
        total_score = 0.0
        total_weight = 0.0
        
        for name, param in parameters.items():
            if name not in masks or name not in information_densities:
                continue
                
            # Get mask and information density
            mask = masks[name]
            density = information_densities[name].item()
            
            # Compute retention ratio (percentage of parameters kept)
            retention = (mask > 0).float().mean().item()
            
            # Compute information preservation score
            # Higher density parameters should have higher retention
            target_retention = 1.0 - self.mask_generator.compute_adaptive_sparsity(density)
            
            # Score based on how close retention is to target
            score = 1.0 - abs(retention - target_retention)
            
            # Weight by information density
            weight = density
            
            total_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            quality = total_score / total_weight
        else:
            quality = 0.0
        
        return quality
    
    def forward(
        self,
        parameters: Dict[str, torch.Tensor],
        gradients: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Apply information-theoretic mask optimization.
        
        Args:
            parameters: Dictionary of parameters
            gradients: Optional dictionary of gradients
            
        Returns:
            Tuple containing:
                - Dictionary of optimized masks
                - Dictionary of optimization metadata
        """
        should_update = self.should_update_masks()
        metadata = {
            "update_performed": should_update,
            "optimization_steps": {}
        }
        
        # Use stored masks if available and not updating
        if not should_update and self.optimized_masks:
            return self.optimized_masks, metadata
        
        # Analyze information content
        information_densities, info_metadata = self.information_analyzer(parameters, gradients)
        metadata["optimization_steps"]["information_analysis"] = info_metadata
        
        # Generate information-guided masks
        initial_masks, gen_metadata = self.mask_generator(parameters, information_densities)
        metadata["optimization_steps"]["mask_generation"] = gen_metadata
        
        # Optimize for mutual information
        parameter_embeddings = self.compute_parameter_embeddings(parameters)
        optimized_masks, mi_metadata = self.mi_optimizer.optimize_masks_for_mutual_information(
            parameter_embeddings,
            initial_masks
        )
        metadata["optimization_steps"]["mutual_information"] = mi_metadata
        
        # Measure optimization quality
        quality = self.measure_optimization_quality(
            parameters,
            information_densities,
            optimized_masks
        )
        
        # Update quality history
        step_name = f"step_{self.step_counter.item()}"
        self.optimization_quality[step_name] = quality
        
        metadata["optimization_quality"] = quality
        metadata["quality_history"] = self.optimization_quality
        
        # Store optimized masks
        self.optimized_masks = optimized_masks
        
        return optimized_masks, metadata