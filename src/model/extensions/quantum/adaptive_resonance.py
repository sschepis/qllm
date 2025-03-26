"""
Adaptive Resonance Patterns Module.

This module enables adaptive resonance patterns that dynamically adjust
to input characteristics, enhancing model flexibility and performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union, Set
import math
import numpy as np

from .quantum_types import MaskType, PatternType
from .quantum_patterns import generate_quantum_pattern


class InputAnalyzer(nn.Module):
    """
    Analyzes input characteristics to guide resonance adaptation.
    
    This module examines input features to extract characteristics that
    can guide the adaptation of resonance patterns.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        feature_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize the input analyzer.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layers
            feature_dim: Dimension of extracted features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        
        # Self-attention for capturing relationships in input
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature extraction network
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Sequence feature combination
        self.sequence_combiner = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Statistical features extraction
        self.stats_extractor = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),  # mean, std, max
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Frequency domain feature extraction
        self.freq_extractor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Final feature integration
        self.feature_integration = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),  # attention, stats, freq
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def extract_attention_features(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract features using self-attention.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, embedding_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Attention-based features [batch_size, feature_dim]
        """
        # Apply self-attention
        attn_output, _ = self.self_attention(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=attention_mask
        )
        
        # Extract features from attention output
        token_features = self.feature_extractor(attn_output)
        
        # Combine across sequence
        if attention_mask is not None:
            # Masked average
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_features = torch.sum(token_features * mask_expanded, dim=1)
            mask_sum = torch.sum(mask_expanded, dim=1) + 1e-6  # Avoid division by zero
            combined_features = sum_features / mask_sum
        else:
            # Simple average
            combined_features = token_features.mean(dim=1)
        
        # Further process combined features
        processed_features = self.sequence_combiner(combined_features)
        
        return processed_features
    
    def extract_statistical_features(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract statistical features from hidden states.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, embedding_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Statistical features [batch_size, feature_dim]
        """
        if attention_mask is not None:
            # Create mask for proper statistics
            mask_expanded = attention_mask.unsqueeze(-1).float()
            
            # Compute masked mean
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            mask_sum = torch.sum(mask_expanded, dim=1) + 1e-6  # Avoid division by zero
            mean_hidden = sum_hidden / mask_sum
            
            # Compute masked std
            diff_squared = (hidden_states - mean_hidden.unsqueeze(1)) ** 2
            masked_diff_squared = diff_squared * mask_expanded
            sum_diff_squared = torch.sum(masked_diff_squared, dim=1)
            var_hidden = sum_diff_squared / mask_sum
            std_hidden = torch.sqrt(var_hidden + 1e-6)
            
            # Compute masked max
            hidden_states_masked = hidden_states * mask_expanded
            max_hidden, _ = torch.max(hidden_states_masked, dim=1)
        else:
            # Compute standard statistics
            mean_hidden = hidden_states.mean(dim=1)
            std_hidden = hidden_states.std(dim=1)
            max_hidden, _ = torch.max(hidden_states, dim=1)
        
        # Concatenate statistics
        stats_concat = torch.cat([mean_hidden, std_hidden, max_hidden], dim=-1)
        
        # Extract features from statistics
        stats_features = self.stats_extractor(stats_concat)
        
        return stats_features
    
    def extract_frequency_features(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract frequency domain features from hidden states.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, embedding_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Frequency features [batch_size, feature_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # For frequency analysis, we need a reasonable sequence length
        if seq_len < 4:
            # For very short sequences, fall back to stats
            return self.extract_statistical_features(hidden_states, attention_mask)
        
        # Apply FFT along sequence dimension (simplified approach)
        # In a real implementation, you might use more sophisticated frequency analysis
        
        # Transpose to [batch_size, embedding_dim, seq_len] for FFT
        transposed = hidden_states.transpose(1, 2)
        
        # For simplicity, we'll just use the magnitude of the first few frequency components
        fft_result = torch.fft.rfft(transposed, dim=2)
        fft_magnitudes = torch.abs(fft_result)
        
        # Take only lower frequency components (more significant)
        num_components = min(4, fft_magnitudes.shape[2])
        low_freq_magnitudes = fft_magnitudes[:, :, :num_components]
        
        # Reshape to [batch_size, embedding_dim * num_components]
        flat_freq = low_freq_magnitudes.reshape(batch_size, -1)
        
        # If flat_freq is too large, reduce dimension with a projection
        if flat_freq.shape[1] > self.embedding_dim:
            # Simple pooling to reduce dimension
            flat_freq = flat_freq[:, :self.embedding_dim]
        elif flat_freq.shape[1] < self.embedding_dim:
            # Pad to embedding dimension
            padding = torch.zeros(batch_size, self.embedding_dim - flat_freq.shape[1], device=flat_freq.device)
            flat_freq = torch.cat([flat_freq, padding], dim=1)
        
        # Extract features from frequency components
        freq_features = self.freq_extractor(flat_freq)
        
        return freq_features
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Analyze input to extract characteristics for resonance adaptation.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, embedding_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Tuple containing:
                - Integrated features [batch_size, feature_dim]
                - Dictionary of component features
        """
        # Extract features using different approaches
        attention_features = self.extract_attention_features(hidden_states, attention_mask)
        stats_features = self.extract_statistical_features(hidden_states, attention_mask)
        freq_features = self.extract_frequency_features(hidden_states, attention_mask)
        
        # Integrate features
        combined_features = torch.cat([
            attention_features,
            stats_features,
            freq_features
        ], dim=-1)
        
        integrated_features = self.feature_integration(combined_features)
        
        # Collect component features
        component_features = {
            "attention": attention_features,
            "statistics": stats_features,
            "frequency": freq_features
        }
        
        return integrated_features, component_features


class PatternParameterization(nn.Module):
    """
    Generates pattern parameters based on input features.
    
    This module maps input features to parameters that control the
    generation of resonance patterns, enabling adaptive behavior.
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_patterns: int = 5,
        num_params_per_pattern: int = 6,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize the pattern parameterization module.
        
        Args:
            feature_dim: Dimension of input features
            num_patterns: Number of pattern types to support
            num_params_per_pattern: Number of parameters per pattern
            hidden_dim: Dimension of hidden layers
            dropout: Dropout probability
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_patterns = num_patterns
        self.num_params_per_pattern = num_params_per_pattern
        
        # Pattern type prediction network
        self.pattern_type_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_patterns),
            nn.Softmax(dim=-1)
        )
        
        # Parameter generation networks (one per pattern type)
        self.param_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_params_per_pattern),
                nn.Sigmoid()  # Parameters in [0, 1] range
            )
            for _ in range(num_patterns)
        ])
        
        # Interpolation coefficient predictor
        self.interpolation_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Coefficient in [0, 1] range
        )
        
        # Define pattern type mapping
        self.pattern_type_mapping = {
            0: PatternType.HARMONIC,
            1: PatternType.HILBERT,
            2: PatternType.CYCLIC,
            3: PatternType.PRIME,
            4: PatternType.ORTHOGONAL
        }
    
    def forward(
        self,
        features: torch.Tensor
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate pattern parameters based on input features.
        
        Args:
            features: Input features [batch_size, feature_dim]
            
        Returns:
            Tuple containing:
                - Dictionary of pattern parameters
                - Dictionary of metadata
        """
        batch_size = features.shape[0]
        
        # Predict pattern type probabilities
        pattern_type_probs = self.pattern_type_predictor(features)
        
        # Determine dominant pattern type for each sample
        dominant_types = torch.argmax(pattern_type_probs, dim=-1)
        
        # Generate parameters for each pattern type
        all_params = []
        for i in range(self.num_patterns):
            params = self.param_generators[i](features)
            all_params.append(params)
        
        all_params = torch.stack(all_params, dim=1)  # [batch_size, num_patterns, num_params]
        
        # Get parameters for dominant type
        batch_indices = torch.arange(batch_size, device=features.device)
        dominant_params = all_params[batch_indices, dominant_types]
        
        # Predict interpolation coefficient
        interpolation_coef = self.interpolation_predictor(features)
        
        # Collect pattern specifications
        pattern_params = {}
        metadata = {}
        
        for i in range(batch_size):
            # Map numeric type to pattern type
            pattern_type = self.pattern_type_mapping[dominant_types[i].item()]
            
            # Get parameters for this sample
            params = dominant_params[i].tolist()
            
            # Create parameter dictionary
            params_dict = {
                "pattern_type": pattern_type,
                "frequency_factor": 1.0 + params[0] * 9.0,  # Scale to [1, 10]
                "phase_shift": params[1] * math.pi * 2,  # Scale to [0, 2π]
                "amplitude_factor": 0.5 + params[2] * 0.5,  # Scale to [0.5, 1.0]
                "decay_rate": 0.1 + params[3] * 0.4,  # Scale to [0.1, 0.5]
                "sparsity": 0.5 + params[4] * 0.4,  # Scale to [0.5, 0.9]
                "symmetry_factor": params[5]  # Already in [0, 1]
            }
            
            # Add interpolation coefficient
            params_dict["interpolation"] = interpolation_coef[i].item()
            
            pattern_params[i] = params_dict
            
            # Record metadata
            metadata[i] = {
                "pattern_type_probs": pattern_type_probs[i].tolist(),
                "dominant_type": dominant_types[i].item(),
                "pattern_type": pattern_type,
                "parameters": params
            }
        
        return pattern_params, metadata


class ResonanceAdapter(nn.Module):
    """
    Adapts resonance patterns based on input characteristics.
    
    This module adjusts resonance patterns to better match the properties
    of the input, enhancing model flexibility and performance.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        feature_dim: int = 64,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_patterns: int = 5,
        dropout: float = 0.1
    ):
        """
        Initialize the resonance adapter.
        
        Args:
            embedding_dim: Dimension of input embeddings
            feature_dim: Dimension of features
            hidden_dim: Dimension of hidden layers
            num_heads: Number of attention heads
            num_patterns: Number of pattern types
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        
        # Input analyzer component
        self.input_analyzer = InputAnalyzer(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Pattern parameterization component
        self.pattern_parameterization = PatternParameterization(
            feature_dim=feature_dim,
            num_patterns=num_patterns,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Pattern integration network
        self.pattern_integration = nn.Sequential(
            nn.Linear(embedding_dim + feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def generate_adaptive_mask(
        self,
        shape: Tuple[int, int],
        params: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Generate an adaptive mask based on parameters.
        
        Args:
            shape: Shape of the mask (rows, cols)
            params: Dictionary of pattern parameters
            
        Returns:
            Adaptive mask tensor
        """
        # Extract parameters
        pattern_type = params["pattern_type"]
        
        # Generate base quantum pattern
        pattern_params = {
            "frequency_factor": params["frequency_factor"],
            "phase_shift": params["phase_shift"],
            "amplitude_factor": params["amplitude_factor"],
            "decay_rate": params["decay_rate"],
            "sparsity": params["sparsity"],
            "symmetry_factor": params["symmetry_factor"]
        }
        
        # Generate mask using quantum pattern generator
        mask = generate_quantum_pattern(
            shape,
            pattern_type,
            **pattern_params
        )
        
        return mask
    
    def interpolate_with_base_mask(
        self,
        adaptive_mask: torch.Tensor,
        base_mask: torch.Tensor,
        interpolation: float
    ) -> torch.Tensor:
        """
        Interpolate between adaptive and base masks.
        
        Args:
            adaptive_mask: Adaptive mask tensor
            base_mask: Base mask tensor
            interpolation: Interpolation coefficient in [0, 1]
            
        Returns:
            Interpolated mask tensor
        """
        return adaptive_mask * interpolation + base_mask * (1 - interpolation)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        base_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Adapt resonance patterns based on input.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, embedding_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
            base_masks: Optional dictionary of base masks to adapt
            
        Returns:
            Tuple containing:
                - Dictionary of adapted masks
                - Dictionary of adaptation metadata
        """
        # Analyze input to extract features
        features, component_features = self.input_analyzer(hidden_states, attention_mask)
        
        # Generate pattern parameters based on features
        pattern_params, param_metadata = self.pattern_parameterization(features)
        
        # Generate adaptive masks
        adaptive_masks = {}
        
        if base_masks is not None:
            for name, base_mask in base_masks.items():
                # Generate adaptive mask for each base mask
                shape = base_mask.shape
                
                # Use parameters from first sample (batch_size=1 assumption)
                params = pattern_params[0]
                
                # Generate adaptive mask
                adaptive_mask = self.generate_adaptive_mask(shape, params)
                
                # Convert to tensor if not already
                if not isinstance(adaptive_mask, torch.Tensor):
                    adaptive_mask = torch.tensor(adaptive_mask, device=base_mask.device)
                
                # Interpolate with base mask
                interpolation = params["interpolation"]
                final_mask = self.interpolate_with_base_mask(
                    adaptive_mask,
                    base_mask,
                    interpolation
                )
                
                adaptive_masks[name] = final_mask
        
        # Prepare adaptation metadata
        metadata = {
            "features": component_features,
            "pattern_params": pattern_params,
            "param_metadata": param_metadata
        }
        
        # Enhance hidden states with adaptation features
        enhanced_states = []
        for i in range(hidden_states.shape[0]):
            sample_features = features[i:i+1]  # Keep batch dimension
            sample_hidden = hidden_states[i:i+1]
            
            # Expand features to match sequence length
            expanded_features = sample_features.expand(
                1, sample_hidden.shape[1], sample_features.shape[-1]
            )
            
            # Concatenate features with hidden states
            combined = torch.cat([sample_hidden, expanded_features], dim=-1)
            
            # Apply integration network
            enhanced = self.pattern_integration(combined)
            
            enhanced_states.append(enhanced)
        
        # Combine enhanced states back to batch
        enhanced_hidden_states = torch.cat(enhanced_states, dim=0)
        
        # Add enhanced hidden states to metadata
        metadata["enhanced_hidden_states"] = enhanced_hidden_states
        
        return adaptive_masks, metadata


class AdaptiveResonanceModule(nn.Module):
    """
    Module for adaptive resonance patterns based on input.
    
    This module enables the dynamic adaptation of resonance patterns
    based on input characteristics, enhancing model flexibility.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        feature_dim: int = 64,
        hidden_dim: int = 256,
        num_heads: int = 4,
        update_interval: int = 100,
        adaptation_rate: float = 0.1,
        dropout: float = 0.1
    ):
        """
        Initialize the adaptive resonance module.
        
        Args:
            embedding_dim: Dimension of input embeddings
            feature_dim: Dimension of extracted features
            hidden_dim: Dimension of hidden layers
            num_heads: Number of attention heads
            update_interval: Interval for updating adaptive patterns
            adaptation_rate: Rate for adaptation updates
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.update_interval = update_interval
        self.adaptation_rate = adaptation_rate
        
        # Step counter for update interval
        self.register_buffer("step_counter", torch.tensor(0))
        
        # Resonance adapter component
        self.resonance_adapter = ResonanceAdapter(
            embedding_dim=embedding_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Resonance quality evaluator
        self.quality_evaluator = nn.Sequential(
            nn.Linear(embedding_dim + feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize adapted masks
        self.adapted_masks = {}
        self.mask_quality_scores = {}
        self.mask_update_history = {}
    
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
    
    def evaluate_mask_quality(
        self,
        hidden_states: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate the quality of resonance with current masks.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, embedding_dim]
            features: Input features [batch_size, feature_dim]
            
        Returns:
            Quality score [batch_size]
        """
        # Expand features to match sequence length
        expanded_features = features.unsqueeze(1).expand(
            -1, hidden_states.shape[1], -1
        )
        
        # Concatenate features with hidden states
        combined = torch.cat([hidden_states, expanded_features], dim=-1)
        
        # Apply quality evaluator
        quality_scores = self.quality_evaluator(combined).squeeze(-1)
        
        # Average across sequence
        avg_quality = quality_scores.mean(dim=1)
        
        return avg_quality
    
    def update_adapted_masks(
        self,
        new_masks: Dict[str, torch.Tensor],
        quality_score: float
    ) -> None:
        """
        Update adapted masks with new masks.
        
        Args:
            new_masks: Dictionary of new adapted masks
            quality_score: Quality score for the new masks
        """
        for name, new_mask in new_masks.items():
            if name in self.adapted_masks:
                # Update existing mask with adaptation rate
                current_mask = self.adapted_masks[name]
                updated_mask = (
                    current_mask * (1 - self.adaptation_rate) +
                    new_mask * self.adaptation_rate
                )
                self.adapted_masks[name] = updated_mask
            else:
                # Add new mask
                self.adapted_masks[name] = new_mask
            
            # Update quality score
            self.mask_quality_scores[name] = quality_score
            
            # Update history
            if name not in self.mask_update_history:
                self.mask_update_history[name] = []
            self.mask_update_history[name].append(quality_score)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        base_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Apply adaptive resonance patterns to masks.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, embedding_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
            base_masks: Optional dictionary of base masks to adapt
            
        Returns:
            Tuple containing:
                - Dictionary of adapted masks
                - Dictionary of adaptation metadata
        """
        # Use base masks if no adapted masks available yet
        if not self.adapted_masks and base_masks is not None:
            self.adapted_masks = {
                name: mask.clone() for name, mask in base_masks.items()
            }
        
        # Analyze input and get adaptation features (always done)
        features, component_features = self.resonance_adapter.input_analyzer(hidden_states, attention_mask)
        
        # Evaluate current mask quality
        quality_score = self.evaluate_mask_quality(hidden_states, features).mean().item()
        
        # Check if we should update masks
        if self.should_update_masks():
            # Generate adaptive masks
            new_masks, adaptation_metadata = self.resonance_adapter(
                hidden_states,
                attention_mask,
                base_masks if not self.adapted_masks else self.adapted_masks
            )
            
            # Update adapted masks
            self.update_adapted_masks(new_masks, quality_score)
            
            # Enhanced hidden states from adaptation
            enhanced_hidden_states = adaptation_metadata.get("enhanced_hidden_states")
        else:
            # No adaptation this step
            adaptation_metadata = {
                "features": component_features
            }
            enhanced_hidden_states = None
        
        # Prepare output metadata
        metadata = {
            "adapted_masks": {k: v.clone() for k, v in self.adapted_masks.items()},
            "quality_score": quality_score,
            "quality_history": self.mask_quality_scores,
            "update_history": self.mask_update_history,
            "updated": self.should_update_masks(),
            **adaptation_metadata
        }
        
        if enhanced_hidden_states is not None:
            metadata["enhanced_hidden_states"] = enhanced_hidden_states
        
        return self.adapted_masks, metadata