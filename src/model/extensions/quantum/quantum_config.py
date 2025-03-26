"""
Quantum Extension Configuration Module.

This module defines configuration parameters for quantum extensions.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .quantum_types import GroupType, MaskType, EvolutionMethod


@dataclass
class QuantumConfig:
    """Base configuration parameters for quantum extensions."""
    
    # Group theory configuration
    group_type: GroupType = "cyclic"
    group_order: int = 5
    
    # Masking configuration
    mask_type: MaskType = "mod"
    mask_sparsity: float = 0.8
    adaptive_threshold: float = 0.1
    
    # Symmetry operation configuration
    use_equivariant_layers: bool = True
    symmetry_preservation_weight: float = 0.1
    auto_discover_symmetries: bool = False
    
    # Advanced configuration options can be added via the extra_config dict
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QuantumConfig':
        """
        Create a configuration object from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            QuantumConfig object with parameters from the dictionary
        """
        # Extract known parameters
        group_type = config_dict.get("group_type", cls.group_type)
        group_order = config_dict.get("group_order", cls.group_order)
        mask_type = config_dict.get("mask_type", cls.mask_type)
        mask_sparsity = config_dict.get("mask_sparsity", cls.mask_sparsity)
        adaptive_threshold = config_dict.get("adaptive_threshold", cls.adaptive_threshold)
        use_equivariant_layers = config_dict.get("use_equivariant_layers", cls.use_equivariant_layers)
        symmetry_preservation_weight = config_dict.get("symmetry_preservation_weight", cls.symmetry_preservation_weight)
        auto_discover_symmetries = config_dict.get("auto_discover_symmetries", cls.auto_discover_symmetries)
        
        # Store any extra configuration parameters
        extra_config = {k: v for k, v in config_dict.items() if k not in {
            "group_type", "group_order", "mask_type", "mask_sparsity", 
            "adaptive_threshold", "use_equivariant_layers", 
            "symmetry_preservation_weight", "auto_discover_symmetries"
        }}
        
        return cls(
            group_type=group_type,
            group_order=group_order,
            mask_type=mask_type,
            mask_sparsity=mask_sparsity,
            adaptive_threshold=adaptive_threshold,
            use_equivariant_layers=use_equivariant_layers,
            symmetry_preservation_weight=symmetry_preservation_weight,
            auto_discover_symmetries=auto_discover_symmetries,
            extra_config=extra_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        config_dict = {
            "group_type": self.group_type,
            "group_order": self.group_order,
            "mask_type": self.mask_type,
            "mask_sparsity": self.mask_sparsity,
            "adaptive_threshold": self.adaptive_threshold,
            "use_equivariant_layers": self.use_equivariant_layers,
            "symmetry_preservation_weight": self.symmetry_preservation_weight,
            "auto_discover_symmetries": self.auto_discover_symmetries,
        }
        
        # Add any extra configuration parameters
        config_dict.update(self.extra_config)
        
        return config_dict


@dataclass
class SymmetryMaskConfig(QuantumConfig):
    """Configuration parameters specific to symmetry mask extensions."""
    
    # Additional mask configuration
    mask_update_interval: int = 1000
    use_gradual_pruning: bool = False
    final_sparsity: float = 0.9
    initial_sparsity: float = 0.5
    pruning_steps: int = 10000
    
    # Prime pattern configuration
    prime_moduli: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    
    # Adaptive mask settings
    importance_threshold: float = 0.01
    importance_measure: str = "magnitude"  # or "gradient", "sensitivity"
    
    # Advanced quantum-inspired configurations
    use_quantum_patterns: bool = False
    harmonic_levels: int = 5
    use_hilbert_projections: bool = False
    hilbert_dim: int = 16
    
    # Dynamic mask evolution settings
    use_dynamic_masks: bool = False
    evolution_rate: float = 0.1
    mask_resonance_factor: float = 0.2
    apply_mask_evolution: bool = True
    evolution_interval: int = 100
    evolution_method: EvolutionMethod = "gradient_sensitive"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SymmetryMaskConfig':
        """
        Create a symmetry mask configuration object from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            SymmetryMaskConfig object with parameters from the dictionary
        """
        # First create base quantum config
        base_config = QuantumConfig.from_dict(config_dict)
        
        # Extract symmetry mask specific parameters
        mask_update_interval = config_dict.get("mask_update_interval", cls.mask_update_interval)
        use_gradual_pruning = config_dict.get("use_gradual_pruning", cls.use_gradual_pruning)
        final_sparsity = config_dict.get("final_sparsity", cls.final_sparsity)
        initial_sparsity = config_dict.get("initial_sparsity", cls.initial_sparsity)
        pruning_steps = config_dict.get("pruning_steps", cls.pruning_steps)
        prime_moduli = config_dict.get("prime_moduli", cls.prime_moduli)
        importance_threshold = config_dict.get("importance_threshold", cls.importance_threshold)
        importance_measure = config_dict.get("importance_measure", cls.importance_measure)
        use_quantum_patterns = config_dict.get("use_quantum_patterns", cls.use_quantum_patterns)
        harmonic_levels = config_dict.get("harmonic_levels", cls.harmonic_levels)
        use_hilbert_projections = config_dict.get("use_hilbert_projections", cls.use_hilbert_projections)
        hilbert_dim = config_dict.get("hilbert_dim", cls.hilbert_dim)
        use_dynamic_masks = config_dict.get("use_dynamic_masks", cls.use_dynamic_masks)
        evolution_rate = config_dict.get("evolution_rate", cls.evolution_rate)
        mask_resonance_factor = config_dict.get("mask_resonance_factor", cls.mask_resonance_factor)
        apply_mask_evolution = config_dict.get("apply_mask_evolution", cls.apply_mask_evolution)
        evolution_interval = config_dict.get("evolution_interval", cls.evolution_interval)
        evolution_method = config_dict.get("evolution_method", cls.evolution_method)
        
        # Store any extra configuration parameters
        extra_config = base_config.extra_config
        
        return cls(
            group_type=base_config.group_type,
            group_order=base_config.group_order,
            mask_type=base_config.mask_type,
            mask_sparsity=base_config.mask_sparsity,
            adaptive_threshold=base_config.adaptive_threshold,
            use_equivariant_layers=base_config.use_equivariant_layers,
            symmetry_preservation_weight=base_config.symmetry_preservation_weight,
            auto_discover_symmetries=base_config.auto_discover_symmetries,
            extra_config=extra_config,
            mask_update_interval=mask_update_interval,
            use_gradual_pruning=use_gradual_pruning,
            final_sparsity=final_sparsity,
            initial_sparsity=initial_sparsity,
            pruning_steps=pruning_steps,
            prime_moduli=prime_moduli,
            importance_threshold=importance_threshold,
            importance_measure=importance_measure,
            use_quantum_patterns=use_quantum_patterns,
            harmonic_levels=harmonic_levels,
            use_hilbert_projections=use_hilbert_projections,
            hilbert_dim=hilbert_dim,
            use_dynamic_masks=use_dynamic_masks,
            evolution_rate=evolution_rate,
            mask_resonance_factor=mask_resonance_factor,
            apply_mask_evolution=apply_mask_evolution,
            evolution_interval=evolution_interval,
            evolution_method=evolution_method,
        )