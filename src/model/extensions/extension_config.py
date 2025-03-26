"""
Extension Configuration Module.

This module defines configuration classes for extensions to the 
Semantic Resonance Language Model.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union


@dataclass
class MultimodalConfig:
    """Configuration for the multimodal extension."""
    
    # Enable/disable multimodal features
    enabled: bool = False
    
    # Vision encoder settings
    vision_encoder_type: str = "resnet"  # or "vit"
    vision_encoder_model: str = "resnet50"
    vision_embedding_dim: int = 768
    vision_primes: List[int] = field(default_factory=lambda: [11, 13, 17, 19, 23])
    
    # Audio encoder settings
    audio_enabled: bool = False
    audio_encoder_type: str = "wav2vec"
    audio_embedding_dim: int = 512
    audio_primes: List[int] = field(default_factory=lambda: [7, 11, 13, 17])
    
    # Cross-modal fusion settings
    fusion_type: str = "attention"  # or "concat", "sum"
    fusion_heads: int = 8
    fusion_dropout: float = 0.1
    fusion_iterations: int = 5
    fusion_entropy_threshold: float = 0.2


@dataclass
class MemoryConfig:
    """Configuration for the extended memory extension."""
    
    # Enable/disable extended memory features
    enabled: bool = False
    
    # Basic memory settings
    memory_size: int = 10000
    memory_key_dim: int = 128
    memory_value_dim: int = 768
    
    # Knowledge graph settings
    use_graph_structure: bool = True
    max_relations: int = 5
    relation_embedding_dim: int = 64
    
    # Retrieval settings
    num_neighbors: int = 10
    use_importance_sampling: bool = True
    temperature: float = 0.1
    
    # Memory persistence
    persistence_enabled: bool = False
    persistence_path: str = "memory/knowledge_graph.pkl"
    persistence_interval: int = 1000  # Steps between saves


@dataclass
class QuantumConfig:
    """Configuration for the quantum group symmetries extension."""
    
    # Enable/disable quantum symmetry features
    enabled: bool = False
    
    # Group theory settings
    group_type: str = "cyclic"  # or "permutation", "orthogonal", "lie"
    group_order: int = 5
    
    # Masking settings
    mask_type: str = "mod"  # or "prime", "adaptive"
    mask_sparsity: float = 0.8  # Percentage of weights to mask
    adaptive_threshold: float = 0.1
    
    # Symmetry operations
    use_equivariant_layers: bool = True
    symmetry_preservation_weight: float = 0.1
    auto_discover_symmetries: bool = False


@dataclass
class ExtensionConfig:
    """
    Configuration for all extensions to the Semantic Resonance Language Model.
    
    This class contains configurations for each extension type and common settings
    that apply to all extensions.
    """
    
    # Global extension settings
    extensions_enabled: bool = True
    default_device: str = "cuda"
    extension_dropout: float = 0.1
    
    # Integration method
    integration_method: str = "sequential"  # or "parallel", "adaptive"
    
    # Individual extension configs
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    
    # Extension-specific feature flags (for fine-grained control)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default feature flags if not provided."""
        default_flags = {
            "multimodal.vision": self.multimodal.enabled,
            "multimodal.audio": self.multimodal.enabled and self.multimodal.audio_enabled,
            "memory.graph": self.memory.enabled and self.memory.use_graph_structure,
            "memory.persistence": self.memory.enabled and self.memory.persistence_enabled,
            "quantum.equivariant": self.quantum.enabled and self.quantum.use_equivariant_layers,
            "quantum.adaptive": self.quantum.enabled and self.quantum.auto_discover_symmetries,
        }
        
        # Only set default flags for keys not explicitly provided
        for key, value in default_flags.items():
            if key not in self.feature_flags:
                self.feature_flags[key] = value
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if a specific feature is enabled.
        
        Args:
            feature_name (str): Name of the feature to check
            
        Returns:
            bool: True if the feature is enabled, False otherwise
        """
        # First check if the feature exists in the feature flags
        if feature_name in self.feature_flags:
            return self.feature_flags[feature_name]
        
        # If not, check if the extension is enabled
        if feature_name.startswith("multimodal"):
            return self.multimodal.enabled
        elif feature_name.startswith("memory"):
            return self.memory.enabled
        elif feature_name.startswith("quantum"):
            return self.quantum.enabled
        
        # Default to False for unknown features
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the configuration
        """
        return {
            "extensions_enabled": self.extensions_enabled,
            "default_device": self.default_device,
            "extension_dropout": self.extension_dropout,
            "integration_method": self.integration_method,
            "multimodal": {k: v for k, v in self.multimodal.__dict__.items()},
            "memory": {k: v for k, v in self.memory.__dict__.items()},
            "quantum": {k: v for k, v in self.quantum.__dict__.items()},
            "feature_flags": self.feature_flags
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExtensionConfig':
        """
        Create a configuration from a dictionary.
        
        Args:
            config_dict (Dict[str, Any]): Dictionary with configuration values
            
        Returns:
            ExtensionConfig: Configuration instance
        """
        # Create default instance
        config = cls()
        
        # Set top-level attributes
        for key in ["extensions_enabled", "default_device", "extension_dropout", "integration_method"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        # Set feature flags
        if "feature_flags" in config_dict:
            config.feature_flags = config_dict["feature_flags"]
        
        # Set multimodal config
        if "multimodal" in config_dict:
            for key, value in config_dict["multimodal"].items():
                if hasattr(config.multimodal, key):
                    setattr(config.multimodal, key, value)
        
        # Set memory config
        if "memory" in config_dict:
            for key, value in config_dict["memory"].items():
                if hasattr(config.memory, key):
                    setattr(config.memory, key, value)
        
        # Set quantum config
        if "quantum" in config_dict:
            for key, value in config_dict["quantum"].items():
                if hasattr(config.quantum, key):
                    setattr(config.quantum, key, value)
        
        # Re-run post-init to ensure derived values are set correctly
        config.__post_init__()
        
        return config