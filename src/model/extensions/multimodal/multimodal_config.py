"""
Multimodal Extension Configuration Module.

This module defines configuration parameters for multimodal extensions.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class MultimodalConfig:
    """Base configuration parameters for multimodal extensions."""
    
    # Basic modality information
    modality_type: str = "generic"
    modality_embedding_dim: int = 768
    
    # Multiple input handling
    supports_multiple_inputs: bool = False
    
    # Projection settings
    prime_mapping: List[int] = field(default_factory=list)
    
    # Advanced configuration options can be added via the extra_config dict
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MultimodalConfig':
        """
        Create a configuration object from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            MultimodalConfig object with parameters from the dictionary
        """
        # Extract known parameters
        modality_type = config_dict.get("modality_type", cls.modality_type)
        modality_embedding_dim = config_dict.get("modality_embedding_dim", cls.modality_embedding_dim)
        supports_multiple_inputs = config_dict.get("supports_multiple_inputs", cls.supports_multiple_inputs)
        prime_mapping = config_dict.get("prime_mapping", cls.prime_mapping)
        
        # Store any extra configuration parameters
        extra_config = {k: v for k, v in config_dict.items() if k not in {
            "modality_type", "modality_embedding_dim", 
            "supports_multiple_inputs", "prime_mapping"
        }}
        
        return cls(
            modality_type=modality_type,
            modality_embedding_dim=modality_embedding_dim,
            supports_multiple_inputs=supports_multiple_inputs,
            prime_mapping=prime_mapping,
            extra_config=extra_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        config_dict = {
            "modality_type": self.modality_type,
            "modality_embedding_dim": self.modality_embedding_dim,
            "supports_multiple_inputs": self.supports_multiple_inputs,
            "prime_mapping": self.prime_mapping,
        }
        
        # Add any extra configuration parameters
        config_dict.update(self.extra_config)
        
        return config_dict


@dataclass
class VisionConfig(MultimodalConfig):
    """Configuration parameters specific to vision extensions."""
    
    # Vision model configuration
    vision_encoder_type: str = "resnet"
    vision_encoder_model: str = "resnet50"
    vision_embedding_dim: int = 768
    vision_primes: List[int] = field(default_factory=lambda: [11, 13, 17, 19, 23])
    
    # Advanced configuration for pre-trained models
    pretrained: bool = True
    freeze_backbone: bool = True
    feature_extraction_layer: str = "penultimate"
    use_spatial_features: bool = False
    
    # Multi-resolution and dynamic fusion settings
    use_multi_resolution: bool = False
    use_dynamic_fusion: bool = False
    relevance_threshold: float = 0.5
    
    # Input image configuration
    image_size: int = 224
    use_patch_tokens: bool = False
    image_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    image_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Fusion configuration
    fusion_type: str = "attention"
    fusion_heads: int = 8
    fusion_iterations: int = 5
    fusion_entropy_threshold: float = 0.2
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VisionConfig':
        """
        Create a vision configuration object from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            VisionConfig object with parameters from the dictionary
        """
        # First create base multimodal config
        base_config = MultimodalConfig.from_dict(config_dict)
        
        # Set modality type to vision
        modality_type = "vision"
        
        # Extract vision-specific parameters
        vision_encoder_type = config_dict.get("vision_encoder_type", cls.vision_encoder_type)
        vision_encoder_model = config_dict.get("vision_encoder_model", cls.vision_encoder_model)
        vision_embedding_dim = config_dict.get("vision_embedding_dim", cls.vision_embedding_dim)
        vision_primes = config_dict.get("vision_primes", cls.vision_primes)
        pretrained = config_dict.get("pretrained", cls.pretrained)
        freeze_backbone = config_dict.get("freeze_backbone", cls.freeze_backbone)
        feature_extraction_layer = config_dict.get("feature_extraction_layer", cls.feature_extraction_layer)
        use_spatial_features = config_dict.get("use_spatial_features", cls.use_spatial_features)
        use_multi_resolution = config_dict.get("use_multi_resolution", cls.use_multi_resolution)
        use_dynamic_fusion = config_dict.get("use_dynamic_fusion", cls.use_dynamic_fusion)
        relevance_threshold = config_dict.get("relevance_threshold", cls.relevance_threshold)
        image_size = config_dict.get("image_size", cls.image_size)
        use_patch_tokens = config_dict.get("use_patch_tokens", cls.use_patch_tokens)
        image_mean = config_dict.get("image_mean", cls.image_mean)
        image_std = config_dict.get("image_std", cls.image_std)
        fusion_type = config_dict.get("fusion_type", cls.fusion_type)
        fusion_heads = config_dict.get("fusion_heads", cls.fusion_heads)
        fusion_iterations = config_dict.get("fusion_iterations", cls.fusion_iterations)
        fusion_entropy_threshold = config_dict.get("fusion_entropy_threshold", cls.fusion_entropy_threshold)
        
        # Store any extra configuration parameters
        extra_config = base_config.extra_config
        
        return cls(
            modality_type=modality_type,
            modality_embedding_dim=base_config.modality_embedding_dim,
            supports_multiple_inputs=base_config.supports_multiple_inputs,
            prime_mapping=base_config.prime_mapping,
            extra_config=extra_config,
            vision_encoder_type=vision_encoder_type,
            vision_encoder_model=vision_encoder_model,
            vision_embedding_dim=vision_embedding_dim,
            vision_primes=vision_primes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            feature_extraction_layer=feature_extraction_layer,
            use_spatial_features=use_spatial_features,
            use_multi_resolution=use_multi_resolution,
            use_dynamic_fusion=use_dynamic_fusion,
            relevance_threshold=relevance_threshold,
            image_size=image_size,
            use_patch_tokens=use_patch_tokens,
            image_mean=image_mean,
            image_std=image_std,
            fusion_type=fusion_type,
            fusion_heads=fusion_heads,
            fusion_iterations=fusion_iterations,
            fusion_entropy_threshold=fusion_entropy_threshold,
        )