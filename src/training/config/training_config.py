"""
Enhanced training configuration for the Quantum Resonance Language Model.

This module defines an extended training configuration that builds on the base
TrainingConfig with additional parameters for the enhanced training system's
components such as adapters, strategies, and extensions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
import copy

from src.config.training_config import TrainingConfig as BaseTrainingConfig


@dataclass
class ModelAdapterConfig:
    """Configuration for model adapters."""
    
    # Adapter type
    adapter_type: str = "standard"  # standard, dialogue, multimodal
    
    # Model loading
    pretrained_model_name_or_path: Optional[str] = None
    from_config: bool = False
    from_checkpoint: bool = False
    
    # Token handling
    add_special_tokens: bool = True
    tokenizer_name: Optional[str] = None
    tokenizer_padding_side: str = "right"
    max_sequence_length: int = 512
    
    # For multimodal adapters
    vision_encoder_name: Optional[str] = None
    vision_config: Dict[str, Any] = field(default_factory=dict)
    image_size: Union[int, Tuple[int, int]] = 224
    
    # Custom adapter settings
    model_adapter_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetAdapterConfig:
    """Configuration for dataset adapters."""
    
    # Adapter type
    adapter_type: str = "standard"  # standard, dialogue, multimodal
    
    # Dataset paths
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    
    # Data processing
    max_samples: Optional[int] = None
    preprocessing_num_workers: int = 4
    shuffle_training: bool = True
    
    # For dialogue adapters
    conversation_template: Optional[str] = None
    user_token: str = "<user>"
    assistant_token: str = "<assistant>"
    system_token: str = "<system>"
    
    # For multimodal adapters
    image_column: str = "image"
    text_column: str = "text"
    
    # Custom adapter settings
    dataset_adapter_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingStrategyConfig:
    """Configuration for training strategies."""
    
    # Strategy type
    strategy_type: str = "standard"  # standard, pretrain, finetune
    
    # Loss configuration
    loss_type: str = "default"  # default, weighted, contrastive
    label_smoothing: float = 0.0
    
    # Advanced training techniques
    use_gradient_checkpointing: bool = False
    use_deepspeed: bool = False
    deepspeed_config: Optional[Dict[str, Any]] = None
    
    # Distributed training
    distributed_training: bool = False
    distributed_world_size: int = 1
    distributed_backend: str = "nccl"
    
    # Data parallelism
    use_ddp: bool = False  # DistributedDataParallel
    use_fsdp: bool = False  # Fully Sharded Data Parallel
    fsdp_config: Dict[str, Any] = field(default_factory=dict)
    
    # Custom strategy settings
    strategy_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtensionConfig:
    """Configuration for training extensions."""
    
    # Extensions to use
    enabled_extensions: List[str] = field(default_factory=list)
    
    # Extension-specific configurations
    extension_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Extension hooks
    register_hooks: List[str] = field(default_factory=list)
    
    # Extension initialization order
    extension_init_order: Optional[List[str]] = None


@dataclass
class OptimizationConfig:
    """Configuration for optimization components."""
    
    # Optimizer settings
    optimizer_type: str = "adamw"
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Parameter-specific settings
    layer_decay: Optional[float] = None
    no_decay_norm_and_bias: bool = True
    custom_parameter_rules: Optional[List[Dict[str, Any]]] = None
    
    # Learning rate scheduler
    lr_scheduler_type: str = "linear_warmup"
    warmup_steps: int = 0
    warmup_ratio: float = 0.1
    
    # Mixed precision training
    use_amp: bool = True
    fp16_opt_level: str = "O1"
    loss_scale: float = 65536.0
    
    # Gradient accumulation and clipping
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Memory optimization settings
    use_gradient_checkpointing: bool = False
    enable_memory_efficient_eval: bool = True
    max_memory_usage_ratio: float = 0.8
    optimize_batch_size: bool = True
    auto_handle_oom: bool = True
    
    # Optimization kwargs
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""
    
    # Checkpoint saving
    save_strategy: str = "epoch"  # epoch, steps, no
    save_steps: int = 0
    save_total_limit: int = 3
    save_safetensors: bool = True
    
    # Checkpoint loading
    resume_from_checkpoint: Optional[str] = None
    auto_resume: bool = True
    ignore_checkpoints: bool = False
    
    # What to save
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_metrics: bool = True
    save_model_config: bool = True
    
    # Checkpoint formatting
    save_as_safetensors: bool = True
    checkpoint_format: str = "pytorch"  # pytorch, safetensors, sharded
    
    # Custom checkpoint settings
    checkpoint_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedTrainingConfig(BaseTrainingConfig):
    """Enhanced training configuration with adapter and strategy support."""
    
    # Enhanced components
    model_adapter: ModelAdapterConfig = field(default_factory=ModelAdapterConfig)
    dataset_adapter: DatasetAdapterConfig = field(default_factory=DatasetAdapterConfig)
    training_strategy: TrainingStrategyConfig = field(default_factory=TrainingStrategyConfig)
    extensions: ExtensionConfig = field(default_factory=ExtensionConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Training framework settings
    trainer_name: str = "enhanced"  # enhanced, standard, dialogue, continuous
    
    # Compatibility - these fields override the base training config
    # but are kept for backwards compatibility
    training_type: str = "enhanced"  # standard, dialogue, enhanced, verbose
    
    def __post_init__(self):
        """Initialize default values based on training type."""
        # Set defaults based on training type
        if self.training_type != "enhanced":
            self._set_defaults_from_training_type()
    
    def _set_defaults_from_training_type(self):
        """Set default configuration based on training type."""
        training_type = self.training_type.lower()
        
        # Configure model and dataset adapters based on training type
        if training_type == "dialogue":
            self.model_adapter.adapter_type = "dialogue"
            self.dataset_adapter.adapter_type = "dialogue"
        elif training_type == "multimodal":
            self.model_adapter.adapter_type = "multimodal"
            self.dataset_adapter.adapter_type = "multimodal"
        
        # Configure optimization based on base config
        self.optimization.learning_rate = self.learning_rate
        self.optimization.weight_decay = self.weight_decay
        self.optimization.optimizer_type = self.optimizer
        self.optimization.lr_scheduler_type = self.lr_scheduler
        self.optimization.warmup_steps = self.warmup_steps
        self.optimization.warmup_ratio = self.warmup_ratio
        self.optimization.max_grad_norm = self.max_grad_norm
        self.optimization.gradient_accumulation_steps = self.accumulation_steps
        self.optimization.use_amp = self.use_mixed_precision
        
        # Configure checkpointing based on base config
        self.checkpointing.save_steps = self.save_steps
        self.checkpointing.auto_resume = self.auto_resume
        self.checkpointing.ignore_checkpoints = self.ignore_checkpoints
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        # Start with the base class dictionary
        result = super().to_dict()
        
        # Add enhanced component dictionaries
        result["model_adapter"] = self._dataclass_to_dict(self.model_adapter)
        result["dataset_adapter"] = self._dataclass_to_dict(self.dataset_adapter)
        result["training_strategy"] = self._dataclass_to_dict(self.training_strategy)
        result["extensions"] = self._dataclass_to_dict(self.extensions)
        result["optimization"] = self._dataclass_to_dict(self.optimization)
        result["checkpointing"] = self._dataclass_to_dict(self.checkpointing)
        
        return result
    
    @staticmethod
    def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
        """Convert a dataclass instance to a dictionary."""
        if hasattr(obj, "__dataclass_fields__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return obj
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhancedTrainingConfig":
        """Create from dictionary representation."""
        # Copy the data to avoid modifying the input
        data_copy = copy.deepcopy(data)
        
        # Extract enhanced component dictionaries
        model_adapter_data = data_copy.pop("model_adapter", {})
        dataset_adapter_data = data_copy.pop("dataset_adapter", {})
        training_strategy_data = data_copy.pop("training_strategy", {})
        extensions_data = data_copy.pop("extensions", {})
        optimization_data = data_copy.pop("optimization", {})
        checkpointing_data = data_copy.pop("checkpointing", {})
        
        # Create component instances
        model_adapter = ModelAdapterConfig(**model_adapter_data) if model_adapter_data else ModelAdapterConfig()
        dataset_adapter = DatasetAdapterConfig(**dataset_adapter_data) if dataset_adapter_data else DatasetAdapterConfig()
        training_strategy = TrainingStrategyConfig(**training_strategy_data) if training_strategy_data else TrainingStrategyConfig()
        extensions = ExtensionConfig(**extensions_data) if extensions_data else ExtensionConfig()
        optimization = OptimizationConfig(**optimization_data) if optimization_data else OptimizationConfig()
        checkpointing = CheckpointConfig(**checkpointing_data) if checkpointing_data else CheckpointConfig()
        
        # Filter the remaining data to only include fields from the base class
        filtered_data = {
            k: v for k, v in data_copy.items()
            if hasattr(cls, k) and not k.startswith("_")
        }
        
        # Create and return the instance
        config = cls(**filtered_data)
        config.model_adapter = model_adapter
        config.dataset_adapter = dataset_adapter
        config.training_strategy = training_strategy
        config.extensions = extensions
        config.optimization = optimization
        config.checkpointing = checkpointing
        
        return config
    
    def update_from_base_config(self, base_config: BaseTrainingConfig) -> "EnhancedTrainingConfig":
        """
        Update this configuration with values from a base training config.
        
        Args:
            base_config: Base training configuration
            
        Returns:
            Updated enhanced training configuration
        """
        # Update base fields
        for field_name, field_value in base_config.__dict__.items():
            if hasattr(self, field_name) and not field_name.startswith("_"):
                setattr(self, field_name, field_value)
        
        # Update derived fields from base config
        self._set_defaults_from_training_type()
        
        return self