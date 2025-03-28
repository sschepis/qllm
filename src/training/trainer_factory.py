"""
Factory functions for the enhanced training system.

This module provides factory functions and factory class for creating trainers and related
components with the appropriate configuration and dependencies.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Type

import torch
import torch.nn as nn

from src.training.trainer_core import TrainerCore
from src.training.unified_trainer import UnifiedTrainer
from src.training.enhanced_trainer import EnhancedTrainer
from src.training.model_adapters.base_adapter import ModelAdapter
from src.training.model_adapters.standard_adapter import StandardModelAdapter
from src.training.model_adapters.dialogue_adapter import DialogueModelAdapter
from src.training.model_adapters.multimodal_adapter import MultimodalModelAdapter
from src.training.dataset_adapters.base_adapter import DatasetAdapter
from src.training.dataset_adapters.standard_adapter import StandardDatasetAdapter
from src.training.dataset_adapters.dialogue_adapter import DialogueDatasetAdapter
from src.training.dataset_adapters.multimodal_adapter import MultimodalDatasetAdapter
from src.training.strategies.base_strategy import TrainingStrategy
from src.training.strategies.standard_strategy import StandardTrainingStrategy
from src.training.strategies.finetune_strategy import FinetuningStrategy
from src.training.extensions.extension_manager import ExtensionManager
from src.training.checkpoints.checkpoint_manager import CheckpointManager
from src.training.metrics.metrics_logger import MetricsLogger
from src.training.config.training_config import EnhancedTrainingConfig
from src.config.training_config import TrainingConfig
from src.config.model_config import ModelConfig


# Get logger
logger = logging.getLogger("quantum_resonance")


def create_model_adapter(
    config: Union[EnhancedTrainingConfig, Dict[str, Any]],
    model_config: Optional[ModelConfig] = None,
    device: Optional[torch.device] = None
) -> ModelAdapter:
    """
    Create a model adapter based on configuration.
    
    Args:
        config: Training configuration
        model_config: Model configuration
        device: Device to use
        
    Returns:
        Initialized model adapter
    """
    # Determine adapter type
    if isinstance(config, dict):
        adapter_type = config.get("model_adapter", {}).get("adapter_type", "standard")
        if adapter_type == "standard":
            adapter_type = config.get("model_type", "standard")
    elif isinstance(config, EnhancedTrainingConfig):
        adapter_type = config.model_adapter.adapter_type
    else:  # Handle legacy TrainingConfig
        adapter_type = getattr(config, "model_type", "standard")
    
    adapter_type = adapter_type.lower()
    
    # Create the appropriate adapter
    if adapter_type == "dialogue":
        return DialogueModelAdapter(
            model_config=model_config,
            training_config=config,
            device=device
        )
    elif adapter_type == "multimodal":
        return MultimodalModelAdapter(
            model_config=model_config,
            training_config=config,
            device=device
        )
    else:  # Default to standard
        return StandardModelAdapter(
            model_config=model_config,
            training_config=config,
            device=device
        )


def create_dataset_adapter(
    config: Union[EnhancedTrainingConfig, Dict[str, Any]],
    tokenizer: Optional[Any] = None,
    image_processor: Optional[Any] = None
) -> DatasetAdapter:
    """
    Create a dataset adapter based on configuration.
    
    Args:
        config: Training configuration
        tokenizer: Tokenizer for text processing
        image_processor: Image processor for multimodal data
        
    Returns:
        Initialized dataset adapter
    """
    # Determine adapter type
    if isinstance(config, dict):
        adapter_type = config.get("dataset_adapter", {}).get("adapter_type", "standard")
        if adapter_type == "standard":
            adapter_type = config.get("model_type", "standard")
    elif isinstance(config, EnhancedTrainingConfig):
        adapter_type = config.dataset_adapter.adapter_type
    else:  # Handle legacy TrainingConfig
        adapter_type = getattr(config, "model_type", "standard")
    
    adapter_type = adapter_type.lower()
    
    # Create the appropriate adapter
    if adapter_type == "dialogue":
        return DialogueDatasetAdapter(
            config=config,
            tokenizer=tokenizer,
            max_seq_length=config.dataset_adapter.model_adapter_kwargs.get("max_sequence_length", 512) 
                if hasattr(config, "dataset_adapter") else 512
        )
    elif adapter_type == "multimodal":
        return MultimodalDatasetAdapter(
            config=config,
            tokenizer=tokenizer,
            image_processor=image_processor
        )
    else:  # Default to standard
        return StandardDatasetAdapter(
            config=config,
            tokenizer=tokenizer
        )


def create_training_strategy(
    config: Union[EnhancedTrainingConfig, Dict[str, Any]]
) -> TrainingStrategy:
    """
    Create a training strategy based on configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Initialized training strategy
    """
    # Determine strategy type
    if isinstance(config, dict):
        strategy_type = config.get("training_strategy", {}).get("strategy_type", "standard")
        if strategy_type == "standard":
            strategy_type = config.get("training_strategy", "standard")
    elif isinstance(config, EnhancedTrainingConfig):
        strategy_type = config.training_strategy.strategy_type
    else:  # Handle legacy TrainingConfig
        strategy_type = getattr(config, "training_strategy", "standard")
    
    strategy_type = strategy_type.lower()
    
    # Create the appropriate strategy
    if strategy_type == "finetune":
        return FinetuningStrategy(config=config)
    elif strategy_type == "pretrain":
        # No pretrain strategy implemented yet, fall back to standard
        logger.warning("Pretrain strategy requested but not implemented, falling back to standard")
        return StandardTrainingStrategy(config=config)
    else:  # Default to standard
        return StandardTrainingStrategy(config=config)


def create_extension_manager(
    config: Union[EnhancedTrainingConfig, Dict[str, Any]]
) -> Optional[ExtensionManager]:
    """
    Create an extension manager based on configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Initialized extension manager or None if no extensions are configured
    """
    # Check if extensions are configured
    if isinstance(config, dict):
        enabled_extensions = config.get("extensions", {}).get("enabled_extensions", [])
        extension_configs = config.get("extensions", {}).get("extension_configs", {})
    elif isinstance(config, EnhancedTrainingConfig):
        enabled_extensions = config.extensions.enabled_extensions
        extension_configs = config.extensions.extension_configs
    else:  # Handle legacy TrainingConfig
        extra_params = getattr(config, "extra_training_params", {})
        enabled_extensions = extra_params.get("extensions", [])
        extension_configs = extra_params.get("extension_configs", {})
    
    # Only create manager if extensions are enabled
    if enabled_extensions:
        return ExtensionManager(
            config=config.extensions if hasattr(config, "extensions") else config
        )
    return None


def create_metrics_logger(
    config: Union[EnhancedTrainingConfig, Dict[str, Any]]
) -> MetricsLogger:
    """
    Create a metrics logger based on configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Initialized metrics logger
    """
    # Get output directory
    if isinstance(config, dict):
        output_dir = config.get("output_dir", "runs/quantum_resonance")
        logging_steps = config.get("logging_steps", 10)
    elif isinstance(config, EnhancedTrainingConfig) or isinstance(config, TrainingConfig):
        output_dir = config.output_dir
        logging_steps = config.logging_steps
    else:
        output_dir = "runs/quantum_resonance"
        logging_steps = 10
    
    # Create metrics logger
    return MetricsLogger(
        log_dir=output_dir,
        logging_steps=logging_steps
    )


def create_checkpoint_manager(
    config: Union[EnhancedTrainingConfig, Dict[str, Any]]
) -> CheckpointManager:
    """
    Create a checkpoint manager based on configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Initialized checkpoint manager
    """
    # Get checkpoint directory
    if isinstance(config, dict):
        output_dir = config.get("output_dir", "runs/quantum_resonance")
        save_total_limit = config.get("checkpointing", {}).get("save_total_limit", 3)
    elif isinstance(config, EnhancedTrainingConfig):
        output_dir = config.output_dir
        save_total_limit = config.checkpointing.save_total_limit
    else:  # Handle legacy TrainingConfig
        output_dir = config.output_dir
        save_total_limit = getattr(config, "save_total_limit", 3)
    
    # Use checkpoint_dir from config if available
    if isinstance(config, EnhancedTrainingConfig):
        checkpoint_dir = config.checkpointing.checkpoint_kwargs.get("checkpoint_dir", None)
    elif isinstance(config, TrainingConfig):
        checkpoint_dir = getattr(config, "checkpoint_dir", None)
    else:
        checkpoint_dir = config.get("checkpoint_dir", None)
    
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
    
    # Create checkpoint manager
    return CheckpointManager(
        config=config,
        output_dir=checkpoint_dir,
        save_total_limit=save_total_limit
    )


def create_trainer(
    model: Optional[nn.Module] = None,
    config: Optional[Union[EnhancedTrainingConfig, TrainingConfig, Dict[str, Any]]] = None,
    model_config: Optional[ModelConfig] = None,
    model_adapter: Optional[ModelAdapter] = None,
    dataset_adapter: Optional[DatasetAdapter] = None,
    strategy: Optional[TrainingStrategy] = None,
    extension_manager: Optional[ExtensionManager] = None,
    metrics_logger: Optional[MetricsLogger] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    device: Optional[torch.device] = None,
    enable_memory_optimizations: bool = True,
    **kwargs  # Accept additional arguments and ignore them
) -> TrainerCore:
    """
    Create a trainer instance with all components.
    
    Args:
        model: Model to train (optional, will be created by model_adapter if not provided)
        config: Training configuration
        model_config: Model configuration
        model_adapter: Model adapter (optional, will be created from config if not provided)
        dataset_adapter: Dataset adapter (optional, will be created from config if not provided)
        strategy: Training strategy (optional, will be created from config if not provided)
        extension_manager: Extension manager (optional, will be created from config if not provided)
        metrics_logger: Metrics logger (optional, will be created from config if not provided)
        checkpoint_manager: Checkpoint manager (optional, will be created from config if not provided)
        device: Device to use for training
        enable_memory_optimizations: Whether to enable memory optimizations like gradient checkpointing
        
    Returns:
        Initialized trainer instance
    """
    # Default configuration if not provided
    if config is None:
        config = EnhancedTrainingConfig()
    
    # Convert legacy TrainingConfig to EnhancedTrainingConfig if needed
    if isinstance(config, TrainingConfig) and not isinstance(config, EnhancedTrainingConfig):
        enhanced_config = EnhancedTrainingConfig()
        enhanced_config.update_from_base_config(config)
        config = enhanced_config
    
    # Get device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model adapter if not provided
    if model_adapter is None:
        model_adapter = create_model_adapter(config, model_config, device)
    
    # Set model if provided
    if model is not None:
        model_adapter.set_model(model)
    
    # Create dataset adapter if not provided
    if dataset_adapter is None:
        dataset_adapter = create_dataset_adapter(
            config,
            tokenizer=model_adapter.get_tokenizer()
        )
    
    # Create training strategy if not provided
    if strategy is None:
        strategy = create_training_strategy(config)
    
    # Create extension manager if not provided
    if extension_manager is None:
        extension_manager = create_extension_manager(config)
    
    # Create metrics logger if not provided
    if metrics_logger is None:
        metrics_logger = create_metrics_logger(config)
    
    # Create checkpoint manager if not provided
    if checkpoint_manager is None:
        checkpoint_manager = create_checkpoint_manager(config)
    
    # Configure memory optimizations if enabled
    if enable_memory_optimizations and torch.cuda.is_available():
        # Enable if using EnhancedTrainingConfig
        if isinstance(config, EnhancedTrainingConfig) and hasattr(config.optimization, "use_gradient_checkpointing"):
            config.optimization.use_gradient_checkpointing = True
            config.optimization.enable_memory_efficient_eval = True
            config.optimization.auto_handle_oom = True
            logger.info("Memory optimizations enabled in configuration")
        
        # For model adapter's model, apply gradient checkpointing directly if needed
        if model_adapter and model_adapter.get_model():
            from src.training.optimization.memory_utils import enable_gradient_checkpointing
            enable_gradient_checkpointing(model_adapter.get_model())
            logger.info("Applied gradient checkpointing to model directly")
    
    # Create and return the trainer
    trainer = TrainerCore(
        config=config,
        model_adapter=model_adapter,
        dataset_adapter=dataset_adapter,
        strategy=strategy,
        extension_manager=extension_manager,
        metrics_logger=metrics_logger,
        checkpoint_manager=checkpoint_manager,
        device=device
    )
    
    # Auto-resume from checkpoint if configured
    if isinstance(config, EnhancedTrainingConfig):
        auto_resume = config.checkpointing.auto_resume
    else:
        auto_resume = getattr(config, "auto_resume", True)
    
    if auto_resume:
        trainer.load_checkpoint()
    
    return trainer


def create_trainer_from_config(
    config_path: str,
    model: Optional[nn.Module] = None,
    model_config: Optional[ModelConfig] = None
) -> TrainerCore:
    """
    Create a trainer from a configuration file.
    
    Args:
        config_path: Path to JSON or YAML configuration file
        model: Model to train (optional)
        model_config: Model configuration (optional)
        
    Returns:
        Initialized trainer instance
    """
    import json
    import os
    
    # Check if file exists
    if not os.path.exists(config_path):
        raise ValueError(f"Configuration file not found: {config_path}")
    
    # Load configuration
    with open(config_path, "r") as f:
        if config_path.endswith(".json"):
            config_dict = json.load(f)
        elif config_path.endswith((".yaml", ".yml")):
            try:
                import yaml
                config_dict = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required for loading YAML configurations. Install with 'pip install pyyaml'.")
        else:
            raise ValueError(f"Unsupported configuration format: {config_path}")
    
    # Create config object
    config = EnhancedTrainingConfig.from_dict(config_dict)
    
    # Create trainer
    return create_trainer(
        model=model,
        config=config,
        model_config=model_config
    )


# Factory class for creating trainers
class TrainerFactory:
    """
    Factory class for creating and configuring trainer instances.
    
    This class provides static methods for creating different types of trainers
    with appropriate configurations and components.
    """
    
    @staticmethod
    def create_trainer(
        model: Optional[nn.Module] = None,
        config: Optional[Union[EnhancedTrainingConfig, TrainingConfig, Dict[str, Any]]] = None,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[Union[EnhancedTrainingConfig, TrainingConfig]] = None,
        data_config: Optional[Any] = None,
        **kwargs
    ) -> TrainerCore:
        """
        Create a trainer instance with the given model and configuration.
        
        Args:
            model: Model to train
            config: Training configuration
            model_config: Model configuration
            training_config: Alternative name for training configuration (for compatibility)
            data_config: Data configuration (for compatibility)
            **kwargs: Additional arguments
            
        Returns:
            Configured trainer instance
        """
        # Use config if provided, otherwise use training_config
        effective_config = config if config is not None else training_config
        
        # Filter out parameters that should not be passed to the underlying function
        filtered_kwargs = kwargs.copy()
        for param in ['training_config', 'data_config']:
            if param in filtered_kwargs:
                filtered_kwargs.pop(param)
        
        # Handle data_config: in the original system, this might be needed elsewhere
        # but we'll include it in our kwargs to capture it
        augmented_kwargs = {}
        if data_config is not None:
            augmented_kwargs['data_config'] = data_config
            
        # Create the underlying trainer with the processed parameters
        return create_trainer(
            model=model,
            config=effective_config,
            model_config=model_config,
            **filtered_kwargs
        )
    
    @staticmethod
    def get_trainer(
        trainer_type: str,
        config: Union[EnhancedTrainingConfig, TrainingConfig],
        model: Optional[nn.Module] = None,
        model_config: Optional[ModelConfig] = None
    ) -> Union[TrainerCore, UnifiedTrainer, EnhancedTrainer]:
        """
        Get a trainer instance by type.
        
        Args:
            trainer_type: Type of trainer
            config: Training configuration
            model: Model to train
            model_config: Model configuration
            
        Returns:
            Configured trainer instance
        """
        trainer_type = trainer_type.lower()
        
        if trainer_type == "unified":
            return TrainerFactory.create_unified_trainer(config, model)
        elif trainer_type == "enhanced":
            return EnhancedTrainer(config, model)
        else:
            return get_trainer_from_name(trainer_type, config, model, model_config)
    
    @staticmethod
    def create_trainer_for_model_type(
        model_type: str,
        config: Union[EnhancedTrainingConfig, TrainingConfig],
        model: Optional[nn.Module] = None,
        model_config: Optional[ModelConfig] = None
    ) -> TrainerCore:
        """
        Create a trainer specialized for a specific model type.
        
        Args:
            model_type: Type of model (standard, dialogue, multimodal)
            config: Training configuration
            model: Model to train
            model_config: Model configuration
            
        Returns:
            Configured trainer instance
        """
        # Configure the adapters based on model type
        if isinstance(config, EnhancedTrainingConfig):
            if model_type == "dialogue":
                config.model_adapter.adapter_type = "dialogue"
                config.dataset_adapter.adapter_type = "dialogue"
            elif model_type == "multimodal":
                config.model_adapter.adapter_type = "multimodal"
                config.dataset_adapter.adapter_type = "multimodal"
            else:
                config.model_adapter.adapter_type = "standard"
                config.dataset_adapter.adapter_type = "standard"
        
        return create_trainer(
            model=model,
            config=config,
            model_config=model_config
        )
    
    @staticmethod
    def create_unified_trainer(
        config: Union[EnhancedTrainingConfig, TrainingConfig],
        model: Optional[nn.Module] = None
    ) -> UnifiedTrainer:
        """
        Create a unified trainer instance.
        
        Args:
            config: Training configuration
            model: Model to train
            
        Returns:
            UnifiedTrainer instance
        """
        return UnifiedTrainer(config, model)


# Helper functions that can be used directly
def get_trainer(
    trainer_type: str,
    config: Union[EnhancedTrainingConfig, TrainingConfig],
    model: Optional[nn.Module] = None,
    model_config: Optional[ModelConfig] = None
) -> Union[TrainerCore, UnifiedTrainer, EnhancedTrainer]:
    """
    Get a trainer instance by type.
    
    Args:
        trainer_type: Type of trainer
        config: Training configuration
        model: Model to train
        model_config: Model configuration
        
    Returns:
        Configured trainer instance
    """
    return TrainerFactory.get_trainer(trainer_type, config, model, model_config)


def create_trainer_for_model_type(
    model_type: str,
    config: Union[EnhancedTrainingConfig, TrainingConfig],
    model: Optional[nn.Module] = None,
    model_config: Optional[ModelConfig] = None
) -> TrainerCore:
    """
    Create a trainer specialized for a specific model type.
    
    Args:
        model_type: Type of model (standard, dialogue, multimodal)
        config: Training configuration
        model: Model to train
        model_config: Model configuration
        
    Returns:
        Configured trainer instance
    """
    return TrainerFactory.create_trainer_for_model_type(model_type, config, model, model_config)


def create_unified_trainer(
    config: Union[EnhancedTrainingConfig, TrainingConfig],
    model: Optional[nn.Module] = None
) -> UnifiedTrainer:
    """
    Create a unified trainer instance.
    
    Args:
        config: Training configuration
        model: Model to train
        
    Returns:
        UnifiedTrainer instance
    """
    return TrainerFactory.create_unified_trainer(config, model)


def get_trainer_from_name(
    trainer_name: str,
    config: Union[EnhancedTrainingConfig, TrainingConfig],
    model: Optional[nn.Module] = None,
    model_config: Optional[ModelConfig] = None
) -> TrainerCore:
    """
    Get a trainer instance by name.
    
    Args:
        trainer_name: Name of the trainer type
        config: Training configuration
        model: Model to train (optional)
        model_config: Model configuration (optional)
        
    Returns:
        Initialized trainer instance
    """
    trainer_name = trainer_name.lower()
    
    # For now, all trainers use TrainerCore with different configurations
    return create_trainer(
        model=model,
        config=config,
        model_config=model_config
    )