"""
Trainer factory for creating appropriate trainer instances.

This module provides factory functions for creating trainer instances
based on configuration, simplifying the trainer initialization process.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Type

import torch

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.config.data_config import DataConfig

from src.training.base_trainer import BaseTrainer
from src.training.enhanced_trainer import EnhancedTrainer
from src.training.standard_trainer import StandardTrainer
from src.training.dialogue_trainer import DialogueTrainer
from src.training.model_adapters import get_model_adapter, ModelAdapter
from src.training.strategies import get_training_strategy, TrainingStrategy
from src.training.extensions import ExtensionManager
from src.training.checkpoints import CheckpointManager
from src.training.metrics import MetricsLogger


def get_trainer(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: DataConfig,
    output_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    trainer_type: Optional[str] = None
) -> Union[BaseTrainer, EnhancedTrainer]:
    """
    Create a trainer instance based on configuration.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        data_config: Data configuration
        output_dir: Directory for outputs
        logger: Logger instance
        trainer_type: Type of trainer to create (default: from training_config)
        
    Returns:
        Initialized trainer instance
        
    Raises:
        ValueError: If trainer_type is not supported
    """
    # Determine trainer type from config if not provided
    if trainer_type is None:
        trainer_type = getattr(training_config, "trainer_type", "enhanced")
    
    # Create logger if not provided
    if logger is None:
        logger = logging.getLogger("quantum_resonance")
    
    # Create trainer based on type
    trainer_type = trainer_type.lower()
    
    if trainer_type == "enhanced":
        return create_enhanced_trainer(
            model_config, training_config, data_config, output_dir, logger
        )
    elif trainer_type == "standard":
        return StandardTrainer(
            model_config, training_config, data_config, output_dir, logger
        )
    elif trainer_type == "dialogue":
        return DialogueTrainer(
            model_config, training_config, data_config, output_dir, logger
        )
    elif trainer_type == "base":
        return BaseTrainer(
            model_config, training_config, data_config, output_dir, logger
        )
    else:
        raise ValueError(f"Unsupported trainer type: {trainer_type}")


def create_enhanced_trainer(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: DataConfig,
    output_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> EnhancedTrainer:
    """
    Create an enhanced trainer with all components.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        data_config: Data configuration
        output_dir: Directory for outputs
        logger: Logger instance
        
    Returns:
        Initialized enhanced trainer
    """
    # Create logger if not provided
    if logger is None:
        logger = logging.getLogger("quantum_resonance")
    
    # Determine device
    device = _get_device(training_config)
    
    # Create model adapter
    model_type = getattr(training_config, "model_type", "standard")
    model_adapter = get_model_adapter(
        model_type,
        model_config,
        training_config,
        device,
        logger
    )
    
    # Create training strategy
    strategy_type = getattr(training_config, "training_strategy", "standard")
    training_strategy = get_training_strategy(
        strategy_type,
        training_config,
        logger
    )
    
    # Create extension manager
    extension_manager = ExtensionManager(
        training_config,
        logger
    )
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        training_config,
        output_dir,
        logger
    )
    
    # Create metrics logger
    metrics_logger = MetricsLogger(
        output_dir or getattr(training_config, "output_dir", "runs/quantum_resonance"),
        log_to_console=True,
        log_to_tensorboard=getattr(training_config, "use_tensorboard", True),
        log_to_file=True,
        logger=logger
    )
    
    # Create enhanced trainer
    return EnhancedTrainer(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        output_dir=output_dir,
        logger=logger,
        model_adapter=model_adapter,
        training_strategy=training_strategy,
        extension_manager=extension_manager,
        checkpoint_manager=checkpoint_manager,
        metrics_logger=metrics_logger
    )


def create_trainer_for_model_type(
    model_type: str,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: DataConfig,
    output_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> EnhancedTrainer:
    """
    Create a trainer optimized for a specific model type.
    
    Args:
        model_type: Type of model ('standard', 'dialogue', 'multimodal')
        model_config: Model configuration
        training_config: Training configuration
        data_config: Data configuration
        output_dir: Directory for outputs
        logger: Logger instance
        
    Returns:
        Initialized trainer for the specific model type
    """
    # Set model type in training config
    setattr(training_config, "model_type", model_type)
    
    # Choose appropriate strategy based on model type
    strategy_map = {
        "standard": "standard",
        "dialogue": "standard",
        "multimodal": "standard",
        "finetune": "finetune"
    }
    
    # Set strategy type in training config
    strategy_type = getattr(training_config, "training_strategy", None)
    if strategy_type is None:
        strategy_type = strategy_map.get(model_type, "standard")
        setattr(training_config, "training_strategy", strategy_type)
    
    # Create enhanced trainer
    return create_enhanced_trainer(
        model_config, training_config, data_config, output_dir, logger
    )


def _get_device(training_config: TrainingConfig) -> torch.device:
    """
    Get device from training configuration.
    
    Args:
        training_config: Training configuration
        
    Returns:
        Device to use
    """
    from src.utils.device import get_device
    return get_device(getattr(training_config, "device", None))