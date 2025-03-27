"""
Training system for Quantum Resonance Language Models.

This package provides a modular, extensible training system that supports
various model types, training strategies, and extensions.
"""

# Base trainers
from src.training.base_trainer import BaseTrainer
from src.training.enhanced_trainer import EnhancedTrainer
from src.training.standard_trainer import StandardTrainer
from src.training.dialogue_trainer import DialogueTrainer

# Factory functions
from src.training.trainer_factory import (
    get_trainer,
    create_enhanced_trainer,
    create_trainer_for_model_type
)

# Subpackages
from src.training.model_adapters import (
    ModelAdapter,
    StandardModelAdapter,
    DialogueModelAdapter,
    MultimodalModelAdapter,
    get_model_adapter
)

from src.training.strategies import (
    TrainingStrategy,
    StandardTrainingStrategy,
    FinetuningStrategy,
    get_training_strategy
)

from src.training.extensions import (
    ExtensionHooks,
    ExtensionManager
)

from src.training.checkpoints import (
    CheckpointManager
)

from src.training.metrics import (
    MetricsLogger
)

__all__ = [
    # Trainers
    'BaseTrainer',
    'EnhancedTrainer',
    'StandardTrainer',
    'DialogueTrainer',
    
    # Factories
    'get_trainer',
    'create_enhanced_trainer',
    'create_trainer_for_model_type',
    
    # Model adapters
    'ModelAdapter',
    'StandardModelAdapter',
    'DialogueModelAdapter',
    'MultimodalModelAdapter',
    'get_model_adapter',
    
    # Training strategies
    'TrainingStrategy',
    'StandardTrainingStrategy',
    'FinetuningStrategy',
    'get_training_strategy',
    
    # Extensions
    'ExtensionHooks',
    'ExtensionManager',
    
    # Checkpoints
    'CheckpointManager',
    
    # Metrics
    'MetricsLogger',
]