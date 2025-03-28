"""
Unified training system for QLLM.

This package provides a flexible, modular, composition-based training system
with better structure and improved extensibility compared to the original
implementations.
"""

from src.training.unified_trainer import UnifiedTrainer
from src.training.enhanced_trainer import EnhancedTrainer
from src.training.trainer_factory import (
    TrainerFactory,
    get_trainer,
    create_trainer_for_model_type,
    create_unified_trainer
)

__all__ = [
    'UnifiedTrainer',  # Primary trainer implementation
    'EnhancedTrainer',  # Legacy implementation (deprecated)
    'TrainerFactory',
    'get_trainer',
    'create_trainer_for_model_type',
    'create_unified_trainer',
]