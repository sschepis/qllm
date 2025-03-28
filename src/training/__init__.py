"""
Training module for QLLM.

This module provides trainers and utilities for training QLLM models.
It has been refactored to consolidate duplicated code into a robust
base trainer class with specialized trainers for different use cases.
"""

from src.training.base_trainer import BaseTrainer
from src.training.unified_trainer import UnifiedTrainer
from src.training.trainer_factory import TrainerFactory
from src.training.trainers import (
    TextTrainer,
    EmpathyTrainer, 
    IntentTrainer,
    DialogueTrainer,
    FunctionCallTrainer,
    StructuredOutputTrainer
)

__all__ = [
    # Base trainer
    'BaseTrainer',
    
    # Unified trainer
    'UnifiedTrainer',
    
    # Factory
    'TrainerFactory',
    
    # Specialized trainers
    'TextTrainer',
    'EmpathyTrainer',
    'IntentTrainer',
    'DialogueTrainer',
    'FunctionCallTrainer',
    'StructuredOutputTrainer'
]