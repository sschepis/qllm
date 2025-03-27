"""
Training package for QLLM.

This package contains the training functionality for the Quantum Resonance
Language Model, including various trainer implementations for different
training approaches (standard, dialogue, verbose).
"""

# Import the TrainerFactory first since it depends on the trainer implementations
from src.training.trainer_factory import TrainerFactory

# Import trainer base class and implementations
from src.training.base_trainer import BaseTrainer
from src.training.standard_trainer import StandardTrainer
from src.training.dialogue_trainer import DialogueTrainer
from src.training.verbose_trainer import VerboseTrainer

__all__ = [
    'TrainerFactory',
    'BaseTrainer',
    'StandardTrainer',
    'DialogueTrainer',
    'VerboseTrainer',
]