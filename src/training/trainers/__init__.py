"""
Specialized trainers for QLLM.

This module provides specialized trainer implementations for different
training scenarios, extending the base trainer with specific behaviors.
"""

from src.training.trainers.text_trainer import TextTrainer
from src.training.trainers.empathy_trainer import EmpathyTrainer
from src.training.trainers.intent_trainer import IntentTrainer
from src.training.trainers.dialogue_trainer import DialogueTrainer
from src.training.trainers.function_call_trainer import FunctionCallTrainer
from src.training.trainers.structured_output_trainer import StructuredOutputTrainer

__all__ = [
    'TextTrainer',
    'EmpathyTrainer',
    'IntentTrainer',
    'DialogueTrainer',
    'FunctionCallTrainer',
    'StructuredOutputTrainer'
]