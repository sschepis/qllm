"""
Training strategies for the enhanced training system.

This package contains strategies that implement different
training approaches for language models.
"""

from src.training.strategies.base_strategy import TrainingStrategy
from src.training.strategies.standard_strategy import StandardTrainingStrategy
from src.training.strategies.finetune_strategy import FinetuningStrategy

__all__ = [
    'TrainingStrategy',
    'StandardTrainingStrategy',
    'FinetuningStrategy',
]

# Strategy registry
TRAINING_STRATEGIES = {
    'standard': StandardTrainingStrategy,
    'finetune': FinetuningStrategy,
}

def get_training_strategy(strategy_type, *args, **kwargs):
    """
    Get the appropriate training strategy for the given type.
    
    Args:
        strategy_type: Type of training strategy ('standard', 'finetune', etc.)
        *args: Arguments to pass to the strategy constructor
        **kwargs: Keyword arguments to pass to the strategy constructor
        
    Returns:
        Initialized training strategy instance
        
    Raises:
        ValueError: If the strategy type is not supported
    """
    if strategy_type not in TRAINING_STRATEGIES:
        raise ValueError(f"Unsupported training strategy: {strategy_type}. "
                         f"Available strategies: {list(TRAINING_STRATEGIES.keys())}")
    
    return TRAINING_STRATEGIES[strategy_type](*args, **kwargs)