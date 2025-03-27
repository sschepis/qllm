"""
Training strategies for the enhanced training system.

This package provides implementation of various training strategies for different
scenarios like standard training, finetuning, and specialized techniques.
"""

import logging
from typing import Any, Optional, Dict, Union, Type

from src.training.strategies.base_strategy import TrainingStrategy
from src.training.strategies.standard_strategy import StandardTrainingStrategy
from src.training.strategies.finetune_strategy import FinetuningStrategy

# Try to import specialized strategies
try:
    from src.training.strategies.dialogue_strategy import DialogueTrainingStrategy
except ImportError:
    # Create placeholder class if not available
    class DialogueTrainingStrategy(StandardTrainingStrategy):
        """Placeholder for DialogueTrainingStrategy."""
        pass

try:
    from src.training.strategies.multimodal_strategy import MultimodalTrainingStrategy
except ImportError:
    # Create placeholder class if not available
    class MultimodalTrainingStrategy(StandardTrainingStrategy):
        """Placeholder for MultimodalTrainingStrategy."""
        pass


def get_training_strategy(
    strategy_type: str,
    config: Any,
    logger: Optional[logging.Logger] = None
) -> TrainingStrategy:
    """
    Create a training strategy based on strategy type.
    
    Args:
        strategy_type: Type of training strategy ('standard', 'finetune', etc.)
        config: Training configuration
        logger: Logger instance
        
    Returns:
        Initialized training strategy
        
    Raises:
        ValueError: If strategy_type is not supported
    """
    # Create logger if not provided
    if logger is None:
        logger = logging.getLogger("quantum_resonance")
    
    # Map strategy type to strategy class
    strategy_map = {
        "standard": StandardTrainingStrategy,
        "finetune": FinetuningStrategy,
        "dialogue": DialogueTrainingStrategy,
        "multimodal": MultimodalTrainingStrategy
    }
    
    # Normalize strategy type
    strategy_type = strategy_type.lower()
    
    # Check if strategy type is supported
    if strategy_type not in strategy_map:
        raise ValueError(
            f"Unsupported training strategy: {strategy_type}. "
            f"Supported strategies: {', '.join(strategy_map.keys())}"
        )
    
    # Get strategy class
    strategy_class = strategy_map[strategy_type]
    
    # Create strategy instance
    logger.info(f"Creating training strategy: {strategy_type}")
    return strategy_class(
        config=config,
        logger=logger
    )


__all__ = [
    'TrainingStrategy',
    'StandardTrainingStrategy',
    'FinetuningStrategy',
    'DialogueTrainingStrategy',
    'MultimodalTrainingStrategy',
    'get_training_strategy',
]