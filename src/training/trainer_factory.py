"""
Trainer factory for the Quantum Resonance Language Model.

This module provides a factory for creating the appropriate trainer
based on the training type specified in the configuration.
"""

import os
import logging
from typing import Dict, Any, Optional, Union

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.config.data_config import DataConfig
from src.training.standard_trainer import StandardTrainer
from src.training.dialogue_trainer import DialogueTrainer
from src.training.verbose_trainer import VerboseTrainer


class TrainerFactory:
    """Factory for creating trainer instances."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the trainer factory.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("qllm_trainer_factory")
    
    def create_trainer(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        output_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Create a trainer instance based on the training type.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            data_config: Data configuration
            output_dir: Directory for outputs
            logger: Logger instance
            
        Returns:
            Trainer instance
            
        Raises:
            ValueError: If the specified training type is not supported
        """
        # Determine the effective training type, prioritizing dataset type over specified training type
        effective_training_type = training_config.training_type.lower()
        
        # Override training type based on dataset if appropriate
        dataset_name = getattr(data_config, "dataset_name", "").lower()
        if dataset_name == "daily_dialog":
            self.logger.info("Dataset type is 'daily_dialog', using DialogueTrainer regardless of specified training type")
            effective_training_type = "dialogue"
            # Update training_config for consistency
            training_config.training_type = "dialogue"
        elif dataset_name == "custom" and getattr(data_config, "system_prompt", None):
            self.logger.info("Custom dataset with system prompt detected, using DialogueTrainer")
            effective_training_type = "dialogue"
            # Update training_config for consistency
            training_config.training_type = "dialogue"
        
        self.logger.info(f"Creating trainer with type: {effective_training_type}")
        
        if effective_training_type == "standard":
            # Create standard trainer
            return StandardTrainer(
                model_config=model_config,
                training_config=training_config,
                data_config=data_config,
                output_dir=output_dir,
                logger=logger or self.logger
            )
        
        elif effective_training_type == "dialogue":
            # Create dialogue trainer
            return DialogueTrainer(
                model_config=model_config,
                training_config=training_config,
                data_config=data_config,
                output_dir=output_dir,
                logger=logger or self.logger
            )
        
        elif effective_training_type == "verbose":
            # Create verbose trainer
            return VerboseTrainer(
                model_config=model_config,
                training_config=training_config,
                data_config=data_config,
                output_dir=output_dir,
                logger=logger or self.logger
            )
        
        else:
            # Invalid training type
            error_msg = f"Unsupported training type: {effective_training_type}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    def get_available_trainers(self) -> Dict[str, str]:
        """
        Get available trainer types with descriptions.
        
        Returns:
            Dictionary of trainer types and descriptions
        """
        return {
            "standard": "Standard language model training",
            "dialogue": "Dialogue-based training for conversational models",
            "verbose": "Detailed training with comprehensive logging and diagnostics"
        }