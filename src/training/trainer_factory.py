"""
Trainer Factory for QLLM.

This module provides a factory for creating appropriate trainers based on
configuration and training scenarios. It has been enhanced to support the
refactored trainer structure and handle various specialized trainers.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Type

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.training.base_trainer import BaseTrainer
from src.training.unified_trainer import UnifiedTrainer
from src.training.trainers.text_trainer import TextTrainer
from src.training.trainers.dialogue_trainer import DialogueTrainer
from src.training.trainers.empathy_trainer import EmpathyTrainer
from src.training.trainers.intent_trainer import IntentTrainer
from src.training.trainers.function_call_trainer import FunctionCallTrainer
from src.training.trainers.structured_output_trainer import StructuredOutputTrainer
from src.config.training_config import TrainingConfig
from src.config.model_config import ModelConfig
from src.config.data_config import DataConfig


logger = logging.getLogger("qllm.training")


class TrainerFactory:
    """
    Factory for creating trainers based on configuration.
    
    This factory creates the appropriate trainer based on the provided
    configuration and training scenario, simplifying the process of
    setting up training for different use cases.
    """
    
    # Registry of trainer types
    TRAINER_TYPES = {
        "base": BaseTrainer,
        "unified": UnifiedTrainer,
        "text": TextTrainer,
        "dialogue": DialogueTrainer,
        "empathy": EmpathyTrainer,
        "intent": IntentTrainer,
        "function_call": FunctionCallTrainer,
        "structured_output": StructuredOutputTrainer
    }
    
    @classmethod
    def create_trainer(
        cls,
        model: nn.Module,
        train_dataloader: DataLoader,
        training_config: TrainingConfig,
        model_config: Optional[ModelConfig] = None,
        data_config: Optional[DataConfig] = None,
        eval_dataloader: Optional[DataLoader] = None,
        trainer_type: Optional[str] = None,
        **kwargs
    ) -> BaseTrainer:
        """
        Create a trainer instance based on the provided configuration.
        
        Args:
            model: Model to train
            train_dataloader: DataLoader for training data
            training_config: Training configuration
            model_config: Optional model configuration
            data_config: Optional data configuration
            eval_dataloader: Optional DataLoader for evaluation data
            trainer_type: Type of trainer to create (overrides auto-detection)
            **kwargs: Additional arguments for the trainer
            
        Returns:
            Configured trainer instance
            
        Raises:
            ValueError: If the requested trainer type is not supported
        """
        # Determine trainer type if not explicitly provided
        if trainer_type is None:
            trainer_type = cls._detect_trainer_type(
                model, training_config, data_config
            )
        
        # Validate trainer type
        if trainer_type not in cls.TRAINER_TYPES:
            valid_types = ", ".join(cls.TRAINER_TYPES.keys())
            raise ValueError(
                f"Unsupported trainer type: {trainer_type}. "
                f"Valid types are: {valid_types}"
            )
        
        # Get trainer class
        trainer_class = cls.TRAINER_TYPES[trainer_type]
        
        # Create trainer instance
        trainer = trainer_class(
            model=model,
            train_dataloader=train_dataloader,
            training_config=training_config,
            model_config=model_config,
            data_config=data_config,
            eval_dataloader=eval_dataloader,
            **kwargs
        )
        
        logger.info(f"Created {trainer_class.__name__} for training")
        return trainer
    
    @classmethod
    def _detect_trainer_type(
        cls,
        model: nn.Module,
        training_config: TrainingConfig,
        data_config: Optional[DataConfig] = None
    ) -> str:
        """
        Detect the appropriate trainer type based on model and configuration.
        
        Args:
            model: Model to train
            training_config: Training configuration
            data_config: Optional data configuration
            
        Returns:
            Detected trainer type
        """
        # Check if training_config explicitly specifies training strategy
        if hasattr(training_config, "training_strategy"):
            strategy = training_config.training_strategy
            
            # Map strategies to trainer types
            if strategy == "dialogue":
                return "dialogue"
            elif strategy in ["pretrain", "finetune", "standard"]:
                # For general text training scenarios
                return "text"
        
        # Check if model has extensions that indicate function calling
        if hasattr(model, "extensions") and hasattr(model.extensions, "get"):
            if model.extensions.get("function_calling", False):
                return "function_call"
        
        # Check model type from config
        if hasattr(training_config, "model_type"):
            model_type = training_config.model_type
            
            if model_type == "dialogue":
                return "dialogue"
            elif model_type == "intent":
                return "intent"
            elif model_type == "empathy":
                return "empathy"
            elif model_type == "structured_output":
                return "structured_output"
        
        # Check data config for clues
        if data_config is not None:
            # Check if dialogue mode is enabled
            if hasattr(data_config, "dialogue_mode") and data_config.dialogue_mode:
                return "dialogue"
        
        # Default to unified trainer for flexibility
        return "unified"
    
    @classmethod
    def register_trainer_type(cls, name: str, trainer_class: Type[BaseTrainer]) -> None:
        """
        Register a new trainer type.
        
        Args:
            name: Name for the trainer type
            trainer_class: Trainer class to register
            
        Raises:
            TypeError: If the trainer class doesn't inherit from BaseTrainer
        """
        # Validate trainer class
        if not issubclass(trainer_class, BaseTrainer):
            raise TypeError(
                f"Trainer class must inherit from BaseTrainer, got {trainer_class.__name__}"
            )
        
        # Register trainer type
        cls.TRAINER_TYPES[name] = trainer_class
        logger.info(f"Registered new trainer type: {name} -> {trainer_class.__name__}")
    
    @classmethod
    def create_from_checkpoint(
        cls,
        checkpoint_path: str,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        trainer_type: Optional[str] = None,
        **kwargs
    ) -> BaseTrainer:
        """
        Create a trainer from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint
            model: Model to train
            train_dataloader: DataLoader for training data
            eval_dataloader: Optional DataLoader for evaluation data
            trainer_type: Type of trainer to create (overrides auto-detection)
            **kwargs: Additional arguments for the trainer
            
        Returns:
            Configured trainer instance with loaded state
            
        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist
            ValueError: If the checkpoint doesn't contain configuration
        """
        # Validate checkpoint path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Load checkpoint to extract configuration
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract configurations
        if "config" in checkpoint_data:
            config = checkpoint_data["config"]
            training_config = TrainingConfig.from_dict(config.get("training", {}))
            model_config = ModelConfig.from_dict(config.get("model", {}))
            data_config = DataConfig.from_dict(config.get("data", {}))
        else:
            # Try to extract from separate keys
            if "training_config" in checkpoint_data:
                training_config = TrainingConfig.from_dict(checkpoint_data["training_config"])
            else:
                raise ValueError("Checkpoint doesn't contain training configuration")
            
            model_config = None
            if "model_config" in checkpoint_data:
                model_config = ModelConfig.from_dict(checkpoint_data["model_config"])
            
            data_config = None
            if "data_config" in checkpoint_data:
                data_config = DataConfig.from_dict(checkpoint_data["data_config"])
        
        # Create trainer with extracted configuration
        trainer = cls.create_trainer(
            model=model,
            train_dataloader=train_dataloader,
            training_config=training_config,
            model_config=model_config,
            data_config=data_config,
            eval_dataloader=eval_dataloader,
            trainer_type=trainer_type,
            resume_from_checkpoint=checkpoint_path,
            **kwargs
        )
        
        logger.info(f"Created trainer from checkpoint: {checkpoint_path}")
        return trainer