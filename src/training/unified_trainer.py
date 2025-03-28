"""
Unified Trainer for QLLM.

This module provides a unified trainer implementation that simplifies the
training interface while leveraging the robust base trainer functionality.
It supports multiple training scenarios through a consistent interface.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Type

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.training.base_trainer import BaseTrainer
from src.config.training_config import TrainingConfig
from src.config.model_config import ModelConfig
from src.config.data_config import DataConfig


logger = logging.getLogger("qllm.training")


class UnifiedTrainer(BaseTrainer):
    """
    Unified trainer that supports multiple training scenarios.
    
    This trainer provides a simplified interface for training different types
    of models, extending the base trainer with specialized functionality for
    different training scenarios like text generation, dialogue, empathy, etc.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        training_config: TrainingConfig,
        model_config: Optional[ModelConfig] = None,
        data_config: Optional[DataConfig] = None,
        eval_dataloader: Optional[DataLoader] = None,
        training_type: str = "standard",
        metrics_callback: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize the unified trainer.
        
        Args:
            model: Model to train
            train_dataloader: DataLoader for training data
            training_config: Training configuration
            model_config: Optional model configuration
            data_config: Optional data configuration
            eval_dataloader: Optional DataLoader for evaluation data
            training_type: Type of training ("standard", "dialogue", "empathy", etc.)
            metrics_callback: Optional callback for custom metrics
            **kwargs: Additional arguments for the base trainer
        """
        # Store training type
        self.training_type = training_type
        self.metrics_callback = metrics_callback
        
        # Initialize base trainer
        super().__init__(
            model=model,
            train_dataloader=train_dataloader,
            training_config=training_config,
            model_config=model_config,
            data_config=data_config,
            eval_dataloader=eval_dataloader,
            **kwargs
        )
        
        # Set up training type-specific hooks
        self._setup_training_type_hooks()
    
    def _setup_training_type_hooks(self) -> None:
        """
        Set up hooks based on the training type.
        """
        # Add different hooks based on training type
        if self.training_type == "dialogue":
            self.add_hook("post_step", self._dialogue_post_step_hook)
            self.add_hook("pre_eval", self._dialogue_pre_eval_hook)
        elif self.training_type == "empathy":
            self.add_hook("post_step", self._empathy_post_step_hook)
        elif self.training_type == "function_call":
            self.add_hook("post_step", self._function_call_post_step_hook)
            self.add_hook("pre_eval", self._function_call_pre_eval_hook)
        elif self.training_type == "structured_output":
            self.add_hook("post_step", self._structured_output_post_step_hook)
        
        # Add metrics callback if provided
        if self.metrics_callback is not None:
            self.add_hook("post_step", self._metrics_callback_hook)
    
    def _forward(self, batch: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        """
        Forward pass through the model, specialized for the training type.
        
        Args:
            batch: Batch of data
            is_training: Whether this is a training forward pass
            
        Returns:
            Dictionary with model outputs, including loss
        """
        if self.training_type == "dialogue":
            return self._dialogue_forward(batch, is_training)
        elif self.training_type == "empathy":
            return self._empathy_forward(batch, is_training)
        elif self.training_type == "intent":
            return self._intent_forward(batch, is_training)
        elif self.training_type == "function_call":
            return self._function_call_forward(batch, is_training)
        elif self.training_type == "structured_output":
            return self._structured_output_forward(batch, is_training)
        else:
            # Standard forward pass
            return super()._forward(batch, is_training)
    
    def _dialogue_forward(self, batch: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        """
        Forward pass for dialogue training.
        
        Args:
            batch: Batch of data
            is_training: Whether this is a training forward pass
            
        Returns:
            Dictionary with model outputs
        """
        # Extract dialogue-specific inputs
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        response_ids = batch.get("response_ids")
        
        # Prepare labels for dialogue training
        if "labels" not in batch and response_ids is not None:
            # Create shifted labels from response for causal language modeling
            labels = response_ids.clone()
            # Mask input part with -100
            if "input_lengths" in batch:
                input_lengths = batch["input_lengths"]
                for i, length in enumerate(input_lengths):
                    labels[i, :length] = -100
            
            # Add labels to batch
            batch["labels"] = labels
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Add dialogue-specific metrics
        if "loss" in outputs and hasattr(outputs, "get"):
            # Calculate perplexity
            perplexity = torch.exp(outputs["loss"])
            outputs["perplexity"] = perplexity
        
        return outputs
    
    def _empathy_forward(self, batch: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        """
        Forward pass for empathy training.
        
        Args:
            batch: Batch of data
            is_training: Whether this is a training forward pass
            
        Returns:
            Dictionary with model outputs
        """
        # Add empathy-specific processing here
        # e.g., adding empathy labels or weights
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Add empathy-specific metrics
        if "empathy_scores" in batch and "loss" in outputs:
            # Calculate weighted empathy loss
            empathy_weights = batch.get("empathy_weights", torch.ones_like(batch["empathy_scores"]))
            empathy_loss = outputs["loss"] * empathy_weights.mean()
            outputs["empathy_loss"] = empathy_loss
        
        return outputs
    
    def _intent_forward(self, batch: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        """
        Forward pass for intent classification training.
        
        Args:
            batch: Batch of data
            is_training: Whether this is a training forward pass
            
        Returns:
            Dictionary with model outputs
        """
        # Add intent-specific processing
        if "intent_labels" in batch and "labels" not in batch:
            batch["labels"] = batch["intent_labels"]
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Add intent-specific metrics
        if "logits" in outputs and "labels" in batch:
            # Calculate accuracy
            predictions = torch.argmax(outputs["logits"], dim=-1)
            accuracy = (predictions == batch["labels"]).float().mean()
            outputs["accuracy"] = accuracy
        
        return outputs
    
    def _function_call_forward(self, batch: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        """
        Forward pass for function call training.
        
        Args:
            batch: Batch of data
            is_training: Whether this is a training forward pass
            
        Returns:
            Dictionary with model outputs
        """
        # Handle function call-specific processing
        if "function_name" in batch and "parameters" in batch:
            # Combine function call components if needed
            batch["function_call"] = {
                "name": batch["function_name"],
                "parameters": batch["parameters"]
            }
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Add function call-specific metrics
        if "function_name_accuracy" not in outputs and "logits" in outputs and "function_name" in batch:
            # Calculate function name accuracy
            name_predictions = outputs.get("function_name_predictions")
            if name_predictions is not None:
                name_accuracy = (name_predictions == batch["function_name"]).float().mean()
                outputs["function_name_accuracy"] = name_accuracy
        
        return outputs
    
    def _structured_output_forward(self, batch: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        """
        Forward pass for structured output training.
        
        Args:
            batch: Batch of data
            is_training: Whether this is a training forward pass
            
        Returns:
            Dictionary with model outputs
        """
        # Handle structured output-specific processing
        if "structure_labels" in batch and "labels" not in batch:
            batch["labels"] = batch["structure_labels"]
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Add structured output-specific metrics
        if "structure_accuracy" not in outputs and "predictions" in outputs and "structure_labels" in batch:
            # Calculate structure accuracy
            structure_accuracy = (outputs["predictions"] == batch["structure_labels"]).float().mean()
            outputs["structure_accuracy"] = structure_accuracy
        
        return outputs
    
    # Training type-specific hooks
    
    def _dialogue_post_step_hook(self, **kwargs) -> None:
        """Hook for dialogue-specific post-step processing."""
        metrics = kwargs.get("metrics", {})
        if "perplexity" in metrics:
            # Log perplexity
            logger.info(f"Dialogue perplexity: {metrics['perplexity']:.4f}")
    
    def _dialogue_pre_eval_hook(self, **kwargs) -> None:
        """Hook for dialogue-specific pre-evaluation processing."""
        # Setup for dialogue evaluation
        if hasattr(self.model, "prepare_for_dialogue_evaluation"):
            self.model.prepare_for_dialogue_evaluation()
    
    def _empathy_post_step_hook(self, **kwargs) -> None:
        """Hook for empathy-specific post-step processing."""
        metrics = kwargs.get("metrics", {})
        if "empathy_loss" in metrics:
            # Log empathy loss
            logger.info(f"Empathy loss: {metrics['empathy_loss']:.4f}")
    
    def _function_call_post_step_hook(self, **kwargs) -> None:
        """Hook for function call-specific post-step processing."""
        metrics = kwargs.get("metrics", {})
        if "function_name_accuracy" in metrics:
            # Log function name accuracy
            logger.info(f"Function name accuracy: {metrics['function_name_accuracy']:.4f}")
    
    def _function_call_pre_eval_hook(self, **kwargs) -> None:
        """Hook for function call-specific pre-evaluation processing."""
        # Setup for function call evaluation
        if hasattr(self.model, "prepare_for_function_call_evaluation"):
            self.model.prepare_for_function_call_evaluation()
    
    def _structured_output_post_step_hook(self, **kwargs) -> None:
        """Hook for structured output-specific post-step processing."""
        metrics = kwargs.get("metrics", {})
        if "structure_accuracy" in metrics:
            # Log structure accuracy
            logger.info(f"Structure accuracy: {metrics['structure_accuracy']:.4f}")
    
    def _metrics_callback_hook(self, **kwargs) -> None:
        """Hook for custom metrics calculation using the callback."""
        if self.metrics_callback is not None:
            # Call metrics callback with current state
            try:
                custom_metrics = self.metrics_callback(
                    model=self.model,
                    step=self.global_step,
                    epoch=self.epoch,
                    metrics=kwargs.get("metrics", {}),
                    batch=kwargs.get("batch", None)
                )
                
                # Log custom metrics
                if custom_metrics:
                    for key, value in custom_metrics.items():
                        logger.info(f"Custom metric {key}: {value:.4f}")
            except Exception as e:
                logger.error(f"Error in metrics callback: {e}")
    
    def update_training_type(self, training_type: str) -> None:
        """
        Update the training type during training.
        
        Args:
            training_type: New training type
        """
        # Update training type
        self.training_type = training_type
        
        # Clear and re-setup hooks
        self.pre_step_hooks = []
        self.post_step_hooks = []
        self.pre_eval_hooks = []
        self.post_eval_hooks = []
        
        # Setup new hooks
        self._setup_training_type_hooks()
        
        logger.info(f"Updated training type to: {training_type}")