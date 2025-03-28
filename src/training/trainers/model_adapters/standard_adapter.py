"""
Standard model adapter for language models in the enhanced training system.

This module provides an implementation of the ModelAdapter for standard 
language models, handling initialization, forward passes, and loss computation
specific to standard language model training.
"""

from typing import Dict, Any, Optional, Union, Tuple
import logging

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.model.semantic_resonance_model import SemanticResonanceModel
from src.training.model_adapters.base_adapter import ModelAdapter


logger = logging.getLogger("quantum_resonance")


class StandardModelAdapter(ModelAdapter):
    """
    Model adapter implementation for standard language models.
    
    This adapter handles model initialization, tokenizer setup, batch processing,
    and loss computation for standard language model training (e.g., causal language
    modeling, next token prediction).
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize standard model adapter.
        
        Args:
            model_config: Configuration for the model architecture
            training_config: Configuration for training
            device: Device to use for model execution
            logger: Logger instance
        """
        super().__init__(model_config, training_config, device)
        self.logger = logger or logging.getLogger("quantum_resonance")
    
    def create_model(self) -> nn.Module:
        """
        Create and initialize the standard language model.
        
        Returns:
            Initialized SemanticResonanceModel instance
        """
        self.logger.info("Initializing standard language model...")
        
        # Create model instance
        model = SemanticResonanceModel(self.model_config)
        model.to(self.device)
        
        # Log model size
        total_params, trainable_params = self.compute_model_size(model)
        self.logger.info(f"Model size: {total_params:,} parameters ({trainable_params:,} trainable)")
        
        return model
    
    def create_tokenizer(self) -> Any:
        """
        Create and initialize the tokenizer.
        
        Returns:
            Initialized tokenizer
        """
        # Get tokenizer name from model_config extras or use default
        tokenizer_name = "gpt2"  # Default tokenizer
        
        # Check if tokenizer_name is in model_config.extra_model_params
        if hasattr(self.model_config, 'extra_model_params') and 'tokenizer_name' in self.model_config.extra_model_params:
            tokenizer_name = self.model_config.extra_model_params['tokenizer_name']
        # For backward compatibility
        elif hasattr(self.model_config, 'tokenizer_name'):
            tokenizer_name = self.model_config.tokenizer_name
        # If training_config has data_config attribute
        elif hasattr(self.training_config, 'data_config') and hasattr(self.training_config.data_config, 'tokenizer_name'):
            tokenizer_name = self.training_config.data_config.tokenizer_name
            
        self.logger.info(f"Loading tokenizer: {tokenizer_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set default pad token if not set
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
            self.logger.info(f"Setting pad_token to eos_token: {tokenizer.pad_token}")
        
        # Ensure model embeddings match tokenizer
        if self.model is not None:
            self.model = self.resize_token_embeddings(self.model, tokenizer)
            self.logger.info(f"Resized token embeddings to match tokenizer size: {len(tokenizer)}")
        
        return tokenizer
    
    def prepare_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch for model processing.
        
        Args:
            batch: Batch from dataloader
            
        Returns:
            Batch prepared for model input
        """
        # Handle different batch formats
        if isinstance(batch, dict):
            prepared_batch = batch
        else:
            # Handle tuple batch (e.g., input_ids, attention_mask, labels)
            try:
                if len(batch) >= 3:
                    input_ids, attention_mask, labels = batch[:3]
                    prepared_batch = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    }
                elif len(batch) == 2:
                    input_ids, attention_mask = batch
                    prepared_batch = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": input_ids.clone()  # Use inputs as labels for language modeling
                    }
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            except Exception as e:
                raise ValueError(f"Error unpacking batch: {e}")
        
        # Ensure required keys are present
        required_keys = ["input_ids", "attention_mask"]
        for key in required_keys:
            if key not in prepared_batch:
                raise ValueError(f"Batch is missing required key: {key}")
        
        # If no labels are present, add dummy labels using input_ids
        if "labels" not in prepared_batch:
            self.logger.warning("No labels found in batch. Using input_ids as labels for language modeling.")
            prepared_batch["labels"] = prepared_batch["input_ids"].clone()
        
        # Move batch to device
        return self.move_to_device(prepared_batch)
    
    def forward(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        return_dict: bool = True
    ) -> Union[Dict[str, Any], Tuple]:
        """
        Perform forward pass with the model.
        
        Args:
            model: Model instance
            batch: Prepared input batch
            return_dict: Whether to return outputs as a dictionary
            
        Returns:
            Model outputs (as dict if return_dict=True, otherwise as tuple)
        """
        # Forward pass through model
        outputs = model(**batch, return_dict=return_dict)
        
        return outputs
    
    def compute_loss(
        self,
        outputs: Union[Dict[str, Any], Tuple],
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute loss from model outputs and batch.
        
        Args:
            outputs: Model outputs
            batch: Input batch that generated the outputs
            
        Returns:
            Loss tensor
        """
        # Extract loss from outputs
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            # Assume first element is loss in tuple outputs
            loss = outputs[0]
        else:
            raise ValueError("Unable to extract loss from model outputs")
        
        # Check for NaN/Inf values
        if torch.isnan(loss).item() or torch.isinf(loss).item():
            self.logger.warning("Detected NaN/Inf in loss calculation")
            # Could handle this differently if needed
        
        return loss