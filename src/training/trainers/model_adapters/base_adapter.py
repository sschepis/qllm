"""
Base interface for model adapters in the enhanced training system.

This module defines the abstract base class that all model adapters must implement.
Model adapters serve as a bridge between the training system and specific model types,
handling model-specific operations such as initialization, forward passes, and loss computation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    Model adapters bridge the training system with specific model architectures,
    handling model-specific initialization, forward passes, loss computation, etc.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the model adapter.
        
        Args:
            model_config: Configuration for the model architecture
            training_config: Configuration for training
            device: Device to use for model execution
        """
        self.model_config = model_config
        self.training_config = training_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def create_model(self) -> nn.Module:
        """
        Create and initialize the model.
        
        Returns:
            Initialized model instance
        """
        pass
    
    @abstractmethod
    def create_tokenizer(self) -> Any:
        """
        Create and initialize the tokenizer.
        
        Returns:
            Initialized tokenizer
        """
        pass
    
    @abstractmethod
    def prepare_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch for model processing.
        
        Args:
            batch: Batch from dataloader
            
        Returns:
            Batch prepared for model input
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    def move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Move batch tensors to the appropriate device.
        
        Args:
            batch: Input batch
            
        Returns:
            Batch with tensors moved to device
        """
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
    
    def get_model(self) -> nn.Module:
        """
        Get the current model instance.
        
        Returns:
            Current model instance
        """
        if self.model is None:
            self.model = self.create_model()
            self.model.to(self.device)
        return self.model
        
    def set_model(self, model: nn.Module) -> None:
        """
        Set the model instance.
        
        Args:
            model: Model instance to use
        """
        self.model = model
        if self.model is not None and self.device is not None:
            self.model.to(self.device)
    
    def get_tokenizer(self) -> Any:
        """
        Get the current tokenizer instance.
        
        Returns:
            Current tokenizer instance
        """
        if self.tokenizer is None:
            self.tokenizer = self.create_tokenizer()
        return self.tokenizer
    
    def set_tokenizer(self, tokenizer: Any) -> None:
        """
        Set the tokenizer instance.
        
        Args:
            tokenizer: Tokenizer instance to use
        """
        self.tokenizer = tokenizer
    
    def resize_token_embeddings(self, model: nn.Module, tokenizer: Any) -> nn.Module:
        """
        Resize model token embeddings to match tokenizer vocabulary.
        
        Args:
            model: Model instance
            tokenizer: Tokenizer instance
            
        Returns:
            Model with resized embeddings
        """
        if hasattr(model, "resize_token_embeddings"):
            return model.resize_token_embeddings(len(tokenizer))
        return model
    
    def compute_model_size(self, model: nn.Module) -> Tuple[int, int]:
        """
        Compute model size statistics.
        
        Args:
            model: Model instance
            
        Returns:
            Tuple of (total_parameters, trainable_parameters)
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params