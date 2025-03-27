"""
Finetuning strategy for pretrained language models.

This module provides a specialized training strategy for finetuning
pretrained language models, with techniques like layer-wise learning rate
decay and parameter-efficient fine-tuning.
"""

from typing import Dict, Any, Optional, Union, Tuple, List, Callable
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast

from src.config.training_config import TrainingConfig
from src.training.strategies.standard_strategy import StandardTrainingStrategy


class FinetuningStrategy(StandardTrainingStrategy):
    """
    Strategy for finetuning pretrained language models.
    
    This strategy extends the standard training strategy with optimizations
    specific to finetuning, such as layer-wise learning rate decay and
    more careful gradient handling.
    """
    
    def __init__(
        self,
        training_config: TrainingConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the finetuning strategy.
        
        Args:
            training_config: Training configuration
            logger: Logger instance
        """
        super().__init__(training_config, logger)
        
        # Finetuning-specific settings
        self.layer_decay = getattr(training_config, "layer_decay", 0.9)
        self.max_grad_norm = getattr(training_config, "max_grad_norm", 0.5)  # Lower for finetuning
        self.weight_decay = getattr(training_config, "weight_decay", 0.01)
        
        # Finetuning typically uses lower learning rates
        if not hasattr(training_config, "finetune_learning_rate"):
            self.learning_rate = getattr(training_config, "learning_rate", 5e-5) / 10
            self.logger.info(f"Using reduced learning rate for finetuning: {self.learning_rate}")
        else:
            self.learning_rate = getattr(training_config, "finetune_learning_rate")
        
        # Early stopping for finetuning
        self.patience = getattr(training_config, "patience", 3)
        self.best_val_loss = float('inf')
        self.no_improvement_count = 0
    
    def create_optimizer(
        self,
        model: nn.Module,
        optimizer_type: str = "adamw",
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None
    ) -> optim.Optimizer:
        """
        Create an optimizer with layer-wise learning rate decay.
        
        Args:
            model: Model instance
            optimizer_type: Type of optimizer
            learning_rate: Learning rate (uses finetuning rate if None)
            weight_decay: Weight decay factor
            
        Returns:
            Initialized optimizer with layer-wise decay
        """
        # Use finetuning-specific values if not provided
        lr = learning_rate if learning_rate is not None else self.learning_rate
        wd = weight_decay if weight_decay is not None else self.weight_decay
        
        # Check if we should use layer-wise learning rate decay
        if self.layer_decay < 1.0:
            return self._create_layer_wise_optimizer(model, optimizer_type, lr, wd)
        else:
            # Use standard optimizer creation
            return super().create_optimizer(model, optimizer_type, lr, wd)
    
    def _create_layer_wise_optimizer(
        self,
        model: nn.Module,
        optimizer_type: str,
        learning_rate: float,
        weight_decay: float
    ) -> optim.Optimizer:
        """
        Create optimizer with layer-wise learning rate decay.
        
        Args:
            model: Model instance
            optimizer_type: Type of optimizer
            learning_rate: Base learning rate
            weight_decay: Weight decay factor
            
        Returns:
            Optimizer with layer-wise decay
        """
        self.logger.info(f"Creating layer-wise optimizer with decay factor: {self.layer_decay}")
        
        # Group parameters by layer depth and weight decay exclusion
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]
        layer_wise_groups = []
        
        # Get all named parameters
        named_parameters = list(model.named_parameters())
        
        # Identify the number of transformer layers
        num_layers = self._get_num_layers(model)
        
        # Process each parameter, assigning learning rate based on layer depth
        for name, param in named_parameters:
            # Skip non-trainable parameters
            if not param.requires_grad:
                continue
            
            # Determine layer index
            layer_idx = self._get_layer_index(name, num_layers)
            
            # Calculate learning rate with decay
            # Parameters in higher layers get higher learning rates
            layer_lr = learning_rate * (self.layer_decay ** (num_layers - 1 - layer_idx))
            
            # Determine weight decay
            decay_status = 0.0 if any(nd in name for nd in no_decay) else weight_decay
            
            # Add parameter to appropriate group
            layer_wise_groups.append({
                "params": [param],
                "lr": layer_lr,
                "weight_decay": decay_status,
                "layer_idx": layer_idx
            })
            
            # Log layer assignment for debugging
            if layer_idx >= 0:
                self.logger.debug(f"Parameter {name} assigned to layer {layer_idx} with lr {layer_lr:.6f}")
        
        # Create optimizer based on type
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == "adamw":
            return optim.AdamW(layer_wise_groups)
        elif optimizer_type == "adam":
            return optim.Adam(layer_wise_groups)
        else:
            self.logger.warning(f"Unsupported optimizer type for layer-wise decay: {optimizer_type}. Using AdamW.")
            return optim.AdamW(layer_wise_groups)
    
    def _get_num_layers(self, model: nn.Module) -> int:
        """
        Determine the number of transformer layers in the model.
        
        Args:
            model: Model instance
            
        Returns:
            Number of transformer layers
        """
        # Look for common patterns in model structure
        if hasattr(model, "num_layers"):
            return model.num_layers
        elif hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
            return model.config.num_hidden_layers
        
        # Count layers by name pattern
        layer_names = [name for name, _ in model.named_modules() 
                      if "layer" in name and name.count(".") == 1]
        
        # If layer names found, return count
        if layer_names:
            return max([int(name.split(".")[-1]) for name in layer_names]) + 1
        
        # Default to 12 layers (standard for many models)
        self.logger.warning("Could not determine number of layers, using default of 12")
        return 12
    
    def _get_layer_index(self, param_name: str, num_layers: int) -> int:
        """
        Determine the layer index of a parameter.
        
        Args:
            param_name: Name of the parameter
            num_layers: Total number of layers
            
        Returns:
            Layer index (0 to num_layers-1), or -1 for non-layer parameters
        """
        # Embeddings are typically in the lowest layer
        if "embed" in param_name:
            return 0
        
        # Check for common layer naming patterns
        # Format: model.layers.X.parameter
        if ".layer." in param_name or ".layers." in param_name:
            parts = param_name.split(".")
            for i, part in enumerate(parts):
                if part == "layer" or part == "layers":
                    if i + 1 < len(parts) and parts[i + 1].isdigit():
                        return int(parts[i + 1])
        
        # Format: model.transformer.layerX.parameter
        for i in range(num_layers):
            if f".layer{i}." in param_name or f".layer.{i}." in param_name:
                return i
        
        # Format: model.encoder.layer.X.parameter
        if ".encoder.layer." in param_name:
            parts = param_name.split("encoder.layer.")[1].split(".")
            if parts[0].isdigit():
                return int(parts[0])
        
        # Head parameters typically get the highest learning rate
        if "head" in param_name or "output" in param_name or "classifier" in param_name:
            return num_layers - 1
        
        # Unknown parameter - use middle layer as default
        return num_layers // 2
    
    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        scheduler: Optional[Any] = None,
        update_gradients: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a single finetuning step.
        
        Args:
            model: Model instance
            batch: Input batch
            optimizer: Optimizer instance
            scaler: Gradient scaler for mixed precision
            scheduler: Learning rate scheduler
            update_gradients: Whether to update gradients or just accumulate them
            
        Returns:
            Dictionary of step metrics (loss, etc.)
        """
        # Use standard training step from parent class
        metrics = super().train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            update_gradients=update_gradients
        )
        
        # Add finetuning-specific metrics
        if update_gradients and "valid_gradients" in metrics and metrics["valid_gradients"]:
            metrics["layer_lrs"] = self._get_layer_learning_rates(optimizer)
        
        return metrics
    
    def _get_layer_learning_rates(self, optimizer: optim.Optimizer) -> Dict[int, float]:
        """
        Get learning rates for each layer from the optimizer.
        
        Args:
            optimizer: Optimizer instance with layer-wise groups
            
        Returns:
            Dictionary mapping layer indices to learning rates
        """
        layer_lrs = {}
        
        for group in optimizer.param_groups:
            if "layer_idx" in group:
                layer_idx = group["layer_idx"]
                if layer_idx not in layer_lrs:
                    layer_lrs[layer_idx] = group["lr"]
        
        return layer_lrs
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            val_loss: Validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_val_loss:
            relative_improvement = (self.best_val_loss - val_loss) / self.best_val_loss
            self.best_val_loss = val_loss
            self.no_improvement_count = 0
            
            self.logger.info(f"Validation loss improved to {val_loss:.6f} ({relative_improvement:.2%} improvement)")
            return False
        else:
            self.no_improvement_count += 1
            self.logger.info(f"No improvement in validation loss for {self.no_improvement_count} evaluations")
            
            if self.no_improvement_count >= self.patience:
                self.logger.info(f"Early stopping triggered after {self.no_improvement_count} evaluations without improvement")
                return True
            
            return False