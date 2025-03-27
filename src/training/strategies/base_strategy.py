"""
Base training strategy for the enhanced training system.

This module defines the abstract base class that all training strategies
must implement. Training strategies handle the algorithm for training models,
including training steps, validation, and optimization procedures.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple, List, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

from src.config.training_config import TrainingConfig


class TrainingStrategy(ABC):
    """
    Abstract base class for training strategies.
    
    Training strategies encapsulate the training algorithm and process,
    including specific training steps, validation procedures, and
    training-specific optimizations.
    """
    
    def __init__(
        self,
        training_config: TrainingConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the training strategy.
        
        Args:
            training_config: Training configuration 
            logger: Logger instance
        """
        self.config = training_config
        self.logger = logger or logging.getLogger("quantum_resonance")
        
        # Common training settings
        self.max_grad_norm = getattr(training_config, "max_grad_norm", 1.0)
        self.accumulation_steps = getattr(training_config, "accumulation_steps", 1)
        self.use_mixed_precision = getattr(training_config, "use_mixed_precision", True)
        
        # Metrics and state
        self.current_epoch = 0
        self.global_step = 0
    
    @abstractmethod
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
        Execute a single training step.
        
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
        pass
    
    @abstractmethod
    def validation_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Execute a single validation step.
        
        Args:
            model: Model instance
            batch: Input batch
            
        Returns:
            Dictionary of validation metrics
        """
        pass
    
    @abstractmethod
    def compute_batch_metrics(
        self,
        outputs: Union[Dict[str, Any], Tuple],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Compute metrics for a batch based on model outputs.
        
        Args:
            outputs: Model outputs
            batch: Input batch
            
        Returns:
            Dictionary of metrics
        """
        pass
    
    def validate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """
        Validate the model on a dataset.
        
        Args:
            model: Model instance
            dataloader: Validation dataloader
            device: Device to run validation on
            
        Returns:
            Dictionary of validation metrics
        """
        model.eval()
        
        all_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device if needed
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                
                # Perform validation step
                metrics = self.validation_step(model, batch)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = 0.0
                    all_metrics[key] += value
                
                num_batches += 1
        
        # Compute average metrics
        for key in all_metrics:
            all_metrics[key] /= max(num_batches, 1)
        
        # Add perplexity if loss is present
        if "loss" in all_metrics:
            try:
                all_metrics["perplexity"] = torch.exp(torch.tensor(all_metrics["loss"])).item()
            except OverflowError:
                all_metrics["perplexity"] = float('inf')
        
        return all_metrics
    
    def clip_gradients(
        self,
        model: nn.Module,
        max_grad_norm: Optional[float] = None
    ) -> float:
        """
        Clip gradients to prevent exploding gradients.
        
        Args:
            model: Model instance
            max_grad_norm: Maximum gradient norm (uses self.max_grad_norm if None)
            
        Returns:
            Total gradient norm before clipping
        """
        # Use instance value if not provided
        max_norm = max_grad_norm if max_grad_norm is not None else self.max_grad_norm
        
        # Skip if max_norm is 0 or negative
        if max_norm <= 0:
            return 0.0
        
        # Clip gradients
        return torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=max_norm
        ).item()
    
    def apply_weight_decay(
        self,
        model: nn.Module,
        weight_decay: float,
        learning_rate: float
    ) -> None:
        """
        Apply weight decay to parameters manually (AdamW-style).
        
        Args:
            model: Model instance
            weight_decay: Weight decay factor
            learning_rate: Current learning rate
        """
        # Skip if weight decay is 0
        if weight_decay <= 0:
            return
        
        # Apply weight decay manually (AdamW-style)
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    param.data = param.data.add(
                        param.data, alpha=-weight_decay * learning_rate
                    )
    
    def create_optimizer(
        self,
        model: nn.Module,
        optimizer_type: str = "adamw",
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None
    ) -> optim.Optimizer:
        """
        Create an optimizer instance.
        
        Args:
            model: Model instance
            optimizer_type: Type of optimizer ("adamw", "adam", "sgd", etc.)
            learning_rate: Learning rate (uses config value if None)
            weight_decay: Weight decay factor (uses config value if None)
            
        Returns:
            Initialized optimizer
            
        Raises:
            ValueError: If optimizer type is not supported
        """
        # Use config values if not provided
        lr = learning_rate if learning_rate is not None else self.config.learning_rate
        wd = weight_decay if weight_decay is not None else getattr(self.config, "weight_decay", 0.01)
        
        # Prepare parameter groups with weight decay exclusion
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]
        param_groups = [
            {
                "params": [p for n, p in model.named_parameters() 
                         if not any(nd in n for nd in no_decay)],
                "weight_decay": wd,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                         if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Create optimizer based on type
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == "adamw":
            return optim.AdamW(param_groups, lr=lr)
        elif optimizer_type == "adam":
            return optim.Adam(param_groups, lr=lr)
        elif optimizer_type == "sgd":
            return optim.SGD(param_groups, lr=lr, momentum=0.9)
        elif optimizer_type == "adafactor":
            try:
                from transformers.optimization import Adafactor
                return Adafactor(param_groups, lr=lr, scale_parameter=False, relative_step=False)
            except ImportError:
                self.logger.warning("Adafactor optimizer not available, falling back to AdamW")
                return optim.AdamW(param_groups, lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def create_scheduler(
        self,
        optimizer: optim.Optimizer,
        scheduler_type: str = "linear",
        num_training_steps: int = 1000,
        num_warmup_steps: Optional[int] = None
    ) -> Optional[Any]:
        """
        Create a learning rate scheduler.
        
        Args:
            optimizer: Optimizer instance
            scheduler_type: Type of scheduler ("linear", "cosine", etc.)
            num_training_steps: Total number of training steps
            num_warmup_steps: Number of warmup steps (uses config or 10% if None)
            
        Returns:
            Initialized scheduler or None if not supported
        """
        # Use config value or 10% of training steps if not provided
        warmup_steps = num_warmup_steps
        if warmup_steps is None:
            warmup_steps = getattr(self.config, "warmup_steps", int(0.1 * num_training_steps))
        
        try:
            from transformers.optimization import (
                get_linear_schedule_with_warmup,
                get_cosine_schedule_with_warmup,
                get_constant_schedule_with_warmup
            )
            
            # Create scheduler based on type
            scheduler_type = scheduler_type.lower()
            
            if scheduler_type == "linear":
                return get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
                )
            elif scheduler_type == "cosine":
                return get_cosine_schedule_with_warmup(
                    optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
                )
            elif scheduler_type == "constant":
                return get_constant_schedule_with_warmup(
                    optimizer, num_warmup_steps=warmup_steps
                )
            elif scheduler_type == "none" or scheduler_type == "":
                return None
            else:
                self.logger.warning(f"Unsupported scheduler type: {scheduler_type}, using linear")
                return get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
                )
        except ImportError:
            self.logger.warning("transformers.optimization not available, falling back to basic schedulers")
            
            # Fallback to basic schedulers
            if scheduler_type == "cosine":
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_training_steps - warmup_steps
                )
            elif scheduler_type == "linear":
                return torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_training_steps - warmup_steps
                )
            elif scheduler_type == "constant":
                return None
            else:
                return None