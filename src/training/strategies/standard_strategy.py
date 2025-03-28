"""
Standard training strategy for the enhanced training system.

This module implements a standard training strategy for language models,
providing general-purpose training and evaluation functionality.
"""

import logging
import math
from typing import Dict, Any, Optional, Union, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.training.strategies.base_strategy import TrainingStrategy


class StandardTrainingStrategy(TrainingStrategy):
    """
    Standard training strategy for language models.
    
    This strategy provides general-purpose training for language models,
    including typical optimizers, schedulers, and validation procedures.
    """
    
    def __init__(
        self,
        config: Any,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the standard training strategy.
        
        Args:
            config: Training configuration
            logger: Logger instance
        """
        super().__init__(config, logger)
        
        # Extract parameters
        self.learning_rate = getattr(config, "learning_rate", 5e-5)
        self.weight_decay = getattr(config, "weight_decay", 0.01)
        self.max_grad_norm = getattr(config, "max_grad_norm", 1.0)
        self.use_mixed_precision = getattr(config, "use_mixed_precision", True)
        self.accumulation_steps = getattr(config, "accumulation_steps", 1)
    
    def create_optimizer(
        self,
        model: nn.Module,
        optimizer_type: str = "adamw",
        **kwargs
    ) -> optim.Optimizer:
        """
        Create an optimizer for the model.
        
        Args:
            model: Model to optimize
            optimizer_type: Type of optimizer
            **kwargs: Additional optimizer parameters
            
        Returns:
            Initialized optimizer
        """
        # Get parameters from kwargs or default to config
        lr = kwargs.get("lr", self.learning_rate)
        weight_decay = kwargs.get("weight_decay", self.weight_decay)
        
        # Get parameters that require gradients
        parameters = [p for p in model.parameters() if p.requires_grad]
        
        # Create optimizer based on type
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == "adamw":
            self.logger.info(f"Creating AdamW optimizer with lr={lr}, weight_decay={weight_decay}")
            return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        
        elif optimizer_type == "adam":
            self.logger.info(f"Creating Adam optimizer with lr={lr}, weight_decay={weight_decay}")
            return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        
        elif optimizer_type == "sgd":
            momentum = kwargs.get("momentum", 0.9)
            self.logger.info(f"Creating SGD optimizer with lr={lr}, momentum={momentum}, weight_decay={weight_decay}")
            return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        elif optimizer_type == "adafactor":
            # Use Adafactor from transformers if available
            try:
                from transformers.optimization import Adafactor
                scale_parameter = kwargs.get("scale_parameter", True)
                relative_step = kwargs.get("relative_step", False)
                warmup_init = kwargs.get("warmup_init", False)
                
                self.logger.info(f"Creating Adafactor optimizer with lr={lr}")
                return Adafactor(
                    parameters,
                    lr=lr if not relative_step else None,
                    scale_parameter=scale_parameter,
                    relative_step=relative_step,
                    warmup_init=warmup_init
                )
            except ImportError:
                self.logger.warning("Adafactor not available, falling back to AdamW")
                return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        
        else:
            self.logger.warning(f"Unknown optimizer type: {optimizer_type}, using AdamW")
            return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    
    def create_scheduler(
        self,
        optimizer: optim.Optimizer,
        scheduler_type: str = "linear",
        num_training_steps: int = 1000,
        num_warmup_steps: int = 0,
        **kwargs
    ) -> Optional[Any]:
        """
        Create a learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            scheduler_type: Type of scheduler
            num_training_steps: Total number of training steps
            num_warmup_steps: Number of warmup steps
            **kwargs: Additional scheduler parameters
            
        Returns:
            Initialized scheduler or None
        """
        scheduler_type = scheduler_type.lower()
        
        # If no scheduler requested
        if scheduler_type == "none" or scheduler_type == "constant":
            return None
        
        # Try to use transformers schedulers if available
        try:
            from transformers.optimization import (
                get_linear_schedule_with_warmup,
                get_cosine_schedule_with_warmup,
                get_cosine_with_hard_restarts_schedule_with_warmup,
                get_polynomial_decay_schedule_with_warmup,
                get_constant_schedule,
                get_constant_schedule_with_warmup
            )
            
            if scheduler_type == "linear":
                self.logger.info(f"Creating linear scheduler with {num_warmup_steps} warmup steps")
                return get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
            
            elif scheduler_type == "cosine":
                self.logger.info(f"Creating cosine scheduler with {num_warmup_steps} warmup steps")
                return get_cosine_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
            
            elif scheduler_type == "cosine_restarts":
                num_cycles = kwargs.get("num_cycles", 1)
                self.logger.info(f"Creating cosine scheduler with {num_warmup_steps} warmup steps and {num_cycles} restarts")
                return get_cosine_with_hard_restarts_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                    num_cycles=num_cycles
                )
            
            elif scheduler_type == "polynomial":
                power = kwargs.get("power", 1.0)
                self.logger.info(f"Creating polynomial scheduler with {num_warmup_steps} warmup steps and power={power}")
                return get_polynomial_decay_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                    power=power
                )
            
            elif scheduler_type == "constant_warmup":
                self.logger.info(f"Creating constant scheduler with {num_warmup_steps} warmup steps")
                return get_constant_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps
                )
                
        except ImportError:
            self.logger.warning("Transformers schedulers not available, using PyTorch schedulers")
        
        # Fall back to PyTorch schedulers
        if scheduler_type == "linear":
            # Simple linear scheduler with warmup using LambdaLR
            def lr_lambda(current_step: int):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(
                    0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
                )
            
            self.logger.info(f"Creating linear scheduler with {num_warmup_steps} warmup steps")
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        elif scheduler_type == "cosine":
            # Cosine scheduler with warmup using CosineAnnealingLR with warmup
            if num_warmup_steps > 0:
                self.logger.warning("PyTorch cosine scheduler does not support warmup, ignoring warmup steps")
                
            self.logger.info("Creating cosine annealing scheduler")
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=num_training_steps, 
                eta_min=kwargs.get("eta_min", 0)
            )
        
        elif scheduler_type == "step":
            step_size = kwargs.get("step_size", num_training_steps // 3)
            gamma = kwargs.get("gamma", 0.1)
            
            self.logger.info(f"Creating step scheduler with step_size={step_size}, gamma={gamma}")
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma
            )
        
        elif scheduler_type == "exponential":
            gamma = kwargs.get("gamma", 0.9)
            
            self.logger.info(f"Creating exponential scheduler with gamma={gamma}")
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=gamma
            )
        
        self.logger.warning(f"Unknown scheduler type: {scheduler_type}, not using scheduler")
        return None
        
    def training_step(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        model_adapter: Any = None,
        global_step: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Adapter method for training_step that matches the interface expected by TrainerCore.
        This delegates to the train_step method with the appropriate parameters.
        
        Args:
            model: Model to train
            batch: Batch of data
            model_adapter: Model adapter (optional)
            global_step: Global step counter
            
        Returns:
            Tuple of (loss, metrics dictionary)
        """
        # Find or create optimizer
        optimizer = None
        if model_adapter:
            optimizer = getattr(model_adapter, 'optimizer', None)
            if optimizer is None and hasattr(model_adapter, 'get_optimizer'):
                optimizer = model_adapter.get_optimizer()
                
        if optimizer is None:
            # Create dummy optimizer as fallback
            self.logger.warning("No optimizer found, using dummy optimizer that won't update weights")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0)  # zero learning rate
        
        # Find or create gradient scaler for mixed precision
        scaler = None
        if model_adapter:
            scaler = getattr(model_adapter, 'grad_scaler', None)
            if scaler is None and hasattr(model_adapter, 'get_grad_scaler'):
                scaler = model_adapter.get_grad_scaler()
                
        # Find or create scheduler
        scheduler = None
        if model_adapter:
            scheduler = getattr(model_adapter, 'lr_scheduler', None)
            if scheduler is None and hasattr(model_adapter, 'get_lr_scheduler'):
                scheduler = model_adapter.get_lr_scheduler()
        
        # Prepare batch with model_adapter if available
        prepared_batch = batch
        if model_adapter and hasattr(model_adapter, 'prepare_batch'):
            prepared_batch = model_adapter.prepare_batch(batch)
        
        # Get or compute loss
        loss = None
        
        # First try to use model_adapter for forward pass if available
        if model_adapter and hasattr(model_adapter, 'forward'):
            try:
                loss, outputs = model_adapter.forward(model, prepared_batch)
            except Exception as e:
                self.logger.warning(f"Error in model_adapter.forward: {e}, falling back to direct model call")
        
        # If we don't have loss yet, try direct model forward pass
        if loss is None:
            try:
                outputs = model(**prepared_batch)
                if isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                elif hasattr(outputs, 'loss'):
                    loss = outputs.loss
            except Exception as e:
                self.logger.error(f"Error in model forward pass: {e}")
                # Create a dummy loss as fallback
                loss = torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
        
        # Call the underlying train_step function with update_gradients=False first
        # so we can capture the metrics without changing the model
        with torch.no_grad():
            try:
                metrics = self.train_step(
                    model=model,
                    batch=prepared_batch,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    update_gradients=False  # Don't update weights, just compute metrics
                )
            except Exception as e:
                self.logger.error(f"Error computing metrics: {e}")
                # Handle any type of loss variable, including strings from error messages
                if isinstance(loss, torch.Tensor) and hasattr(loss, 'item'):
                    metrics = {"loss": loss.item()}
                elif isinstance(loss, (int, float)):
                    metrics = {"loss": float(loss)}
                else:
                    # Convert anything else to string and use a default value
                    self.logger.error(f"Unexpected loss type: {type(loss)}, value: {str(loss)}")
                    metrics = {"loss": 0.0}
        
        # Now do the actual gradient step with update_gradients=True
        try:
            self.train_step(
                model=model,
                batch=prepared_batch,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                update_gradients=True  # Actually update the weights
            )
        except Exception as e:
            self.logger.error(f"Error updating model weights: {e}")
        
        return loss, metrics
    
    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        optimizer: optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        scheduler: Optional[Any] = None,
        update_gradients: bool = True
    ) -> Dict[str, Any]:
        """
        Perform a single training step.
        
        Args:
            model: Model to train
            batch: Batch of data
            optimizer: Optimizer to use
            scaler: Gradient scaler for mixed-precision training
            scheduler: Learning rate scheduler
            update_gradients: Whether to update gradients (for gradient accumulation)
            
        Returns:
            Dictionary of step statistics
        """
        # Set model to training mode
        model.train()
        
        # Track metrics
        step_metrics = {}
        
        # Check if mixed precision should be used
        use_mixed_precision = scaler is not None
        
        # Extract batch inputs through adapter if available
        if hasattr(model, "adapter") and hasattr(model.adapter, "prepare_batch"):
            batch = model.adapter.prepare_batch(batch)
        
        # Forward pass with autocast for mixed precision
        if use_mixed_precision:
            with torch.cuda.amp.autocast():
                if hasattr(model, "adapter") and hasattr(model.adapter, "forward"):
                    # Use adapter's forward method
                    loss, outputs = model.adapter.forward(model, batch)
                else:
                    # Use standard forward with input_ids and attention_mask
                    outputs = model(**batch)
                    loss = outputs.get("loss")
                    
                # Save loss for reporting
                step_metrics["loss"] = loss.item()
        else:
            # Standard precision forward
            if hasattr(model, "adapter") and hasattr(model.adapter, "forward"):
                # Use adapter's forward method
                loss, outputs = model.adapter.forward(model, batch)
            else:
                # Use standard forward with input_ids and attention_mask
                outputs = model(**batch)
                loss = outputs.get("loss")
                
            # Save loss for reporting
            step_metrics["loss"] = loss.item()
        
        # Scale the loss for gradient accumulation if needed
        if self.accumulation_steps > 1:
            loss = loss / self.accumulation_steps
        
        # Backward pass with scaler for mixed precision
        if use_mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update parameters if this is an update step
        if update_gradients:
            # Compute gradient norm for logging
            grad_norm = self._compute_grad_norm(model)
            step_metrics["grad_norm"] = grad_norm
            
            # Clip gradients
            if self.max_grad_norm > 0:
                if use_mixed_precision:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            
            # Update parameters with scaler for mixed precision
            if use_mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()
                step_metrics["learning_rate"] = self.get_learning_rate(optimizer)
        
        # Return metrics
        return step_metrics
    
    def validate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """
        Validate the model on a dataset.
        
        Args:
            model: Model to validate
            dataloader: Validation dataloader
            device: Device to use
            
        Returns:
            Dictionary of validation metrics
        """
        # Set model to evaluation mode
        model.eval()
        
        # Track metrics
        total_loss = 0
        total_elements = 0
        all_metrics = {}
        
        # Check if model has adapter for computing metrics
        compute_metrics = None
        if hasattr(model, "adapter") and hasattr(model.adapter, "compute_metrics"):
            compute_metrics = model.adapter.compute_metrics
        
        # Validation loop
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                
                # Extract batch inputs through adapter if available
                if hasattr(model, "adapter") and hasattr(model.adapter, "prepare_batch"):
                    batch = model.adapter.prepare_batch(batch)
                
                # Forward pass
                if hasattr(model, "adapter") and hasattr(model.adapter, "forward"):
                    # Use adapter's forward method
                    loss, outputs = model.adapter.forward(model, batch, is_training=False)
                else:
                    # Use standard forward
                    outputs = model(**batch)
                    loss = outputs.get("loss")
                
                # Get batch size
                batch_size = batch.get("input_ids").size(0) if "input_ids" in batch else 1
                
                # Track loss
                total_loss += loss.item() * batch_size
                total_elements += batch_size
                
                # Compute additional metrics if possible
                if compute_metrics:
                    batch_metrics = compute_metrics(outputs, batch)
                    
                    # Aggregate metrics
                    for key, value in batch_metrics.items():
                        if key not in all_metrics:
                            all_metrics[key] = 0
                        all_metrics[key] += value * batch_size
        
        # Compute average loss
        avg_loss = total_loss / total_elements if total_elements > 0 else float("inf")
        
        # Compute average metrics
        metrics = {"loss": avg_loss}
        for key, value in all_metrics.items():
            metrics[key] = value / total_elements if total_elements > 0 else 0
        
        # Compute perplexity
        metrics["perplexity"] = math.exp(avg_loss)
        
        return metrics
    
    def _compute_grad_norm(self, model: nn.Module) -> float:
        """
        Compute gradient norm for the model.
        
        Args:
            model: Model to compute gradient norm for
            
        Returns:
            Gradient norm
        """
        total_norm = 0.0
        parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        
        if len(parameters) == 0:
            return 0.0
            
        for p in parameters:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            
        total_norm = total_norm ** 0.5
        return total_norm