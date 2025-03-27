"""
Standard training strategy for language models.

This module provides an implementation of a standard training strategy
for language model training, supporting features like mixed precision,
gradient accumulation, and proper optimization.
"""

from typing import Dict, Any, Optional, Union, Tuple, List, Callable
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast

from src.config.training_config import TrainingConfig
from src.training.strategies.base_strategy import TrainingStrategy


class StandardTrainingStrategy(TrainingStrategy):
    """
    Standard training strategy for language models.
    
    This strategy implements common training procedures for standard language
    models, including proper forward/backward passes, mixed precision training,
    gradient accumulation, and optimization.
    """
    
    def __init__(
        self,
        training_config: TrainingConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the standard training strategy.
        
        Args:
            training_config: Training configuration
            logger: Logger instance
        """
        super().__init__(training_config, logger)
        
        # Strategy-specific settings
        self.logging_steps = getattr(training_config, "logging_steps", 10)
        self.eval_steps = getattr(training_config, "eval_steps", 0)
        self.save_steps = getattr(training_config, "save_steps", 0)
        
        # NaN prevention - specific to standard language model training
        self.detect_anomaly = getattr(training_config, "detect_anomaly", False)
    
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
        model.train()
        
        # Get device from model
        device = next(model.parameters()).device
        
        # Enable anomaly detection during training if requested
        torch.autograd.set_detect_anomaly(self.detect_anomaly)
        
        # Forward pass with mixed precision
        with autocast(device_type=device.type, enabled=self.use_mixed_precision):
            outputs = model(**batch)
            
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            # Handle gradient accumulation
            if self.accumulation_steps > 1:
                loss = loss / self.accumulation_steps
        
        # Backward pass
        if scaler is not None and self.use_mixed_precision and device.type == 'cuda':
            # Use scaler for mixed precision
            scaler.scale(loss).backward()
        else:
            # Standard backward
            loss.backward()
        
        # Extract unscaled loss for reporting
        step_loss = loss.item() * (self.accumulation_steps if self.accumulation_steps > 1 else 1)
        
        # Update weights if accumulation complete
        if update_gradients:
            grad_norm = 0.0
            
            # Unscale gradients if using mixed precision
            if scaler is not None and self.use_mixed_precision and device.type == 'cuda':
                scaler.unscale_(optimizer)
            
            # Clip gradients to prevent exploding gradients
            grad_norm = self.clip_gradients(model, self.max_grad_norm)
            
            # Check for invalid gradients
            valid_gradients = True
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        valid_gradients = False
                        self.logger.warning(f"NaN/Inf gradients detected at step {self.global_step}. Skipping update.")
                        break
            
            if valid_gradients:
                # Update weights
                if scaler is not None and self.use_mixed_precision and device.type == 'cuda':
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # Update learning rate scheduler
                if scheduler is not None:
                    scheduler.step()
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Update step count
                self.global_step += 1
            
            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            
            # Return step metrics
            return {
                "loss": step_loss,
                "grad_norm": grad_norm,
                "learning_rate": current_lr,
                "valid_gradients": valid_gradients
            }
        
        # Return step metrics without gradient updates
        return {
            "loss": step_loss
        }
    
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
        model.eval()
        
        # Get device from model
        device = next(model.parameters()).device
        
        # Forward pass with mixed precision
        with torch.no_grad():
            with autocast(device_type=device.type, enabled=self.use_mixed_precision):
                outputs = model(**batch)
                
                # Compute metrics based on outputs
                metrics = self.compute_batch_metrics(outputs, batch)
        
        return metrics
    
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
        metrics = {}
        
        # Extract loss
        if isinstance(outputs, dict) and "loss" in outputs:
            metrics["loss"] = outputs["loss"].item()
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            metrics["loss"] = outputs[0].item()
        else:
            # No explicit loss, return empty metrics
            return metrics
        
        # Skip NaN losses
        if torch.isnan(torch.tensor(metrics["loss"])).item() or torch.isinf(torch.tensor(metrics["loss"])).item():
            self.logger.warning("Skipping NaN/Inf loss in metrics calculation")
            metrics["loss"] = 0.0
            return metrics
        
        # Calculate additional metrics
        metrics["perplexity"] = torch.exp(torch.tensor(metrics["loss"])).item()
        
        # Additional metrics like accuracy if labels and logits are available
        if isinstance(outputs, dict) and "logits" in outputs and "labels" in batch:
            logits = outputs["logits"]
            labels = batch["labels"]
            
            # Calculate token-level accuracy if shapes match
            if logits.shape[:-1] == labels.shape:
                # Exclude padding tokens (-100) from accuracy calculation
                mask = (labels != -100).float()
                predictions = logits.argmax(dim=-1)
                correct = ((predictions == labels) * mask).sum().item()
                total = mask.sum().item()
                
                if total > 0:
                    metrics["accuracy"] = correct / total
        
        return metrics