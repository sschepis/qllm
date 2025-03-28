"""
Base Trainer for QLLM.

This module provides a consolidated base trainer implementation that handles
common training functionality, reducing code duplication across specialized
trainers. It implements core training loops, optimization, checkpointing,
and logging functionality.
"""

import os
import time
import math
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Type, Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.core.checkpoint import CheckpointManager, save_checkpoint, load_checkpoint
from src.config.training_config import TrainingConfig
from src.config.model_config import ModelConfig
from src.config.data_config import DataConfig


logger = logging.getLogger("qllm.training")


class BaseTrainer:
    """
    Base trainer class that centralizes common training functionality.
    
    This class provides a robust implementation of common training patterns,
    reducing code duplication across specialized trainers. It handles:
    - Training and evaluation loops
    - Checkpoint management
    - Optimization setup
    - Mixed precision training
    - Logging and metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        training_config: TrainingConfig,
        model_config: Optional[ModelConfig] = None,
        data_config: Optional[DataConfig] = None,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        """
        Initialize the base trainer.
        
        Args:
            model: Model to train
            train_dataloader: DataLoader for training data
            training_config: Training configuration
            model_config: Optional model configuration
            data_config: Optional data configuration
            eval_dataloader: Optional DataLoader for evaluation data
            optimizer: Optional custom optimizer
            lr_scheduler: Optional learning rate scheduler
            checkpoint_manager: Optional checkpoint manager
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Store configurations
        self.training_config = training_config
        self.model_config = model_config
        self.data_config = data_config
        
        # Set device based on configuration
        self.device = self._setup_device()
        
        # Move model to device
        self.model.to(self.device)
        
        # Set up optimizer and scheduler
        self.optimizer = optimizer or self._setup_optimizer()
        self.lr_scheduler = lr_scheduler or self._setup_scheduler()
        
        # Set up checkpoint manager
        self.checkpoint_manager = checkpoint_manager or self._setup_checkpoint_manager()
        
        # Initialize training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.metrics_history = {
            "train": [],
            "eval": []
        }
        
        # Set up mixed precision training
        self.scaler = self._setup_amp()
        
        # Initialize hooks
        self.pre_training_hooks = []
        self.post_training_hooks = []
        self.pre_epoch_hooks = []
        self.post_epoch_hooks = []
        self.pre_step_hooks = []
        self.post_step_hooks = []
        self.pre_eval_hooks = []
        self.post_eval_hooks = []
        
        # Initialize counters and timers
        self.training_start_time = 0
        self.last_log_time = 0
        self.accumulated_loss = 0
        self.accumulated_metrics = {}
    
    def _setup_device(self) -> torch.device:
        """
        Set up the device for training.
        
        Returns:
            Device to use for training
        """
        if self.training_config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.training_config.device)
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """
        Set up the optimizer based on the training configuration.
        
        Returns:
            Configured optimizer
        """
        # Get optimizer parameters from the model, including parameter groups
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.training_config.weight_decay
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]
        
        # Create the optimizer based on the specified type
        if self.training_config.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.training_config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.training_config.optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.training_config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.training_config.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.training_config.learning_rate,
                momentum=0.9
            )
        else:
            # Default to AdamW
            logger.warning(f"Unknown optimizer type: {self.training_config.optimizer_type}. Using AdamW.")
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.training_config.learning_rate
            )
        
        return optimizer
    
    def _setup_scheduler(self) -> Optional[Any]:
        """
        Set up the learning rate scheduler based on the training configuration.
        
        Returns:
            Configured scheduler or None
        """
        # Skip if no scheduler specified
        if self.training_config.scheduler_type.lower() == "none":
            return None
        
        # Calculate training steps for schedule
        train_dataset_length = len(self.train_dataloader.dataset)
        num_steps_per_epoch = train_dataset_length // self.training_config.batch_size
        if train_dataset_length % self.training_config.batch_size != 0:
            num_steps_per_epoch += 1
        
        # Account for gradient accumulation
        num_steps_per_epoch = math.ceil(num_steps_per_epoch / self.training_config.accumulation_steps)
        
        # Calculate total training steps
        total_training_steps = num_steps_per_epoch * self.training_config.max_epochs
        
        # Calculate warmup steps
        warmup_steps = self.training_config.warmup_steps
        if warmup_steps == 0 and self.training_config.warmup_ratio > 0:
            warmup_steps = int(total_training_steps * self.training_config.warmup_ratio)
        
        # Create the scheduler based on the specified type
        if self.training_config.scheduler_type.lower() == "linear":
            from torch.optim.lr_scheduler import LinearLR
            return LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_training_steps - warmup_steps,
                last_epoch=-1
            )
        elif self.training_config.scheduler_type.lower() == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                self.optimizer,
                T_max=total_training_steps - warmup_steps,
                eta_min=1e-6
            )
        elif self.training_config.scheduler_type.lower() == "cosine_with_restarts":
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=num_steps_per_epoch,
                T_mult=2
            )
        elif self.training_config.scheduler_type.lower() == "constant":
            from torch.optim.lr_scheduler import ConstantLR
            return ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=total_training_steps
            )
        elif self.training_config.scheduler_type.lower() == "constant_with_warmup":
            # Create a custom warmup scheduler
            class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
                def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
                    def lr_lambda(current_step):
                        if current_step < warmup_steps:
                            return float(current_step) / float(max(1, warmup_steps))
                        return 1.0
                    super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)
            
            return WarmupConstantSchedule(
                self.optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_training_steps
            )
        
        # Default to None
        logger.warning(f"Unknown scheduler type: {self.training_config.scheduler_type}. No scheduler will be used.")
        return None
    
    def _setup_checkpoint_manager(self) -> CheckpointManager:
        """
        Set up the checkpoint manager.
        
        Returns:
            Configured checkpoint manager
        """
        output_dir = os.path.join(
            self.training_config.output_dir,
            "checkpoints"
        )
        
        return CheckpointManager(
            output_dir=output_dir,
            max_checkpoints=self.training_config.save_total_limit,
            save_best_metric="loss",
            lower_better=True,
            save_interval_epochs=1 if self.training_config.save_strategy == "epochs" else 0,
            save_interval_steps=self.training_config.save_steps if self.training_config.save_strategy == "steps" else 0
        )
    
    def _setup_amp(self) -> Optional[torch.cuda.amp.GradScaler]:
        """
        Set up automatic mixed precision training.
        
        Returns:
            GradScaler for mixed precision or None if not enabled
        """
        if self.training_config.use_mixed_precision:
            # Check for CUDA availability for mixed precision
            if torch.cuda.is_available():
                return torch.cuda.amp.GradScaler()
            else:
                logger.warning("Mixed precision training is not available without CUDA. Disabling.")
                return None
        return None
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        This method orchestrates the complete training process, including:
        - Initialization
        - Training loop (epochs and steps)
        - Evaluation
        - Checkpointing
        - Cleanup
        
        Returns:
            Dictionary with training results and metrics
        """
        # Run pre-training hooks
        self._run_hooks(self.pre_training_hooks)
        
        # Record training start time
        self.training_start_time = time.time()
        self.last_log_time = self.training_start_time
        
        # Try to resume from checkpoint if requested
        if self.training_config.auto_resume or self.training_config.resume_from_checkpoint:
            self._resume_from_checkpoint()
        
        logger.info(f"Starting training for {self.training_config.max_epochs} epochs")
        
        # Training loop
        for epoch in range(self.epoch, self.training_config.max_epochs):
            self.epoch = epoch
            
            # Run pre-epoch hooks
            self._run_hooks(self.pre_epoch_hooks, epoch=epoch)
            
            # Training epoch
            train_metrics = self._train_epoch()
            
            # Store training metrics
            self.metrics_history["train"].append(train_metrics)
            
            # Evaluation if needed
            if self.eval_dataloader is not None and self.training_config.should_evaluate(epoch, self.global_step):
                eval_metrics = self.evaluate()
                self.metrics_history["eval"].append(eval_metrics)
                
                # Track best model for checkpointing
                if eval_metrics["loss"] < self.best_metric:
                    self.best_metric = eval_metrics["loss"]
                    if self.training_config.save_only_best:
                        self._save_checkpoint(is_best=True)
            
            # Save checkpoint if needed
            if self.training_config.should_save_checkpoint(epoch, self.global_step):
                self._save_checkpoint()
            
            # Run post-epoch hooks
            self._run_hooks(self.post_epoch_hooks, epoch=epoch, metrics=train_metrics)
            
            # Log epoch summary
            self._log_epoch_summary(epoch, train_metrics)
        
        # Final evaluation
        final_metrics = {}
        if self.eval_dataloader is not None:
            final_metrics = self.evaluate()
            logger.info(f"Final evaluation metrics: {final_metrics}")
        
        # Save final checkpoint
        self._save_checkpoint(is_final=True)
        
        # Calculate training duration
        training_duration = time.time() - self.training_start_time
        hours, remainder = divmod(training_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Run post-training hooks
        self._run_hooks(self.post_training_hooks, metrics=final_metrics)
        
        # Return training results
        return {
            "epochs_completed": self.epoch + 1,
            "global_steps": self.global_step,
            "best_metric": self.best_metric,
            "final_metrics": final_metrics,
            "training_duration": training_duration,
            "metrics_history": self.metrics_history
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with metrics for the epoch
        """
        self.model.train()
        epoch_start_time = time.time()
        
        # Reset epoch metrics
        epoch_loss = 0.0
        epoch_metrics = {}
        num_samples = 0
        
        # Create progress logger
        num_batches = len(self.train_dataloader)
        log_interval = max(1, num_batches // 10)  # Log roughly 10 times per epoch
        
        # Iterate over batches
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Convert batch to tensors on the device
            batch = self._batch_to_device(batch)
            
            # Run pre-step hooks
            self._run_hooks(self.pre_step_hooks, batch=batch, batch_idx=batch_idx)
            
            # Forward and backward pass
            step_metrics = self._train_step(batch, batch_idx)
            
            # Update epoch metrics
            batch_size = self._get_batch_size(batch)
            num_samples += batch_size
            epoch_loss += step_metrics["loss"] * batch_size
            
            # Accumulate other metrics
            for key, value in step_metrics.items():
                if key != "loss":
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    epoch_metrics[key] += value * batch_size
            
            # Run post-step hooks
            self._run_hooks(self.post_step_hooks, metrics=step_metrics, batch_idx=batch_idx)
            
            # Log progress
            if batch_idx % log_interval == 0 or batch_idx == num_batches - 1:
                self._log_step_progress(batch_idx, num_batches, step_metrics)
        
        # Calculate average metrics
        epoch_loss /= num_samples
        for key in epoch_metrics:
            epoch_metrics[key] /= num_samples
        
        # Add loss to metrics dictionary
        epoch_metrics["loss"] = epoch_loss
        
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        epoch_metrics["duration"] = epoch_duration
        
        return epoch_metrics
    
    def _train_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary with metrics for the step
        """
        # Determine if this is an accumulation step
        is_accumulation_step = (batch_idx + 1) % self.training_config.accumulation_steps != 0
        
        # Prepare for mixed precision if enabled
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = self._forward(batch)
                loss = outputs["loss"]
                
                # Scale loss for gradient accumulation
                loss = loss / self.training_config.accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Optimize if not accumulating gradients
            if not is_accumulation_step:
                # Clip gradients if specified
                if self.training_config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.training_config.max_grad_norm
                    )
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Update learning rate
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # Increment global step
                self.global_step += 1
        else:
            # Standard precision training
            # Forward pass
            outputs = self._forward(batch)
            loss = outputs["loss"]
            
            # Scale loss for gradient accumulation
            loss = loss / self.training_config.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Optimize if not accumulating gradients
            if not is_accumulation_step:
                # Clip gradients if specified
                if self.training_config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.training_config.max_grad_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update learning rate
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # Increment global step
                self.global_step += 1
        
        # Extract metrics from outputs
        metrics = {"loss": loss.item() * self.training_config.accumulation_steps}  # Unscale the loss
        
        # Add other metrics from outputs
        for key, value in outputs.items():
            if key != "loss" and isinstance(value, (int, float, torch.Tensor)):
                if isinstance(value, torch.Tensor):
                    # Only convert to item if it's a scalar tensor
                    if value.numel() == 1:
                        value = value.item()
                    else:
                        # Skip large tensors or use mean value
                        if key not in ['hidden_states', 'attentions', 'cross_attentions']:
                            value = value.mean().item()
                        else:
                            # Skip attention tensors entirely
                            continue
                metrics[key] = value
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Skip if no evaluation dataloader
        if self.eval_dataloader is None:
            logger.warning("No evaluation dataloader provided. Skipping evaluation.")
            return {}
        
        # Run pre-evaluation hooks
        self._run_hooks(self.pre_eval_hooks)
        
        # Set model to evaluation mode
        self.model.eval()
        eval_start_time = time.time()
        
        # Reset evaluation metrics
        eval_loss = 0.0
        eval_metrics = {}
        num_samples = 0
        
        # Disable gradient computation for evaluation
        with torch.no_grad():
            # Iterate over batches
            for batch_idx, batch in enumerate(self.eval_dataloader):
                # Convert batch to tensors on the device
                batch = self._batch_to_device(batch)
                
                # Forward pass
                outputs = self._forward(batch, is_training=False)
                loss = outputs["loss"]
                
                # Update evaluation metrics
                batch_size = self._get_batch_size(batch)
                num_samples += batch_size
                eval_loss += loss.item() * batch_size
                
                # Accumulate other metrics
                for key, value in outputs.items():
                    if key != "loss" and isinstance(value, (int, float, torch.Tensor)):
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        
                        if key not in eval_metrics:
                            eval_metrics[key] = 0.0
                        
                        eval_metrics[key] += value * batch_size
        
        # Calculate average metrics
        eval_loss /= num_samples
        for key in eval_metrics:
            eval_metrics[key] /= num_samples
        
        # Add loss to metrics dictionary
        eval_metrics["loss"] = eval_loss
        
        # Calculate evaluation duration
        eval_duration = time.time() - eval_start_time
        eval_metrics["duration"] = eval_duration
        
        # Log evaluation results
        logger.info(f"Evaluation results at step {self.global_step}: Loss = {eval_loss:.4f}, Duration = {eval_duration:.2f}s")
        for key, value in eval_metrics.items():
            if key not in ["loss", "duration"]:
                logger.info(f"  {key} = {value:.4f}")
        
        # Run post-evaluation hooks
        self._run_hooks(self.post_eval_hooks, metrics=eval_metrics)
        
        # Return evaluation metrics
        return eval_metrics
    
    def _forward(self, batch: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        """
        Forward pass through the model.
        
        This method should be overridden by subclasses to handle specific
        model forward pass requirements.
        
        Args:
            batch: Batch of data
            is_training: Whether this is a training forward pass
            
        Returns:
            Dictionary with model outputs, including loss
        """
        # Default implementation assumes batch has input_ids, attention_mask, and labels
        # This should be overridden by specialized trainers
        return self.model(**batch)
    
    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move batch tensors to the device.
        
        Args:
            batch: Batch of data
            
        Returns:
            Batch with tensors moved to the device
        """
        # Move tensors to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        
        return batch
    
    def _get_batch_size(self, batch: Dict[str, Any]) -> int:
        """
        Get the batch size from a batch.
        
        Args:
            batch: Batch of data
            
        Returns:
            Batch size
        """
        # Try to get batch size from common keys
        for key in ["input_ids", "inputs", "x", "data"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                return batch[key].size(0)
        
        # Fall back to the first tensor in the batch
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value.size(0)
        
        # Default batch size
        return 1
    
    def _resume_from_checkpoint(self) -> bool:
        """
        Resume training from a checkpoint.
        
        Returns:
            True if training was resumed, False otherwise
        """
        checkpoint_path = self.training_config.resume_from_checkpoint
        
        # If auto_resume is enabled, find the latest checkpoint
        if checkpoint_path is None and self.training_config.auto_resume:
            checkpoint_path = "latest"
        
        # Skip if no checkpoint path
        if checkpoint_path is None:
            return False
        
        # Load checkpoint
        metadata = self.checkpoint_manager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            path=None if checkpoint_path == "latest" else checkpoint_path
        )
        
        # Update training state from checkpoint
        if "epoch" in metadata:
            self.epoch = metadata["epoch"]
        
        if "step" in metadata:
            self.global_step = metadata["step"]
        
        if "best_metric" in metadata:
            self.best_metric = metadata["best_metric"]
        
        logger.info(f"Resumed training from checkpoint at epoch {self.epoch}, step {self.global_step}")
        return True
    
    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False) -> None:
        """
        Save a checkpoint.
        
        Args:
            is_best: Whether this is the best checkpoint so far
            is_final: Whether this is the final checkpoint
        """
        # Prepare checkpoint name
        checkpoint_name = None
        if is_final:
            checkpoint_name = f"final_checkpoint_epoch_{self.epoch}_step_{self.global_step}.pt"
        elif is_best:
            checkpoint_name = f"best_checkpoint_epoch_{self.epoch}_step_{self.global_step}.pt"
        
        # Collect metrics
        metrics = {
            "loss": self.accumulated_loss / max(1, self.global_step),
            **self.accumulated_metrics
        }
        
        # Save checkpoint using checkpoint manager
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            epoch=self.epoch,
            step=self.global_step,
            loss=metrics["loss"],
            metrics=metrics,
            filename=checkpoint_name,
            force=is_best or is_final
        )
        
        if checkpoint_path:
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _log_step_progress(self, batch_idx: int, num_batches: int, metrics: Dict[str, float]) -> None:
        """
        Log progress during a training step.
        
        Args:
            batch_idx: Index of the current batch
            num_batches: Total number of batches
            metrics: Metrics from the current step
        """
        current_time = time.time()
        time_elapsed = current_time - self.last_log_time
        self.last_log_time = current_time
        
        # Calculate progress
        progress = (batch_idx + 1) / num_batches * 100
        
        # Calculate speed
        examples_per_second = (
            self.training_config.batch_size / time_elapsed
            if time_elapsed > 0 else 0
        )
        
        # Log progress
        logger.info(
            f"Epoch {self.epoch+1}/{self.training_config.max_epochs} "
            f"[{batch_idx+1}/{num_batches} ({progress:.0f}%)] "
            f"Loss: {metrics['loss']:.4f}, "
            f"Speed: {examples_per_second:.1f} examples/s"
        )
    
    def _log_epoch_summary(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log summary of an epoch.
        
        Args:
            epoch: Current epoch
            metrics: Metrics from the epoch
        """
        # Calculate epoch duration
        duration = metrics.get("duration", 0)
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Log summary
        logger.info(
            f"Epoch {epoch+1}/{self.training_config.max_epochs} completed in "
            f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        )
        
        # Log metrics
        logger.info(f"Epoch {epoch+1} metrics:")
        for key, value in metrics.items():
            if key != "duration":
                logger.info(f"  {key}: {value:.4f}")
    
    def _run_hooks(self, hooks: List[Callable], **kwargs) -> None:
        """
        Run a list of hooks.
        
        Args:
            hooks: List of hook functions to run
            **kwargs: Additional arguments to pass to the hooks
        """
        for hook in hooks:
            try:
                hook(self, **kwargs)
            except Exception as e:
                logger.error(f"Error running hook {hook.__name__}: {e}")
    
    def add_hook(self, hook_type: str, hook_fn: Callable) -> None:
        """
        Add a hook to the trainer.
        
        Args:
            hook_type: Type of hook ('pre_training', 'post_training', etc.)
            hook_fn: Hook function to add
        """
        if hook_type == "pre_training":
            self.pre_training_hooks.append(hook_fn)
        elif hook_type == "post_training":
            self.post_training_hooks.append(hook_fn)
        elif hook_type == "pre_epoch":
            self.pre_epoch_hooks.append(hook_fn)
        elif hook_type == "post_epoch":
            self.post_epoch_hooks.append(hook_fn)
        elif hook_type == "pre_step":
            self.pre_step_hooks.append(hook_fn)
        elif hook_type == "post_step":
            self.post_step_hooks.append(hook_fn)
        elif hook_type == "pre_eval":
            self.pre_eval_hooks.append(hook_fn)
        elif hook_type == "post_eval":
            self.post_eval_hooks.append(hook_fn)
        else:
            logger.warning(f"Unknown hook type: {hook_type}")
    
    def get_learning_rate(self) -> float:
        """
        Get the current learning rate.
        
        Returns:
            Current learning rate
        """
        return self.optimizer.param_groups[0]["lr"]
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the trainer configuration.
        
        Returns:
            Training configuration
        """
        return {
            "training": self.training_config.to_dict() if hasattr(self.training_config, "to_dict") else self.training_config,
            "model": self.model_config.to_dict() if hasattr(self.model_config, "to_dict") else self.model_config,
            "data": self.data_config.to_dict() if hasattr(self.data_config, "to_dict") else self.data_config,
        }