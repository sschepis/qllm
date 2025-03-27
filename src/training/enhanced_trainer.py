"""
Enhanced trainer for the QLLM training system.

This module provides the core enhanced trainer implementation that coordinates
various components like model adapters, training strategies, and extensions.
"""

import os
import logging
import time
from typing import Dict, Any, Optional, Union, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.config.data_config import DataConfig

from src.training.model_adapters import ModelAdapter
from src.training.strategies import TrainingStrategy
from src.training.extensions import ExtensionManager
from src.training.checkpoints import CheckpointManager
from src.training.metrics import MetricsLogger


class EnhancedTrainer:
    """
    Enhanced trainer for the QLLM training system.
    
    This trainer coordinates model adapters, training strategies, and extensions
    to provide a flexible and modular training experience.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        output_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        model_adapter: Optional[ModelAdapter] = None,
        training_strategy: Optional[TrainingStrategy] = None,
        extension_manager: Optional[ExtensionManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        metrics_logger: Optional[Any] = None
    ):
        """
        Initialize the enhanced trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            data_config: Data configuration
            output_dir: Directory for outputs
            logger: Logger instance
            model_adapter: Model adapter instance
            training_strategy: Training strategy instance
            extension_manager: Extension manager instance
            checkpoint_manager: Checkpoint manager instance
            metrics_logger: Metrics logger instance
        """
        # Configure logger
        self.logger = logger or logging.getLogger("quantum_resonance")
        
        # Store configurations
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        
        # Set output directory
        self.output_dir = output_dir
        if self.output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_name = getattr(model_config, "model_name", "quantum_resonance")
            self.output_dir = os.path.join("runs", f"{model_name}_{timestamp}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        
        # Set model adapter
        self.model_adapter = model_adapter
        
        # Set training strategy
        self.training_strategy = training_strategy
        
        # Set extension manager
        self.extension_manager = extension_manager
        
        # Set checkpoint manager
        self.checkpoint_manager = checkpoint_manager
        
        # Set metrics logger
        self.metrics_logger = metrics_logger
        
        # Initialize components to None (will be set in initialize_* methods)
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.dataloaders = {}
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.early_stop = False
        
        # Save configuration
        self._save_config()
    
    def initialize_model(self) -> nn.Module:
        """
        Initialize the model.
        
        Returns:
            Initialized model
        """
        self.logger.info("Initializing model")
        
        # Create model using adapter
        if self.model_adapter is None:
            raise ValueError("Model adapter not set. Call set_model_adapter first.")
        
        self.model = self.model_adapter.create_model()
        
        # Register model with adapter
        self.model_adapter.set_model(self.model)
        
        # Register extensions with model if available
        if self.extension_manager is not None:
            self.model = self.extension_manager.register_with_model(self.model)
        
        # Set model as attribute on trainer for easy access
        if hasattr(self.model, "trainer"):
            self.logger.warning("Model already has 'trainer' attribute, not overwriting")
        else:
            self.model.trainer = self
        
        # Set adapter as attribute on model for easy access
        if hasattr(self.model, "adapter"):
            self.logger.warning("Model already has 'adapter' attribute, not overwriting")
        else:
            self.model.adapter = self.model_adapter
        
        return self.model
    
    def initialize_tokenizer(self) -> Any:
        """
        Initialize the tokenizer.
        
        Returns:
            Initialized tokenizer
        """
        self.logger.info("Initializing tokenizer")
        
        # Create tokenizer using adapter
        if self.model_adapter is None:
            raise ValueError("Model adapter not set. Call set_model_adapter first.")
        
        self.tokenizer = self.model_adapter.create_tokenizer()
        
        # Register tokenizer with adapter
        self.model_adapter.set_tokenizer(self.tokenizer)
        
        return self.tokenizer
    
    def initialize_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Initialize dataloaders.
        
        Returns:
            Dictionary of dataloaders
        """
        self.logger.info("Initializing dataloaders")
        
        # Import dataloader utils
        from src.data.dataloader_utils import create_dataloaders
        
        # Create dataloaders
        self.dataloaders = create_dataloaders(
            self.data_config,
            self.tokenizer,
            self.training_config.batch_size,
            self.training_config.num_workers
        )
        
        return self.dataloaders
    
    def initialize_optimizer(self) -> Tuple[torch.optim.Optimizer, Any]:
        """
        Initialize optimizer and scheduler.
        
        Returns:
            Tuple of (optimizer, scheduler)
        """
        self.logger.info("Initializing optimizer and scheduler")
        
        # Check if model is initialized
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model first.")
        
        # Get optimizer type
        optimizer_type = getattr(self.training_config, "optimizer_type", "adamw")
        
        # Create optimizer using strategy
        self.optimizer = self.training_strategy.create_optimizer(
            self.model,
            optimizer_type=optimizer_type,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
        # Get scheduler type
        scheduler_type = getattr(self.training_config, "scheduler_type", "linear")
        
        # Calculate number of training steps
        if "train" not in self.dataloaders:
            self.logger.warning("Training dataloader not initialized. Cannot calculate number of training steps.")
            num_training_steps = 1000
        else:
            train_dataloader = self.dataloaders["train"]
            steps_per_epoch = len(train_dataloader)
            num_training_steps = steps_per_epoch * self.training_config.max_epochs
        
        # Calculate number of warmup steps
        num_warmup_steps = int(self.training_config.warmup_ratio * num_training_steps)
        
        # Create scheduler using strategy
        self.scheduler = self.training_strategy.create_scheduler(
            self.optimizer,
            scheduler_type=scheduler_type,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps
        )
        
        # Initialize gradient scaler for mixed precision training if enabled
        self.scaler = None
        if self.training_config.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Using mixed precision training with gradient scaling")
        
        return self.optimizer, self.scheduler
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Dictionary of training results
        """
        # Check if model and optimizer are initialized
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model first.")
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized. Call initialize_optimizer first.")
        if "train" not in self.dataloaders:
            raise ValueError("Training dataloader not initialized. Call initialize_dataloaders first.")
        
        # Get training parameters
        max_epochs = self.training_config.max_epochs
        accumulation_steps = getattr(self.training_config, "accumulation_steps", 1)
        
        # Log training start
        self.logger.info(f"Starting training for {max_epochs} epochs")
        self.logger.info(f"Accumulation steps: {accumulation_steps}")
        
        # Training loop
        for epoch in range(self.current_epoch, max_epochs):
            # Update current epoch
            self.current_epoch = epoch
            
            # Call extension manager pre-epoch hooks
            if self.extension_manager is not None:
                self.extension_manager.pre_epoch(self.model, epoch)
            
            # Train single epoch
            epoch_metrics = self.train_epoch()
            
            # Evaluate if validation dataloader exists
            val_metrics = None
            if "val" in self.dataloaders or "validation" in self.dataloaders:
                val_metrics = self.evaluate()
                
                # Log validation metrics
                if self.metrics_logger is not None:
                    self.metrics_logger.log_validation_metrics(val_metrics, epoch)
                
                # Check for early stopping if using finetune strategy
                if hasattr(self.training_strategy, "check_early_stopping"):
                    val_loss = val_metrics.get("loss", float('inf'))
                    if self.training_strategy.check_early_stopping(val_loss):
                        self.logger.info("Early stopping triggered")
                        self.early_stop = True
                        break
            
            # Save checkpoint
            if self.checkpoint_manager is not None:
                # Check if checkpoint should be saved
                if self.checkpoint_manager.should_save_checkpoint(epoch, self.global_step):
                    # Get extension data if available
                    extension_data = None
                    if self.extension_manager is not None:
                        extension_data = {
                            name: extension.get_state_dict() if hasattr(extension, "get_state_dict") else None
                            for name, extension in self.extension_manager.get_extensions().items()
                        }
                    
                    # Save checkpoint
                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch=epoch,
                        global_step=self.global_step,
                        model_config=self.model_config,
                        training_config=self.training_config,
                        extension_data=extension_data,
                        metadata={
                            "epoch_metrics": epoch_metrics,
                            "val_metrics": val_metrics
                        }
                    )
                    
                    # Log checkpoint
                    self.logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Call extension manager post-epoch hooks
            if self.extension_manager is not None:
                self.extension_manager.post_epoch(self.model, epoch, epoch_metrics)
            
            # Check for early stopping
            if self.early_stop:
                self.logger.info("Early stopping")
                break
        
        # Log training end
        self.logger.info("Training completed")
        
        # Return training results
        return {
            "epochs_trained": self.current_epoch + 1,
            "global_step": self.global_step,
            "early_stopped": self.early_stop,
            "best_val_loss": self.best_val_loss
        }
    
    def train_epoch(self) -> Dict[str, Any]:
        """
        Train the model for a single epoch.
        
        Returns:
            Dictionary of epoch metrics
        """
        # Set model to training mode
        self.model.train()
        
        # Get training dataloader
        train_dataloader = self.dataloaders["train"]
        
        # Initialize metrics
        epoch_metrics = {
            "train_loss": 0.0,
            "steps": 0
        }
        
        # Start timing
        start_time = time.time()
        
        # Get accumulation steps
        accumulation_steps = getattr(self.training_config, "accumulation_steps", 1)
        
        # Training loop
        for step, batch in enumerate(train_dataloader):
            # Call extension manager pre-batch hooks
            if self.extension_manager is not None:
                self.extension_manager.pre_batch(self.model, batch)
            
            # Check if this is an update step
            is_update_step = (step + 1) % accumulation_steps == 0 or step == len(train_dataloader) - 1
            
            # Train step
            step_metrics = self.training_strategy.train_step(
                self.model,
                batch,
                self.optimizer,
                self.scaler,
                self.scheduler,
                update_gradients=is_update_step
            )
            
            # Update metrics
            epoch_metrics["train_loss"] += step_metrics.get("loss", 0.0)
            epoch_metrics["steps"] += 1
            
            # Update global step if gradients were updated
            if is_update_step:
                self.global_step += 1
            
            # Call extension manager post-batch hooks
            if self.extension_manager is not None:
                self.extension_manager.post_batch(
                    self.model,
                    batch,
                    step_metrics.get("loss", 0.0),
                    self.global_step
                )
            
            # Log metrics
            if self.metrics_logger is not None and is_update_step:
                self.metrics_logger.log_training_step(step_metrics, self.global_step)
            
            # Log progress
            if step % 10 == 0:
                self.logger.debug(
                    f"Epoch {self.current_epoch+1} | Step {step}/{len(train_dataloader)} | "
                    f"Loss: {step_metrics.get('loss', 0.0):.4f} | "
                    f"LR: {self.training_strategy.get_learning_rate(self.optimizer):.2e}"
                )
        
        # Calculate average loss
        epoch_metrics["train_loss"] /= max(epoch_metrics["steps"], 1)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        epoch_metrics["epoch_time"] = epoch_time
        
        # Log epoch metrics
        self.logger.info(
            f"Epoch {self.current_epoch+1} completed in {epoch_time:.1f}s | "
            f"Loss: {epoch_metrics['train_loss']:.4f} | "
            f"Steps: {epoch_metrics['steps']}"
        )
        
        # Log metrics
        if self.metrics_logger is not None:
            self.metrics_logger.log_epoch_metrics(epoch_metrics, self.current_epoch)
        
        return epoch_metrics
    
    def evaluate(self, dataloader_name: str = "val") -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader_name: Name of the dataloader to use for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Check if model is initialized
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model first.")
        
        # Handle alternative dataloader name
        if dataloader_name not in self.dataloaders and dataloader_name == "val":
            dataloader_name = "validation"
        
        # Check if evaluation dataloader exists
        if dataloader_name not in self.dataloaders:
            raise ValueError(f"Dataloader '{dataloader_name}' not found.")
        
        # Get evaluation dataloader
        eval_dataloader = self.dataloaders[dataloader_name]
        
        # Call extension manager pre-validation hooks
        if self.extension_manager is not None:
            self.extension_manager.pre_validation(self.model)
        
        # Get device
        device = next(self.model.parameters()).device
        
        # Start timing
        start_time = time.time()
        
        # Evaluate model
        val_metrics = self.training_strategy.validate(
            self.model,
            eval_dataloader,
            device
        )
        
        # Calculate evaluation time
        eval_time = time.time() - start_time
        val_metrics["eval_time"] = eval_time
        
        # Update best validation loss
        if val_metrics.get("loss", float('inf')) < self.best_val_loss:
            self.best_val_loss = val_metrics.get("loss", float('inf'))
            val_metrics["is_best"] = True
        
        # Log validation metrics
        self.logger.info(
            f"Validation completed in {eval_time:.1f}s | "
            f"Loss: {val_metrics.get('loss', 'N/A'):.4f} | "
            f"Perplexity: {val_metrics.get('perplexity', 'N/A'):.4f}"
        )
        
        # Call extension manager post-validation hooks
        if self.extension_manager is not None:
            self.extension_manager.post_validation(self.model, val_metrics)
        
        return val_metrics
    
    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """
        Save a checkpoint of the model and training state.
        
        Args:
            path: Path to save checkpoint (if None, generates path)
            
        Returns:
            Path to the saved checkpoint
        """
        # Generate checkpoint path if not provided
        if path is None and self.checkpoint_manager is not None:
            path = self.checkpoint_manager._get_checkpoint_path(self.current_epoch, self.global_step)
        elif path is None:
            # Fallback if checkpoint manager not available
            os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
            path = os.path.join(
                self.output_dir, 
                "checkpoints", 
                f"checkpoint_epoch_{self.current_epoch+1}_step_{self.global_step}.pt"
            )
        
        # Get extension data if available
        extension_data = None
        if self.extension_manager is not None:
            extension_data = {
                name: extension.get_state_dict() if hasattr(extension, "get_state_dict") else None
                for name, extension in self.extension_manager.get_extensions().items()
            }
        
        # Save checkpoint
        if self.checkpoint_manager is not None:
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch=self.current_epoch,
                global_step=self.global_step,
                model_config=self.model_config,
                training_config=self.training_config,
                extension_data=extension_data,
                path=path
            )
        else:
            # Fallback if checkpoint manager not available
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "extension_data": extension_data
            }
            
            torch.save(checkpoint, path)
            checkpoint_path = path
        
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Dictionary with checkpoint info
        """
        # Check if model and optimizer are initialized
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model first.")
        
        # Load checkpoint
        if self.checkpoint_manager is not None:
            checkpoint_info = self.checkpoint_manager.load_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                path=path
            )
        else:
            # Fallback if checkpoint manager not available
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Checkpoint file not found: {path}")
            
            checkpoint = torch.load(path, map_location=next(self.model.parameters()).device)
            
            # Load model state
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
            # Load optimizer state if available
            if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Load scheduler state if available
            if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            # Update training state
            self.current_epoch = checkpoint.get("epoch", 0)
            self.global_step = checkpoint.get("global_step", 0)
            
            # Create checkpoint info
            checkpoint_info = {
                "path": path,
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "extension_data": checkpoint.get("extension_data", None)
            }
        
        # Update training state
        self.current_epoch = checkpoint_info.get("epoch", 0)
        self.global_step = checkpoint_info.get("global_step", 0)
        
        # Load extension data if available
        if self.extension_manager is not None and "extension_data" in checkpoint_info:
            extension_data = checkpoint_info["extension_data"]
            if extension_data:
                for name, state_dict in extension_data.items():
                    if name in self.extension_manager.get_extensions():
                        extension = self.extension_manager.get_extension(name)
                        if hasattr(extension, "load_state_dict") and state_dict is not None:
                            extension.load_state_dict(state_dict)
        
        self.logger.info(f"Loaded checkpoint from {path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch+1}, step {self.global_step}")
        
        return checkpoint_info
    
    def set_model_adapter(self, model_adapter: ModelAdapter) -> None:
        """
        Set the model adapter.
        
        Args:
            model_adapter: Model adapter instance
        """
        self.model_adapter = model_adapter
    
    def set_training_strategy(self, training_strategy: TrainingStrategy) -> None:
        """
        Set the training strategy.
        
        Args:
            training_strategy: Training strategy instance
        """
        self.training_strategy = training_strategy
    
    def set_extension_manager(self, extension_manager: ExtensionManager) -> None:
        """
        Set the extension manager.
        
        Args:
            extension_manager: Extension manager instance
        """
        self.extension_manager = extension_manager
    
    def set_checkpoint_manager(self, checkpoint_manager: Any) -> None:
        """
        Set the checkpoint manager.
        
        Args:
            checkpoint_manager: Checkpoint manager instance
        """
        self.checkpoint_manager = checkpoint_manager
    
    def set_metrics_logger(self, metrics_logger: Any) -> None:
        """
        Set the metrics logger.
        
        Args:
            metrics_logger: Metrics logger instance
        """
        self.metrics_logger = metrics_logger
    
    def _save_config(self) -> None:
        """Save configuration to output directory."""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save config if possible
        config_path = os.path.join(self.output_dir, "config.json")
        try:
            from src.config.config_manager import ConfigManager
            config_manager = ConfigManager()
            
            # Convert config objects to dict
            config_dict = {
                "model": config_manager.model_config_to_dict(self.model_config),
                "training": config_manager.training_config_to_dict(self.training_config),
                "data": config_manager.data_config_to_dict(self.data_config)
            }
            
            # Save config
            config_manager.save_config(config_dict, config_path)
            self.logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save configuration: {e}")