"""
Enhanced trainer for the Quantum Resonance Language Model.

This module provides a modular, extensible trainer that supports various model types,
training strategies, and extensions through a plugin architecture.
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Type

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.config.data_config import DataConfig
from src.utils.device import get_device

from src.training.model_adapters import (
    ModelAdapter, get_model_adapter,
    StandardModelAdapter, DialogueModelAdapter, MultimodalModelAdapter
)
from src.training.strategies import (
    TrainingStrategy, get_training_strategy, 
    StandardTrainingStrategy, FinetuningStrategy
)
from src.training.extensions import ExtensionManager
from src.training.checkpoints import CheckpointManager
from src.training.metrics import MetricsLogger


class EnhancedTrainer:
    """
    Enhanced trainer implementation for QLLM.
    
    This trainer provides a modular architecture that supports different model types,
    training strategies, and extensions through a plugin system. It handles the
    coordination between these components and provides a unified interface for
    training and evaluation.
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
        metrics_logger: Optional[MetricsLogger] = None
    ):
        """
        Initialize the enhanced trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            data_config: Data configuration
            output_dir: Directory for outputs (default: from training_config)
            logger: Logger instance
            model_adapter: Model adapter instance (created from config if None)
            training_strategy: Training strategy instance (created from config if None)
            extension_manager: Extension manager instance (created from config if None)
            checkpoint_manager: Checkpoint manager instance (created from config if None)
            metrics_logger: Metrics logger instance (created from config if None)
        """
        # Store configurations
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        
        # Set output directory
        self.output_dir = output_dir or getattr(training_config, "output_dir", None)
        if not self.output_dir:
            # Default output directory
            self.output_dir = os.path.join("runs", "quantum_resonance")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set logger
        self.logger = logger or logging.getLogger("quantum_resonance")
        
        # Set device
        self.device = self._get_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components (use provided instances or create from config)
        self.model_adapter = model_adapter
        self.training_strategy = training_strategy
        self.extension_manager = extension_manager
        self.checkpoint_manager = checkpoint_manager
        self.metrics_logger = metrics_logger
        
        # Data components
        self.dataloaders: Dict[str, DataLoader] = {}
        
        # Training state
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Mixed precision support
        self.use_mixed_precision = getattr(training_config, "use_mixed_precision", True)
        self.scaler = None
        if self.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Using mixed precision training")
        
        # Save configuration
        self.save_config()
        
        # Initialize all components
        self.initialize()
    
    def _get_device(self) -> torch.device:
        """
        Get the appropriate device.
        
        Returns:
            Device to use
        """
        device_str = getattr(self.training_config, "device", None)
        return get_device(device_str)
    
    def save_config(self) -> None:
        """Save configuration to the output directory."""
        # Create config dictionary
        config = {
            "model": self.model_config.to_dict() if hasattr(self.model_config, "to_dict") else vars(self.model_config),
            "training": self.training_config.to_dict() if hasattr(self.training_config, "to_dict") else vars(self.training_config),
            "data": self.data_config.to_dict() if hasattr(self.data_config, "to_dict") else vars(self.data_config)
        }
        
        # Save config
        try:
            with open(os.path.join(self.output_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
            self.logger.info(f"Configuration saved to {os.path.join(self.output_dir, 'config.json')}")
        except Exception as e:
            self.logger.warning(f"Error saving configuration: {e}")
    
    def initialize(self) -> None:
        """Initialize all components of the training system."""
        # Initialize model adapter if not provided
        if self.model_adapter is None:
            model_type = getattr(self.training_config, "model_type", "standard")
            self.model_adapter = get_model_adapter(
                model_type, 
                self.model_config, 
                self.training_config, 
                self.device, 
                self.logger
            )
            self.logger.info(f"Using model adapter: {type(self.model_adapter).__name__}")
        
        # Initialize training strategy if not provided
        if self.training_strategy is None:
            strategy_type = getattr(self.training_config, "training_strategy", "standard")
            self.training_strategy = get_training_strategy(
                strategy_type,
                self.training_config,
                self.logger
            )
            self.logger.info(f"Using training strategy: {type(self.training_strategy).__name__}")
        
        # Initialize extension manager if not provided
        if self.extension_manager is None:
            self.extension_manager = ExtensionManager(
                self.training_config,
                self.logger
            )
            self.logger.info(f"Extension manager initialized with {len(self.extension_manager.get_extension_names())} extensions")
        
        # Initialize checkpoint manager if not provided
        if self.checkpoint_manager is None:
            self.checkpoint_manager = CheckpointManager(
                self.training_config,
                self.output_dir,
                self.logger
            )
            self.logger.info(f"Checkpoint manager initialized")
        
        # Initialize metrics logger if not provided
        if self.metrics_logger is None:
            self.metrics_logger = MetricsLogger(
                self.output_dir,
                log_to_console=True,
                log_to_tensorboard=getattr(self.training_config, "use_tensorboard", True),
                log_to_file=True,
                logger=self.logger
            )
            self.logger.info(f"Metrics logger initialized")
        
        # Initialize model and tokenizer through the adapter
        self.initialize_model_and_tokenizer()
        
        # Initialize dataloaders
        self.initialize_dataloaders()
        
        # Initialize optimizer and scheduler
        self.initialize_optimizer_and_scheduler()
        
        # Try to resume from checkpoint if enabled
        if getattr(self.training_config, "auto_resume", False):
            self.try_resume_from_checkpoint()
    
    def initialize_model_and_tokenizer(self) -> None:
        """Initialize model and tokenizer using the model adapter."""
        self.logger.info("Initializing model and tokenizer")
        
        # Create model through adapter
        self.model = self.model_adapter.create_model()
        
        # Register extensions with model
        self.model = self.extension_manager.register_with_model(self.model)
        
        # Create tokenizer through adapter
        self.tokenizer = self.model_adapter.create_tokenizer()
        
        # Set tokenizer in adapter (in case it was created externally)
        self.model_adapter.set_tokenizer(self.tokenizer)
        
        # Set model in adapter (in case it was created externally)
        self.model_adapter.set_model(self.model)
    
    def initialize_dataloaders(self) -> None:
        """Initialize dataloaders from configuration."""
        self.logger.info("Initializing dataloaders")
        
        # Import the appropriate dataloaders function
        from src.data.dataloaders import get_appropriate_dataloaders
        
        # Get batch sizes from training config
        batch_size = getattr(self.training_config, "batch_size", 8)
        eval_batch_size = getattr(self.training_config, "eval_batch_size", batch_size)
        
        # Get dataloaders
        self.dataloaders = get_appropriate_dataloaders(
            data_config=self.data_config,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=getattr(self.data_config, "preprocessing_num_workers", 0)
        )
        
        # Log dataset sizes
        self.logger.info(f"Dataset sizes:")
        for split, dataloader in self.dataloaders.items():
            self.logger.info(f"  {split}: {len(dataloader.dataset)} examples, {len(dataloader)} batches")
    
    def initialize_optimizer_and_scheduler(self) -> None:
        """Initialize optimizer and learning rate scheduler from training strategy."""
        self.logger.info("Initializing optimizer and scheduler")
        
        # Calculate total training steps
        num_epochs = getattr(self.training_config, "max_epochs", 1)
        steps_per_epoch = len(self.dataloaders.get("train", []))
        total_steps = num_epochs * steps_per_epoch
        self.logger.info(f"Training for {num_epochs} epochs, {steps_per_epoch} steps per epoch, {total_steps} total steps")
        
        # Create optimizer from training strategy
        optimizer_type = getattr(self.training_config, "optimizer", "adamw")
        self.optimizer = self.training_strategy.create_optimizer(
            self.model,
            optimizer_type=optimizer_type
        )
        
        # Create scheduler from training strategy
        scheduler_type = getattr(self.training_config, "lr_scheduler", "linear")
        self.scheduler = self.training_strategy.create_scheduler(
            self.optimizer,
            scheduler_type=scheduler_type,
            num_training_steps=total_steps,
            num_warmup_steps=getattr(self.training_config, "warmup_steps", int(0.1 * total_steps))
        )
        
        self.logger.info(f"Using optimizer {optimizer_type} with learning rate {self.optimizer.param_groups[0]['lr']}")
        if self.scheduler:
            self.logger.info(f"Using learning rate scheduler: {scheduler_type}")
    
    def try_resume_from_checkpoint(self) -> bool:
        """
        Try to resume training from the latest checkpoint.
        
        Returns:
            True if successfully resumed, False otherwise
        """
        self.logger.info("Checking for existing checkpoints to resume from")
        
        # Find latest checkpoint
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        if latest_checkpoint is None:
            self.logger.info("No checkpoints found to resume from")
            return False
        
        try:
            # Load checkpoint
            checkpoint_info = self.checkpoint_manager.load_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                path=latest_checkpoint
            )
            
            # Restore state
            if "global_step" in checkpoint_info:
                self.global_step = checkpoint_info["global_step"]
            if "epoch" in checkpoint_info:
                self.current_epoch = checkpoint_info["epoch"]
            
            self.logger.info(f"Resumed from checkpoint at epoch {self.current_epoch}, global step {self.global_step}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Error resuming from checkpoint: {e}")
            return False
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the model for a specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train (default: from training_config)
            
        Returns:
            Dictionary of training statistics
        """
        # Set number of epochs
        num_epochs = num_epochs or getattr(self.training_config, "max_epochs", 1)
        
        # Check if components are initialized
        if self.model is None or self.tokenizer is None or self.optimizer is None:
            self.initialize()
        
        # Get train dataloader
        train_dataloader = self.dataloaders.get("train")
        if not train_dataloader:
            raise ValueError("No training dataloader available")
        
        # Log training start
        self.logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        # Initialize training state
        start_epoch = self.current_epoch
        
        # Extension hooks: pre-training
        self.extension_manager.hooks.execute_hooks("pre_epoch", self.model, start_epoch)
        
        # Main training loop
        train_stats = {}
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.current_epoch = epoch
            
            # Log epoch start
            self.logger.info(f"Starting epoch {epoch+1}/{start_epoch + num_epochs}")
            epoch_start_time = time.time()
            
            # Extension hooks: pre-epoch
            self.extension_manager.pre_epoch(self.model, epoch)
            
            # Train for one epoch
            epoch_stats = self.train_epoch(train_dataloader)
            
            # Evaluate on validation set
            if "validation" in self.dataloaders:
                eval_stats = self.evaluate("validation")
                val_loss = eval_stats.get("loss", float("inf"))
                
                # Log validation results
                self.metrics_logger.log_evaluation(
                    eval_stats,
                    step=self.global_step,
                    epoch=epoch,
                    split="val"
                )
                
                # Check for best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.logger.info(f"New best validation loss: {val_loss:.6f}")
                    
                    # Save best model
                    best_model_path = os.path.join(self.output_dir, "best_model.pt")
                    self.save_checkpoint(best_model_path)
                
                # Check for early stopping if using the finetune strategy
                if isinstance(self.training_strategy, FinetuningStrategy):
                    if self.training_strategy.check_early_stopping(val_loss):
                        self.logger.info("Early stopping triggered")
                        break
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
            
            # Extension hooks: post-epoch
            epoch_metrics = {**epoch_stats, **(eval_stats if "validation" in self.dataloaders else {})}
            self.extension_manager.post_epoch(self.model, epoch, epoch_metrics)
            
            # Store epoch stats
            train_stats[f"epoch_{epoch+1}"] = epoch_stats
            if "validation" in self.dataloaders:
                train_stats[f"epoch_{epoch+1}_eval"] = eval_stats
            
            # Save checkpoint if configured
            if self.checkpoint_manager.should_save_checkpoint(epoch, self.global_step):
                checkpoint_path = self.checkpoint_manager._get_checkpoint_path(epoch, self.global_step)
                self.save_checkpoint(checkpoint_path)
            
            # Log epoch metrics
            self.metrics_logger.log_epoch(
                {**epoch_stats, **(eval_stats if "validation" in self.dataloaders else {})},
                epoch
            )
        
        # Calculate total training time
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        
        # Extension hooks: post-training
        self.extension_manager.hooks.execute_hooks("post_epoch", self.model, self.current_epoch)
        
        # Save final model
        self.save_checkpoint(os.path.join(self.output_dir, "final_model.pt"))
        
        # Log best metrics
        self.metrics_logger.log_best_metrics()
        
        # Close metrics logger
        self.metrics_logger.close()
        
        return train_stats
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            
        Returns:
            Dictionary of epoch statistics
        """
        # Set model to training mode
        self.model.train()
        
        # Initialize epoch statistics
        epoch_stats = {
            "train_loss": 0.0,
            "steps": 0,
            "examples": 0,
            "grad_norm": 0.0
        }
        
        # Training loop
        num_batches = len(dataloader)
        for batch_idx, batch in enumerate(dataloader):
            # Extension hooks: pre-batch
            self.extension_manager.pre_batch(self.model, batch)
            
            # Prepare batch with model adapter
            prepared_batch = self.model_adapter.prepare_batch(batch)
            
            # Determine if gradients should be updated (for gradient accumulation)
            accumulation_steps = getattr(self.training_config, "accumulation_steps", 1)
            update_gradients = (batch_idx + 1) % accumulation_steps == 0 or batch_idx == num_batches - 1
            
            # Train step with strategy
            step_metrics = self.training_strategy.train_step(
                self.model,
                prepared_batch,
                self.optimizer,
                self.scaler,
                self.scheduler,
                update_gradients
            )
            
            # Update statistics
            epoch_stats["train_loss"] += step_metrics.get("loss", 0.0)
            epoch_stats["steps"] += 1
            epoch_stats["examples"] += prepared_batch.get("input_ids", prepared_batch.get("inputs", [])).shape[0]
            
            # Update global step if gradients were updated
            if update_gradients:
                self.global_step += 1
                
                # Extension hooks: post-batch
                self.extension_manager.post_batch(
                    self.model, 
                    prepared_batch, 
                    step_metrics.get("loss", 0.0), 
                    self.global_step
                )
                
                # Log step metrics
                self.metrics_logger.log_training_step(
                    step_metrics,
                    step=self.global_step,
                    epoch=self.current_epoch
                )
                
                # Accumulate grad norm
                if "grad_norm" in step_metrics:
                    epoch_stats["grad_norm"] += step_metrics["grad_norm"]
            
            # Log progress periodically
            if batch_idx % max(1, num_batches // 10) == 0:
                completion = (batch_idx + 1) / num_batches
                self.logger.info(
                    f"Epoch {self.current_epoch+1} progress: {completion:.1%} "
                    f"({batch_idx+1}/{num_batches}), "
                    f"loss: {step_metrics.get('loss', 0.0):.6f}"
                )
        
        # Compute average metrics
        if epoch_stats["steps"] > 0:
            epoch_stats["train_loss"] /= epoch_stats["steps"]
            epoch_stats["grad_norm"] /= max(1, epoch_stats["steps"] // accumulation_steps)
        
        return epoch_stats
    
    def evaluate(self, split: str = "validation") -> Dict[str, Any]:
        """
        Evaluate the model on a specific dataset split.
        
        Args:
            split: Dataset split to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get dataloader for the specified split
        dataloader = self.dataloaders.get(split)
        if not dataloader:
            self.logger.warning(f"No dataloader available for split: {split}")
            return {}
        
        # Extension hooks: pre-validation
        self.extension_manager.pre_validation(self.model)
        
        # Validate with strategy
        validation_metrics = self.training_strategy.validate(
            self.model,
            dataloader,
            self.device
        )
        
        # Extension hooks: post-validation
        self.extension_manager.post_validation(self.model, validation_metrics)
        
        return validation_metrics
    
    def save_checkpoint(
        self,
        path: Optional[str] = None,
        include_optimizer: bool = True
    ) -> str:
        """
        Save a model checkpoint.
        
        Args:
            path: Path to save checkpoint (if None, generates a path)
            include_optimizer: Whether to include optimizer state
            
        Returns:
            Path to the saved checkpoint
        """
        # Collect extension data
        extension_data = {}
        for name, extension in self.extension_manager.get_extensions().items():
            if hasattr(extension, "get_state_dict") and callable(extension.get_state_dict):
                try:
                    extension_data[name] = extension.get_state_dict()
                except Exception as e:
                    self.logger.warning(f"Error getting state from extension {name}: {e}")
        
        # Collect additional metadata
        metadata = {
            "global_step": self.global_step,
            "version": "1.0.0",  # Version of checkpoint format
            "extension_names": self.extension_manager.get_extension_names()
        }
        
        # Save checkpoint with checkpoint manager
        return self.checkpoint_manager.save_checkpoint(
            self.model,
            self.optimizer if include_optimizer else None,
            self.scheduler if include_optimizer else None,
            epoch=self.current_epoch,
            global_step=self.global_step,
            model_config=self.model_config,
            training_config=self.training_config,
            extension_data=extension_data,
            metadata=metadata,
            path=path
        )
    
    def load_model_for_inference(self, path: str) -> nn.Module:
        """
        Load a model for inference.
        
        Args:
            path: Path to model checkpoint
            
        Returns:
            Loaded model
        """
        # Initialize model if needed
        if self.model is None:
            self.initialize_model_and_tokenizer()
        
        # Load checkpoint
        self.checkpoint_manager.load_checkpoint(
            self.model,
            path=path,
            load_optimizer=False,
            load_scheduler=False
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        return self.model
    
    def get_model_and_tokenizer(self) -> Tuple[nn.Module, Any]:
        """
        Get the model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if self.model is None or self.tokenizer is None:
            self.initialize_model_and_tokenizer()
        
        return self.model, self.tokenizer