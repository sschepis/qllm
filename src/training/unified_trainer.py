"""
Unified trainer implementation for the Quantum Resonance Language Model.

This trainer consolidates functionality from all previous trainers (base, standard, 
enhanced, dialogue) into a single, highly configurable implementation using a 
composition-based architecture.
"""

import os
import logging
import time
import math
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.config.data_config import DataConfig

from src.training.model_adapters import ModelAdapter, get_model_adapter
from src.training.strategies import TrainingStrategy, get_training_strategy
from src.training.checkpoints import CheckpointManager
from src.training.extensions import ExtensionManager
from src.training.metrics import MetricsLogger


class UnifiedTrainer:
    """
    Unified trainer that consolidates functionality from all previous trainers.
    
    This trainer implementation uses composition rather than inheritance, making it
    highly configurable through its components (adapters, strategies, managers, etc.)
    rather than through subclassing.
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
        Initialize the unified trainer.
        
        Args:
            model_config: Configuration for the model architecture
            training_config: Configuration for training
            data_config: Configuration for data loading
            output_dir: Directory for outputs (checkpoints, logs, etc.)
            logger: Logger instance
            model_adapter: Model adapter for model-specific operations
            training_strategy: Training strategy for optimization approaches
            extension_manager: Manager for training extensions
            checkpoint_manager: Manager for checkpoint operations
            metrics_logger: Logger for training metrics
        """
        # Store configurations
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        
        # Set output directory
        self.output_dir = output_dir or getattr(training_config, "output_dir", "runs/quantum_resonance")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set logger
        self.logger = logger or logging.getLogger("quantum_resonance")
        
        # Initialize state
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.dataloaders = {}
        self.device = self._get_device()
        self.scaler = None  # For mixed precision training
        
        # Set up components (using provided or creating default instances)
        self._setup_components(
            model_adapter, 
            training_strategy, 
            extension_manager, 
            checkpoint_manager, 
            metrics_logger
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.training_start_time = None
        
        self.logger.info(f"Initialized UnifiedTrainer (output_dir: {self.output_dir})")
    
    def _setup_components(
        self,
        model_adapter: Optional[ModelAdapter] = None,
        training_strategy: Optional[TrainingStrategy] = None,
        extension_manager: Optional[ExtensionManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        metrics_logger: Optional[MetricsLogger] = None
    ) -> None:
        """
        Set up trainer components (adapters, strategies, managers, etc.).
        
        Args:
            model_adapter: Optional model adapter
            training_strategy: Optional training strategy
            extension_manager: Optional extension manager
            checkpoint_manager: Optional checkpoint manager
            metrics_logger: Optional metrics logger
        """
        # Model adapter
        if model_adapter is not None:
            self.model_adapter = model_adapter
        else:
            model_type = getattr(self.training_config, "model_type", "standard")
            self.model_adapter = get_model_adapter(
                model_type,
                self.model_config,
                self.training_config,
                self.device,
                self.logger
            )
        
        # Training strategy
        if training_strategy is not None:
            self.training_strategy = training_strategy
        else:
            strategy_type = getattr(self.training_config, "training_strategy", "standard")
            self.training_strategy = get_training_strategy(
                strategy_type,
                self.training_config,
                self.logger
            )
        
        # Extension manager
        if extension_manager is not None:
            self.extension_manager = extension_manager
        else:
            self.extension_manager = ExtensionManager(
                self.training_config,
                self.logger
            )
        
        # Checkpoint manager
        if checkpoint_manager is not None:
            self.checkpoint_manager = checkpoint_manager
        else:
            self.checkpoint_manager = CheckpointManager(
                self.training_config,
                self.output_dir,
                self.logger
            )
        
        # Metrics logger
        if metrics_logger is not None:
            self.metrics_logger = metrics_logger
        else:
            self.metrics_logger = MetricsLogger(
                self.output_dir,
                log_to_console=True,
                log_to_tensorboard=getattr(self.training_config, "use_tensorboard", True),
                log_to_file=True,
                logger=self.logger
            )
    
    def _get_device(self) -> torch.device:
        """
        Get device based on configuration.
        
        Returns:
            Device to use for training
        """
        from src.utils.device import get_device
        return get_device(getattr(self.training_config, "device", None))
    
    # -------------------------------------------------------------------
    # Initialization methods
    # -------------------------------------------------------------------
    
    def initialize_model(self) -> nn.Module:
        """
        Initialize the model.
        
        Returns:
            Initialized model
        """
        self.logger.info("Initializing model")
        self.model = self.model_adapter.create_model()
        self.model.to(self.device)
        
        # Set model in the adapter for consistency
        if hasattr(self.model_adapter, "set_model"):
            self.model_adapter.set_model(self.model)
        
        return self.model
    
    def initialize_tokenizer(self) -> Any:
        """
        Initialize the tokenizer.
        
        Returns:
            Initialized tokenizer
        """
        self.logger.info("Initializing tokenizer")
        self.tokenizer = self.model_adapter.create_tokenizer()
        
        # Set tokenizer in the adapter for consistency
        if hasattr(self.model_adapter, "set_tokenizer"):
            self.model_adapter.set_tokenizer(self.tokenizer)
        
        return self.tokenizer
    
    def initialize_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Initialize data loaders.
        
        Returns:
            Dictionary of data loaders
        """
        self.logger.info("Initializing dataloaders")
        
        # Import dataloader utils
        from src.data.dataloader_utils import create_dataloaders
        
        # Create dataloaders
        self.dataloaders = create_dataloaders(
            self.data_config,
            self.tokenizer,
            self.training_config.batch_size,
            getattr(self.training_config, "num_workers", 4)
        )
        
        return self.dataloaders
    
    def initialize_optimizer(self) -> Tuple[Optimizer, Any]:
        """
        Initialize optimizer and scheduler.
        
        Returns:
            Tuple of (optimizer, scheduler)
        """
        self.logger.info("Initializing optimizer and scheduler")
        
        # Determine optimizer type
        optimizer_type = getattr(self.training_config, "optimizer", "adamw")
        
        # Create optimizer using strategy
        self.optimizer = self.training_strategy.create_optimizer(
            self.model,
            optimizer_type=optimizer_type,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
        # Determine scheduler type
        scheduler_type = getattr(self.training_config, "lr_scheduler", "cosine")
        
        # Calculate training steps
        if "train" not in self.dataloaders:
            self.logger.warning("Training dataloader not initialized. Cannot calculate number of training steps.")
            num_training_steps = 1000
        else:
            train_dataloader = self.dataloaders["train"]
            steps_per_epoch = len(train_dataloader)
            num_training_steps = steps_per_epoch * self.training_config.max_epochs
        
        # Calculate number of warmup steps
        warmup_ratio = getattr(self.training_config, "warmup_ratio", 0.1)
        num_warmup_steps = int(warmup_ratio * num_training_steps)
        
        # Create scheduler using strategy
        self.scheduler = self.training_strategy.create_scheduler(
            self.optimizer,
            scheduler_type=scheduler_type,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps
        )
        
        # Initialize gradient scaler for mixed precision training if enabled
        self.scaler = None
        if getattr(self.training_config, "use_mixed_precision", True) and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        
        return self.optimizer, self.scheduler
    
    def initialize_all(self) -> None:
        """
        Initialize all components in the correct order.
        """
        self.initialize_model()
        self.initialize_tokenizer()
        self.initialize_dataloaders()
        self.initialize_optimizer()
        self._save_config()  # Save configuration for reproducibility
    
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
    
    # -------------------------------------------------------------------
    # Training methods
    # -------------------------------------------------------------------
    
    def train(self) -> Dict[str, float]:
        """
        Train the model.
        
        Returns:
            Dictionary of final metrics
        """
        self.logger.info("Starting training")
        self.training_start_time = time.time()
        
        # Initialize components if needed
        if self.model is None:
            self.initialize_all()
        
        # Get training parameters
        max_epochs = self.training_config.max_epochs
        accumulation_steps = getattr(self.training_config, "accumulation_steps", 1)
        logging_steps = getattr(self.training_config, "logging_steps", 10)
        
        # No direct "training start" hook in ExtensionManager
        self.logger.info("Starting training")
        
        # Train for specified number of epochs
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            self.logger.info(f"Starting epoch {epoch + 1}/{max_epochs}")
            
            # Execute epoch start hooks
            self.extension_manager.pre_epoch(self.model, epoch)
            
            # Train for one epoch
            metrics = self._train_epoch(
                epoch=epoch,
                accumulation_steps=accumulation_steps,
                logging_steps=logging_steps
            )
            
            # Evaluate at the end of the epoch
            if "validation" in self.dataloaders:
                eval_metrics = self.evaluate("validation")
                # Add validation metrics to epoch metrics
                metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
                # Log validation metrics separately
                self.metrics_logger.log_validation_metrics(eval_metrics, epoch=epoch)
            
            # Log epoch metrics
            self.metrics_logger.log_epoch_metrics(metrics, epoch=epoch)
            
            # Save checkpoint
            if getattr(self.training_config, "save_every_epoch", True):
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    step=self.global_step
                )
            
            # Execute epoch end hooks
            self.extension_manager.post_epoch(self.model, epoch, metrics)
        
        # Training complete
        total_time = time.time() - self.training_start_time
        self.logger.info(f"Training complete. Total time: {total_time:.2f}s")
        
        # No direct training end hook, but we can log completion
        
        return metrics
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch (public method to match enhanced_trainer API).
        
        Returns:
            Dictionary of metrics for the epoch
        """
        self.logger.info(f"Training epoch {self.current_epoch + 1}")
        
        # Make sure model is in training mode
        self.model.train()
        
        # Train for one epoch using the internal implementation
        metrics = self._train_epoch(
            epoch=self.current_epoch,
            accumulation_steps=getattr(self.training_config, "accumulation_steps", 1),
            logging_steps=getattr(self.training_config, "logging_steps", 10)
        )
        
        # Return metrics in the expected format
        return {
            "train_loss": metrics["loss"],
            "epoch_time": metrics["epoch_time"]
        }
    
    def _train_epoch(
        self, 
        epoch: int, 
        accumulation_steps: int = 1,
        logging_steps: int = 10
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            accumulation_steps: Number of steps to accumulate gradients
            logging_steps: Number of steps between logging
            
        Returns:
            Dictionary of epoch metrics
        """
        if "train" not in self.dataloaders:
            raise ValueError("Training dataloader not found. Call initialize_dataloaders() first.")
        
        train_dataloader = self.dataloaders["train"]
        self.model.train()
        
        total_loss = 0.0
        epoch_step = 0
        epoch_start_time = time.time()
        
        # Set up batch metrics
        batch_metrics = {}
        
        # Training loop
        for batch_idx, batch in enumerate(train_dataloader):
            # Prepare batch
            prepared_batch = self.model_adapter.prepare_batch(batch)
            
            # Execute batch start hooks
            self.extension_manager.pre_batch(self.model, prepared_batch)
            
            # Forward pass using model adapter
            with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                outputs = self.model_adapter.forward(self.model, prepared_batch)
                loss = self.model_adapter.compute_loss(outputs, prepared_batch)
                loss = loss / accumulation_steps
            
            # Backward pass with gradient accumulation
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update metrics
            batch_loss = loss.item() * accumulation_steps
            total_loss += batch_loss
            batch_metrics["loss"] = batch_loss
            
            # Execute batch end hooks
            self.extension_manager.post_batch(
                self.model, prepared_batch, loss, self.global_step
            )
            
            # Optimizer step after accumulation or at end of dataloader
            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
                # Gradient clipping
                max_grad_norm = getattr(self.training_config, "max_grad_norm", 1.0)
                if max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Update global step
                self.global_step += 1
                epoch_step += 1
                
                # Log progress
                if epoch_step % logging_steps == 0:
                    avg_loss = total_loss / epoch_step
                    lr = self.optimizer.param_groups[0]['lr']
                    progress = batch_idx / len(train_dataloader) * 100
                    examples_per_second = (batch_idx + 1) * len(prepared_batch["input_ids"]) / (time.time() - epoch_start_time)
                    
                    self.logger.info(
                        f"Epoch: {epoch+1}/{self.training_config.max_epochs} | "
                        f"Step: {epoch_step}/{len(train_dataloader)} ({progress:.1f}%) | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.6f} | "
                        f"Examples/sec: {examples_per_second:.1f}"
                    )
                    
                    # Log to metrics logger
                    step_metrics = {
                        "loss": avg_loss,
                        "learning_rate": lr,
                        "examples_per_second": examples_per_second
                    }
                    self.metrics_logger.log_training_step(
                        step_metrics,
                        step=self.global_step
                    )
        
        # Compute epoch metrics
        epoch_loss = total_loss / epoch_step
        epoch_time = time.time() - epoch_start_time
        
        self.logger.info(
            f"Epoch {epoch+1} completed in {epoch_time:.2f}s | "
            f"Loss: {epoch_loss:.4f}"
        )
        
        return {
            "loss": epoch_loss,
            "epoch_time": epoch_time
        }
    
    # -------------------------------------------------------------------
    # Evaluation methods
    # -------------------------------------------------------------------
    
    def evaluate(self, split: str = "validation") -> Dict[str, float]:
        """
        Evaluate the model on the specified data split.
        
        Args:
            split: Data split to evaluate on ('validation', 'test', etc.)
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info(f"Evaluating on {split} split")
        
        if split not in self.dataloaders:
            raise ValueError(f"Data split '{split}' not found in dataloaders.")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get dataloader
        eval_dataloader = self.dataloaders[split]
        
        # Execute evaluation start hooks
        self.extension_manager.pre_validation(self.model)
        
        # Evaluation loop
        total_loss = 0.0
        total_samples = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Prepare batch
                prepared_batch = self.model_adapter.prepare_batch(batch)
                batch_size = len(prepared_batch["input_ids"])
                total_samples += batch_size
                
                # Forward pass
                outputs = self.model_adapter.forward(self.model, prepared_batch)
                loss = self.model_adapter.compute_loss(outputs, prepared_batch)
                
                # Update metrics
                total_loss += loss.item() * batch_size
                
                # Store outputs for metric calculation
                if "labels" in prepared_batch:
                    all_outputs.append(outputs)
                    all_labels.append(prepared_batch["labels"])
        
        # Calculate metrics
        metrics = {"loss": total_loss / total_samples}
        
        # Calculate additional metrics if possible
        if all_outputs and all_labels:
            additional_metrics = self._calculate_metrics(all_outputs, all_labels)
            metrics.update(additional_metrics)
        
        # Log evaluation metrics
        self.logger.info(f"Evaluation results ({split}): {metrics}")
        
        # Execute evaluation end hooks
        self.extension_manager.post_validation(self.model, metrics)
        
        return metrics
    
    def _calculate_metrics(self, outputs: List[Any], labels: List[torch.Tensor]) -> Dict[str, float]:
        """
        Calculate additional metrics from outputs and labels.
        
        Args:
            outputs: Model outputs
            labels: Ground truth labels
            
        Returns:
            Dictionary of additional metrics
        """
        # This is a placeholder - specific implementations can override this
        # to add more sophisticated metrics calculations
        return {}

    # -------------------------------------------------------------------
    # Prediction methods
    # -------------------------------------------------------------------
    
    def predict(self, inputs: Union[str, List[str], Dict[str, torch.Tensor]]) -> Any:
        """
        Generate predictions for the given inputs.
        
        Args:
            inputs: Input text or tensors
            
        Returns:
            Model predictions
        """
        # Ensure model is initialized and in eval mode
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        self.model.eval()
        
        # Convert inputs to tensor format if needed
        if isinstance(inputs, str) or (isinstance(inputs, list) and isinstance(inputs[0], str)):
            if self.tokenizer is None:
                raise ValueError("Tokenizer not initialized. Call initialize_tokenizer() first.")
            
            # Tokenize inputs
            encoded_inputs = self.tokenizer(
                inputs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
        else:
            # Assume inputs are already properly formatted
            encoded_inputs = inputs
            if isinstance(encoded_inputs, dict):
                encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
        
        # Generate predictions
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
        
        return outputs
    
    # -------------------------------------------------------------------
    # Checkpoint methods
    # -------------------------------------------------------------------
    
    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint (default: auto-generated in output_dir)
            
        Returns:
            Path where checkpoint was saved
        """
        return self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            step=self.global_step
        )
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        state = self.checkpoint_manager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            path=path
        )
        
        if state and "epoch" in state:
            self.current_epoch = state["epoch"] + 1
        
        if state and "global_step" in state:
            self.global_step = state["global_step"]
        
        self.logger.info(f"Loaded checkpoint from {path}")