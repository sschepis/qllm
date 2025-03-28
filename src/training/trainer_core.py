"""
Core training loop implementation for the enhanced training system.

This module provides the central training loop mechanism with configurable
components, extensibility, and comprehensive lifecycle management.
"""

import os
import time
import logging
import math
from typing import Dict, List, Optional, Union, Any, Callable, Tuple, Type

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.training.model_adapters.base_adapter import ModelAdapter
from src.training.dataset_adapters.base_adapter import DatasetAdapter
from src.training.strategies.base_strategy import TrainingStrategy
from src.training.extensions.extension_manager import ExtensionManager
from src.training.optimization.optimizer import create_optimizer, apply_gradients
from src.training.optimization.lr_scheduler import create_scheduler
from src.training.optimization.grad_scaler import GradScalerManager
from src.training.optimization.memory_utils import (
    optimize_memory_for_evaluation,
    enable_gradient_checkpointing,
    print_memory_stats,
    try_batch_optimization,
    handle_cuda_oom
)
from src.training.checkpoints.checkpoint_manager import CheckpointManager
from src.training.metrics.metrics_logger import MetricsLogger
from src.training.config.training_config import EnhancedTrainingConfig
from src.utils.device import get_device


# Get logger
logger = logging.getLogger("quantum_resonance")


class TrainerCore:
    """
    Core training loop implementation for the enhanced training system.
    
    This class encapsulates the central training loop with configurable components,
    handling the orchestration of model adapters, dataset adapters, training
    strategies, extensions, optimization, and metrics tracking.
    """
    
    def __init__(
        self,
        config: EnhancedTrainingConfig,
        model_adapter: ModelAdapter,
        dataset_adapter: Optional[DatasetAdapter] = None,
        strategy: Optional[TrainingStrategy] = None,
        extension_manager: Optional[ExtensionManager] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the trainer core.
        
        Args:
            config: Enhanced training configuration
            model_adapter: Model adapter instance
            dataset_adapter: Dataset adapter instance
            strategy: Training strategy instance
            extension_manager: Extension manager instance
            metrics_logger: Metrics logger instance
            checkpoint_manager: Checkpoint manager instance
            device: Device to use for training
        """
        self.config = config
        self.model_adapter = model_adapter
        self.dataset_adapter = dataset_adapter
        self.strategy = strategy
        
        # Set device
        self.device = device or get_device(config.device)
        self.model_adapter.device = self.device
        
        # Initialize extension manager if not provided
        if extension_manager is None and getattr(config, "extensions", None) is not None:
            extension_manager = ExtensionManager(config.extensions)
        self.extension_manager = extension_manager
        
        # Initialize metrics logger if not provided
        if metrics_logger is None:
            metrics_logger = MetricsLogger(
                log_dir=config.output_dir,
                logging_steps=config.logging_steps
            )
        self.metrics_logger = metrics_logger
        
        # Initialize checkpoint manager if not provided
        if checkpoint_manager is None:
            checkpoint_dir = config.checkpointing.checkpoint_kwargs.get("checkpoint_dir", None)
            if checkpoint_dir is None:
                checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
            
            checkpoint_manager = CheckpointManager(
                config=config,
                output_dir=checkpoint_dir,
                save_total_limit=config.checkpointing.save_total_limit
            )
        self.checkpoint_manager = checkpoint_manager
        
        # Set up optimization components
        opt_config = config.optimization
        
        # Set up model, dataloaders, optimizer and scheduler
        self.model = model_adapter.get_model()
        self._setup_dataloaders()
        
        # Initialize optimizer
        self.optimizer = create_optimizer(
            model=self.model,
            config=opt_config,
            optimizer_name=opt_config.optimizer_type,
            learning_rate=opt_config.learning_rate,
            weight_decay=opt_config.weight_decay,
            **opt_config.optimizer_kwargs
        )
        
        # Calculate number of training steps
        if self.train_dataloader is not None:
            self.num_update_steps_per_epoch = len(self.train_dataloader) // opt_config.gradient_accumulation_steps
            self.total_training_steps = self.num_update_steps_per_epoch * config.max_epochs
        else:
            self.num_update_steps_per_epoch = 0
            self.total_training_steps = 0
        
        # Initialize scheduler
        self.scheduler = create_scheduler(
            optimizer=self.optimizer,
            config=opt_config,
            scheduler_name=opt_config.lr_scheduler_type,
            num_training_steps=self.total_training_steps,
            **opt_config.scheduler_kwargs
        )
        
        # Initialize gradient scaler for mixed precision
        self.grad_scaler = GradScalerManager(
            config=opt_config,
            device=self.device,
            enabled=opt_config.use_amp
        )
        
        # Initialize early stopping
        self._setup_early_stopping()
        
        # State tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_start_time = 0
        self.is_training = False
        
        # Set random seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        # Apply memory optimizations if enabled
        self._setup_memory_optimizations()
        
        # Log initialization
        self._log_initialization()
    
    def _setup_dataloaders(self) -> None:
        """
        Set up training, validation, and test dataloaders.
        """
        if self.dataset_adapter is None:
            self.train_dataloader = None
            self.val_dataloader = None
            self.test_dataloader = None
            return
        
        # Get batch sizes from config
        batch_size = self.config.batch_size
        eval_batch_size = getattr(self.config, "eval_batch_size", batch_size)
        
        # Create dataloaders
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.dataset_adapter.create_dataloaders(
            train_batch_size=batch_size,
            eval_batch_size=eval_batch_size
        )
        
        if self.train_dataloader is None:
            logger.warning("No training dataloader available")
    
    def _setup_early_stopping(self) -> None:
        """
        Set up early stopping configuration.
        """
        # Get early stopping configuration
        patience = getattr(self.config, "early_stopping_patience", 3)
        threshold = getattr(self.config, "early_stopping_threshold", 0.01)
        
        # Initialize early stopping state
        self.early_stopping_patience = patience
        self.early_stopping_threshold = threshold
        self.early_stopping_counter = 0
        self.early_stopping_best_score = None
        self.early_stopping_triggered = False
    
    def _setup_memory_optimizations(self) -> None:
        """
        Setup memory optimizations based on configuration.
        """
        # Get optimization config
        opt_config = self.config.optimization
        
        # Apply memory optimizations
        if hasattr(opt_config, "use_gradient_checkpointing") and opt_config.use_gradient_checkpointing:
            logger.info("Enabling gradient checkpointing for memory efficiency")
            enable_gradient_checkpointing(self.model)
        
        # Log memory state at initialization
        if torch.cuda.is_available():
            logger.info("Initial memory state:")
            print_memory_stats()
            
            # Configure PyTorch for memory efficiency if needed
            if hasattr(opt_config, "auto_handle_oom") and opt_config.auto_handle_oom:
                # Set environment variables if not already set
                import os
                if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                    logger.info("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for better memory management")
    
    def _log_initialization(self) -> None:
        """
        Log initialization details.
        """
        # Log basic information
        logger.info(f"TrainerCore initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Mixed precision: {self.grad_scaler.is_enabled}")
        logger.info(f"  Max epochs: {self.config.max_epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        
        # Log dataloader information
        if self.train_dataloader is not None:
            logger.info(f"  Training samples: {len(self.train_dataloader.dataset)}")
            logger.info(f"  Training steps per epoch: {self.num_update_steps_per_epoch}")
            logger.info(f"  Total training steps: {self.total_training_steps}")
        
        if self.val_dataloader is not None:
            logger.info(f"  Validation samples: {len(self.val_dataloader.dataset)}")
        
        # Log optimizer and scheduler
        logger.info(f"  Optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f"  Learning rate: {self.config.optimization.learning_rate}")
        logger.info(f"  Scheduler: {self.scheduler.__class__.__name__}")
        
        # Log extension information
        if self.extension_manager is not None:
            # Use get_extension_names() instead of get_active_extensions()
            extensions = self.extension_manager.get_extension_names()
            if extensions:
                logger.info(f"  Active extensions: {', '.join(extensions)}")
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Dictionary of training results
        """
        # Record start time
        self.training_start_time = time.time()
        self.is_training = True
        
        logger.info("Starting training")
        
        # Run before_training hook if extensions are available
        if self.extension_manager is not None:
            self.extension_manager.run_hook("before_training", trainer=self)
        
        # Training loop
        self.model.train()
        
        try:
            for epoch in range(self.epoch, self.config.max_epochs):
                self.epoch = epoch
                epoch_start_time = time.time()
                
                # Run before_epoch hook
                if self.extension_manager is not None:
                    self.extension_manager.run_hook("before_epoch", trainer=self, epoch=epoch)
                
                # Training epoch
                epoch_metrics = self._train_epoch()
                
                # Log epoch metrics
                epoch_time = time.time() - epoch_start_time
                self.metrics_logger.log_metrics(
                    metrics=epoch_metrics,
                    step=self.global_step,
                    epoch=epoch,
                    prefix="train"
                )
                
                logger.info(f"Epoch {epoch+1}/{self.config.max_epochs} completed in {epoch_time:.2f}s")
                for key, value in epoch_metrics.items():
                    logger.info(f"  {key}: {value:.6f}")
                
                # Validation
                if self.val_dataloader is not None:
                    val_metrics = self._validate()
                    
                    # Log validation metrics
                    self.metrics_logger.log_metrics(
                        metrics=val_metrics,
                        step=self.global_step,
                        epoch=epoch,
                        prefix="val"
                    )
                    
                    # Check early stopping
                    if self._check_early_stopping(val_metrics.get("loss", float("inf"))):
                        logger.info("Early stopping triggered")
                        break
                    
                    # Save best model
                    val_loss = val_metrics.get("loss", float("inf"))
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(is_best=True)
                        logger.info(f"New best model saved with val loss: {val_loss:.6f}")
                
                # Run after_epoch hook
                if self.extension_manager is not None:
                    self.extension_manager.run_hook("after_epoch", trainer=self, epoch=epoch, metrics=epoch_metrics)
                
                # Save checkpoint if configured to do so
                save_strategy = self.config.checkpointing.save_strategy
                if save_strategy == "epoch":
                    self._save_checkpoint()
                
                # Check if training should stop early
                if self.early_stopping_triggered:
                    logger.info("Training stopped early")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
        
        finally:
            # Training complete or interrupted
            self.is_training = False
            
            # Run after_training hook
            if self.extension_manager is not None:
                self.extension_manager.run_hook("after_training", trainer=self)
            
            # Final evaluation on test set
            test_metrics = None
            if self.test_dataloader is not None:
                test_metrics = self._evaluate(self.test_dataloader, "test")
                
                # Log test metrics
                self.metrics_logger.log_metrics(
                    metrics=test_metrics,
                    step=self.global_step,
                    epoch=self.epoch,
                    prefix="test"
                )
                
                logger.info("Test results:")
                for key, value in test_metrics.items():
                    logger.info(f"  {key}: {value:.6f}")
            
            # Compute training time
            total_time = time.time() - self.training_start_time
            logger.info(f"Training completed in {total_time/60:.2f} minutes")
            
            # Return training statistics
            results = {
                "best_val_loss": self.best_val_loss,
                "total_steps": self.global_step,
                "epochs_completed": self.epoch + 1,
                "training_time": total_time
            }
            
            if test_metrics is not None:
                results["test_metrics"] = test_metrics
            
            return results
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Public interface to train the model for one epoch.
        This method is called by menu_handlers.py.
        
        Returns:
            Dictionary of training metrics for the epoch
        """
        return self._train_epoch()
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Internal implementation to train the model for one epoch.
        
        Returns:
            Dictionary of training metrics for the epoch
        """
        # Set model to training mode
        self.model.train()
        
        # Reset metrics
        epoch_metrics = {
            "loss": 0.0,
            "lr": 0.0,
            "grad_norm": 0.0
        }
        epoch_steps = 0
        
        # Progress bar
        progress_bar = tqdm(
            total=len(self.train_dataloader),
            desc=f"Epoch {self.epoch+1}/{self.config.max_epochs}",
            disable=logger.level > logging.INFO
        )
        
        # Accumulate gradients over these steps
        accumulation_steps = self.config.optimization.gradient_accumulation_steps
        
        # Training loop
        for step, batch in enumerate(self.train_dataloader):
            # Run before_step hook
            if self.extension_manager is not None:
                self.extension_manager.run_hook("before_step", trainer=self, batch=batch, step=step)
            
            # Process batch using either the strategy or model adapter
            if self.strategy is not None:
                loss, step_metrics = self.strategy.training_step(
                    model=self.model,
                    batch=batch,
                    model_adapter=self.model_adapter,
                    global_step=self.global_step
                )
            else:
                # Prepare batch using model adapter
                prepared_batch = self.model_adapter.prepare_batch(batch)
                
                # Move batch to device
                prepared_batch = self.model_adapter.move_to_device(prepared_batch)
                
                # Forward pass with AMP
                with self.grad_scaler():
                    outputs = self.model_adapter.forward(self.model, prepared_batch)
                    loss = self.model_adapter.compute_loss(outputs, prepared_batch)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
                
                # Basic step metrics
                step_metrics = {"loss": loss.item() * accumulation_steps}
            
            # Backward pass with gradient accumulation
            if self.grad_scaler.is_enabled:
                self.grad_scaler.scale_loss(loss, self.optimizer).backward()
            else:
                loss.backward()
            
            # Update metrics
            epoch_metrics["loss"] += step_metrics["loss"]
            epoch_steps += 1
            
            # Update learning rate metrics
            epoch_metrics["lr"] = self.scheduler.get_last_lr()[0]
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": step_metrics["loss"],
                "lr": epoch_metrics["lr"]
            })
            
            # Logging at specified intervals
            if self.global_step % self.config.logging_steps == 0:
                self.metrics_logger.log_metrics(
                    metrics=step_metrics,
                    step=self.global_step,
                    epoch=self.epoch,
                    prefix="train_step"
                )
            
            # Optimizer step on gradient accumulation boundaries
            if (step + 1) % accumulation_steps == 0:
                # Apply gradients with gradient norm calculation included
                grad_norm = apply_gradients(
                    optimizer=self.optimizer,
                    grad_scaler=self.grad_scaler,
                    max_grad_norm=self.config.optimization.max_grad_norm
                )
                
                # Record gradient norm for metrics
                if self.config.optimization.max_grad_norm > 0:
                    epoch_metrics["grad_norm"] += grad_norm
                
                # Update learning rate
                self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Update step count
                self.global_step += 1
                
                # Run after_optimization_step hook
                if self.extension_manager is not None:
                    self.extension_manager.run_hook(
                        "after_optimization_step",
                        trainer=self,
                        step=step,
                        metrics=step_metrics
                    )
                
                # Validation at specified intervals
                eval_steps = getattr(self.config, "eval_steps", 0)
                if eval_steps > 0 and self.val_dataloader is not None and self.global_step % eval_steps == 0:
                    val_metrics = self._validate()
                    
                    # Log validation metrics
                    self.metrics_logger.log_metrics(
                        metrics=val_metrics,
                        step=self.global_step,
                        epoch=self.epoch,
                        prefix="val"
                    )
                    
                    # Check early stopping
                    if self._check_early_stopping(val_metrics.get("loss", float("inf"))):
                        logger.info("Early stopping triggered during epoch")
                        progress_bar.close()
                        return self._compute_average_metrics(epoch_metrics, epoch_steps)
                    
                    # Save best model
                    val_loss = val_metrics.get("loss", float("inf"))
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(is_best=True)
                    
                    # Return to training mode
                    self.model.train()
                
                # Save checkpoint at specified intervals
                save_steps = self.config.checkpointing.save_steps
                if save_steps > 0 and self.global_step % save_steps == 0:
                    self._save_checkpoint()
            
            # Run after_step hook
            if self.extension_manager is not None:
                self.extension_manager.run_hook(
                    "after_step",
                    trainer=self,
                    batch=batch,
                    step=step,
                    metrics=step_metrics
                )
        
        # Close progress bar
        progress_bar.close()
        
        # Compute average metrics
        avg_metrics = self._compute_average_metrics(epoch_metrics, epoch_steps)
        
        return avg_metrics
    
    def _validate(self) -> Dict[str, float]:
        """
        Evaluate the model on the validation set.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_dataloader is None:
            return {}
        
        # Run evaluation
        metrics = self._evaluate(self.val_dataloader, "val")
        
        return metrics
    
    def _evaluate(
        self,
        dataloader: DataLoader,
        split: str = "val"
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataloader.
        
        Args:
            dataloader: Dataloader to evaluate on
            split: Data split name ("val" or "test")
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Apply memory optimizations for evaluation
        try:
            # Get batch size
            batch_size = dataloader.batch_size
            
            # Enable gradient checkpointing if available GPU memory is limited
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                if gpu_usage > 0.7:  # If GPU is already 70%+ utilized
                    logger.info("Enabling memory optimizations for evaluation")
                    enable_gradient_checkpointing(self.model)
                    print_memory_stats()
        except Exception as e:
            logger.warning(f"Error during memory optimization setup: {e}")
        
        # Run before_evaluation hook
        if self.extension_manager is not None:
            self.extension_manager.run_hook("before_evaluation", trainer=self, split=split)
        
        # Initialize metrics
        metrics = {"loss": 0.0}
        num_batches = 0
        
        # Progress bar
        progress_bar = tqdm(
            dataloader,
            desc=f"Evaluating on {split}",
            disable=logger.level > logging.INFO
        )
        
        try:
            with torch.no_grad():
                for batch in progress_bar:
                    try:
                        # Process batch using either strategy or model adapter
                        if self.strategy is not None:
                            batch_metrics = self.strategy.validation_step(
                                model=self.model,
                                batch=batch,
                                model_adapter=self.model_adapter
                            )
                        else:
                            # Prepare batch using model adapter
                            prepared_batch = self.model_adapter.prepare_batch(batch, is_train=False)
                            
                            # Move batch to device
                            prepared_batch = self.model_adapter.move_to_device(prepared_batch)
                            
                            # Forward pass with AMP to reduce memory usage
                            with torch.cuda.amp.autocast(enabled=self.grad_scaler.is_enabled):
                                outputs = self.model_adapter.forward(self.model, prepared_batch)
                                loss = self.model_adapter.compute_loss(outputs, prepared_batch)
                            
                            # Set batch metrics
                            batch_metrics = {"loss": loss.item()}
                        
                        # Update metrics
                        for key, value in batch_metrics.items():
                            if key not in metrics:
                                metrics[key] = 0.0
                            metrics[key] += value
                        
                        num_batches += 1
                        
                        # Update progress bar
                        progress_bar.set_postfix({k: v / (num_batches or 1) for k, v in metrics.items()})
                        
                    except torch.cuda.OutOfMemoryError:
                        # Handle OOM by processing batch in smaller chunks
                        torch.cuda.empty_cache()
                        logger.warning("CUDA out of memory during evaluation. Trying with smaller chunks.")
                        
                        # Try to process batch in two chunks
                        input_key = next((k for k in batch.keys() if isinstance(batch[k], torch.Tensor)
                                        and batch[k].dim() > 0), None)
                        
                        if input_key and batch[input_key].size(0) > 1:
                            # Split batch in half
                            mid = batch[input_key].size(0) // 2
                            
                            # Process first half
                            try:
                                first_half = {k: v[:mid] if isinstance(v, torch.Tensor) and v.size(0) > mid else v
                                            for k, v in batch.items()}
                                
                                batch_metrics1 = self._process_eval_batch(first_half)
                                
                                # Process second half
                                second_half = {k: v[mid:] if isinstance(v, torch.Tensor) and v.size(0) > mid else v
                                             for k, v in batch.items()}
                                
                                batch_metrics2 = self._process_eval_batch(second_half)
                                
                                # Combine metrics
                                batch_metrics = {}
                                for key in set(batch_metrics1.keys()) | set(batch_metrics2.keys()):
                                    value1 = batch_metrics1.get(key, 0)
                                    value2 = batch_metrics2.get(key, 0)
                                    batch_metrics[key] = (value1 + value2) / 2
                                
                                # Update metrics
                                for key, value in batch_metrics.items():
                                    if key not in metrics:
                                        metrics[key] = 0.0
                                    metrics[key] += value
                                
                                num_batches += 1
                                
                            except Exception as e:
                                logger.error(f"Error processing split batch: {e}")
                                # Skip this batch if we can't process it in chunks
                                continue
                        else:
                            logger.error("Cannot split batch for OOM recovery - skipping batch")
                            continue
        
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                logger.error("CUDA out of memory error during evaluation")
                print_memory_stats()
        
        # Close progress bar
        progress_bar.close()
        
        # Compute averages if we have data
        if num_batches > 0:
            avg_metrics = {k: v / num_batches for k, v in metrics.items()}
        else:
            avg_metrics = metrics
        
        # Run after_evaluation hook
        if self.extension_manager is not None:
            self.extension_manager.run_hook(
                "after_evaluation",
                trainer=self,
                split=split,
                metrics=avg_metrics
            )
        
        return avg_metrics
    
    def _process_eval_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Process a single evaluation batch with OOM handling.
        
        Args:
            batch: Input batch
            
        Returns:
            Dictionary of batch metrics
        """
        # Process batch using either strategy or model adapter
        if self.strategy is not None:
            return self.strategy.validation_step(
                model=self.model,
                batch=batch,
                model_adapter=self.model_adapter
            )
        else:
            # Prepare batch using model adapter
            prepared_batch = self.model_adapter.prepare_batch(batch, is_train=False)
            
            # Move batch to device
            prepared_batch = self.model_adapter.move_to_device(prepared_batch)
            
            # Forward pass with AMP to reduce memory usage
            with torch.cuda.amp.autocast(enabled=self.grad_scaler.is_enabled):
                outputs = self.model_adapter.forward(self.model, prepared_batch)
                loss = self.model_adapter.compute_loss(outputs, prepared_batch)
            
            # Set batch metrics
            return {"loss": loss.item()}
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            val_loss: Validation loss
            
        Returns:
            True if training should stop early
        """
        if self.early_stopping_patience <= 0:
            return False
        
        if self.early_stopping_best_score is None:
            self.early_stopping_best_score = val_loss
            return False
        
        # Check if the validation loss improved
        threshold = self.early_stopping_threshold
        if val_loss < self.early_stopping_best_score - threshold:
            # Validation loss improved
            self.early_stopping_best_score = val_loss
            self.early_stopping_counter = 0
            return False
        else:
            # Validation loss did not improve
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.early_stopping_triggered = True
                return True
            return False
    
    def _save_checkpoint(
        self,
        is_best: bool = False
    ) -> None:
        """
        Save a checkpoint.
        
        Args:
            is_best: Whether this is the best checkpoint so far
        """
        # Skip if checkpointing is disabled
        if not self.config.checkpointing.save_strategy or self.checkpoint_manager is None:
            return
        
        # Prepare metadata
        metrics = {
            "train_loss": self.metrics_logger.get_latest_metric("train", "loss"),
            "val_loss": self.metrics_logger.get_latest_metric("val", "loss"),
            "best_val_loss": self.best_val_loss
        }
        
        # Get additional metadata from extensions
        extension_data = {}
        if self.extension_manager is not None:
            extension_result = self.extension_manager.run_hook(
                "on_save_checkpoint",
                trainer=self
            )
            if extension_result:
                extension_data = extension_result
        
        # Create checkpoint path
        if is_best:
            path = os.path.join(self.checkpoint_manager.checkpoint_dir, "best_model.pt")
        else:
            path = None  # Let checkpoint manager create the path
        
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer if self.config.checkpointing.save_optimizer else None,
            scheduler=self.scheduler if self.config.checkpointing.save_scheduler else None,
            epoch=self.epoch,
            global_step=self.global_step,
            metrics=metrics if self.config.checkpointing.save_metrics else None,
            path=path,
            extension_data=extension_data
        )
    
    def load_checkpoint(
        self,
        path: Optional[str] = None
    ) -> bool:
        """
        Load a checkpoint.
        
        Args:
            path: Path to checkpoint (if None, loads the latest checkpoint)
            
        Returns:
            True if loading was successful
        """
        if self.checkpoint_manager is None:
            return False
        
        # Check if we should use the resume_from_checkpoint path
        if path is None and hasattr(self.config.checkpointing, "resume_from_checkpoint"):
            path = self.config.checkpointing.resume_from_checkpoint
        
        # Load checkpoint
        checkpoint_info = self.checkpoint_manager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer if self.config.checkpointing.save_optimizer else None,
            scheduler=self.scheduler if self.config.checkpointing.save_scheduler else None,
            path=path
        )
        
        if not checkpoint_info:
            logger.warning("No checkpoint info returned")
            return False
        
        # Update trainer state
        if "epoch" in checkpoint_info:
            self.epoch = checkpoint_info["epoch"]
        
        if "global_step" in checkpoint_info:
            self.global_step = checkpoint_info["global_step"]
        
        if "metrics" in checkpoint_info and "best_val_loss" in checkpoint_info["metrics"]:
            self.best_val_loss = checkpoint_info["metrics"]["best_val_loss"]
        
        # Process extension data
        if "extension_data" in checkpoint_info and self.extension_manager is not None:
            self.extension_manager.run_hook(
                "on_load_checkpoint",
                trainer=self,
                checkpoint_data=checkpoint_info["extension_data"]
            )
        
        logger.info(f"Checkpoint loaded from {path or 'latest'}")
        logger.info(f"Resuming from epoch {self.epoch+1}, global step {self.global_step}")
        
        return True
    
    def _compute_average_metrics(
        self,
        metrics: Dict[str, float],
        num_steps: int
    ) -> Dict[str, float]:
        """
        Compute average metrics.
        
        Args:
            metrics: Raw metrics
            num_steps: Number of steps
            
        Returns:
            Dictionary of averaged metrics
        """
        return {k: v / max(num_steps, 1) for k, v in metrics.items()}
    
    def _get_grad_norm(self) -> float:
        """
        Calculate gradient norm for all parameters.
        
        Returns:
            Gradient norm
        """
        total_norm = 0.0
        
        parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
        
        if parameters:
            for p in parameters:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            
            total_norm = math.sqrt(total_norm)
        
        return total_norm
    
    def get_current_lr(self) -> float:
        """
        Get current learning rate.
        
        Returns:
            Current learning rate
        """
        return self.scheduler.get_last_lr()[0]
    
    @property
    def output_dir(self) -> str:
        """
        Get the output directory for training artifacts.
        
        Returns:
            Output directory path
        """
        # Try to get from config
        if hasattr(self.config, "output_dir"):
            return self.config.output_dir
        
        # Try to get from checkpoint manager
        if self.checkpoint_manager and hasattr(self.checkpoint_manager, "output_dir"):
            return self.checkpoint_manager.output_dir
        
        # Try to get from metrics logger
        if self.metrics_logger and hasattr(self.metrics_logger, "log_dir"):
            return self.metrics_logger.log_dir
        
        # Default fallback
        return os.path.join("runs", "quantum_resonance")
    
    def evaluate(self, dataloader: Optional[DataLoader] = None, split: str = "val") -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: Dataloader to evaluate on (defaults to val_dataloader)
            split: Name of the split for reporting metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Use provided dataloader or fall back to val_dataloader
        if dataloader is None:
            if self.val_dataloader is None:
                logger.error("No validation dataloader available")
                return {"loss": float('inf'), "error": 1.0}
            dataloader = self.val_dataloader
        
        # Run evaluation
        try:
            return self._validate(dataloader, split)
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            # Return default metrics on error
            return {"loss": float('inf'), "error": 1.0}
    
    def get_model(self) -> nn.Module:
        """
        Get the current model.
        
        Returns:
            Current model
        """
        return self.model
    
    def get_metrics_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get training metrics history.
        
        Returns:
            Dictionary of metrics history
        """
        return self.metrics_logger.get_metrics_history()
    
    # Adapter methods to match the expected interface in menu_handlers.py
    def initialize_model(self) -> None:
        """
        Initialize the model if it hasn't been initialized yet.
        This is an adapter method to match the interface expected by menu_handlers.py.
        """
        if self.model_adapter and not self.model_adapter.get_model():
            logger.info("Initializing model through model adapter")
            self.model_adapter.create_model()
        elif not self.model_adapter:
            logger.error("Cannot initialize model: model_adapter is None")
        else:
            logger.info("Model already initialized")
    
    def initialize_tokenizer(self) -> None:
        """
        Initialize the tokenizer if it hasn't been initialized yet.
        This is an adapter method to match the interface expected by menu_handlers.py.
        """
        if self.model_adapter and not self.model_adapter.get_tokenizer():
            logger.info("Initializing tokenizer through model adapter")
            self.model_adapter.create_tokenizer()
        elif not self.model_adapter:
            logger.error("Cannot initialize tokenizer: model_adapter is None")
        else:
            logger.info("Tokenizer already initialized")
    
    def initialize_dataloaders(self) -> None:
        """
        Initialize the dataloaders if they haven't been initialized yet.
        This is an adapter method to match the interface expected by menu_handlers.py.
        """
        try:
            if self.dataset_adapter:
                logger.info("Initializing dataloaders through dataset adapter")
                
                # Get train dataloader with fallback
                try:
                    self.train_dataloader = self.dataset_adapter.get_train_dataloader()
                    if self.train_dataloader is None:
                        # Create a dummy dataloader if the real one isn't available
                        logger.warning("Train dataloader is None, creating dummy dataloader")
                        from torch.utils.data import TensorDataset, DataLoader
                        import torch
                        dummy_dataset = TensorDataset(torch.zeros(10, 5), torch.zeros(10))
                        self.train_dataloader = DataLoader(dummy_dataset, batch_size=2)
                except Exception as e:
                    logger.error(f"Error getting train dataloader: {e}")
                    # Create a dummy dataloader as fallback
                    from torch.utils.data import TensorDataset, DataLoader
                    import torch
                    dummy_dataset = TensorDataset(torch.zeros(10, 5), torch.zeros(10))
                    self.train_dataloader = DataLoader(dummy_dataset, batch_size=2)
                
                # Get validation dataloader with fallback
                try:
                    self.val_dataloader = self.dataset_adapter.get_val_dataloader()
                except Exception as e:
                    logger.error(f"Error getting val dataloader: {e}")
                    self.val_dataloader = None
                
                # For compatibility with menu_handlers.py
                self._dataloaders = {
                    "train": self.train_dataloader,
                    "val": self.val_dataloader
                }
                
                logger.info(f"Train dataloader initialized with {len(self.train_dataloader)} batches")
                if self.val_dataloader:
                    logger.info(f"Val dataloader initialized with {len(self.val_dataloader)} batches")
            else:
                logger.error("Cannot initialize dataloaders: dataset_adapter is None")
                # Create dummy dataloaders for graceful fallback
                from torch.utils.data import TensorDataset, DataLoader
                import torch
                dummy_dataset = TensorDataset(torch.zeros(10, 5), torch.zeros(10))
                self.train_dataloader = DataLoader(dummy_dataset, batch_size=2)
                self.val_dataloader = DataLoader(dummy_dataset, batch_size=2)
                
                self._dataloaders = {
                    "train": self.train_dataloader,
                    "val": self.val_dataloader
                }
        except Exception as e:
            logger.error(f"Error during dataloader initialization: {e}")
            # Provide fallback dataloaders
            from torch.utils.data import TensorDataset, DataLoader
            import torch
            dummy_dataset = TensorDataset(torch.zeros(10, 5), torch.zeros(10))
            self.train_dataloader = DataLoader(dummy_dataset, batch_size=2)
            self.val_dataloader = DataLoader(dummy_dataset, batch_size=2)
            
            self._dataloaders = {
                "train": self.train_dataloader,
                "val": self.val_dataloader
            }
    
    @property
    def dataloaders(self) -> Dict[str, DataLoader]:
        """
        Property that provides dictionary access to dataloaders.
        For compatibility with code that expects a dataloaders dictionary.
        
        Returns:
            Dictionary mapping split names to dataloaders
        """
        if not hasattr(self, '_dataloaders'):
            self._dataloaders = {}
            
            if hasattr(self, 'train_dataloader') and self.train_dataloader is not None:
                self._dataloaders["train"] = self.train_dataloader
                
            if hasattr(self, 'val_dataloader') and self.val_dataloader is not None:
                self._dataloaders["val"] = self.val_dataloader
        
        # Ensure we always have a train dataloader
        if "train" not in self._dataloaders or self._dataloaders["train"] is None:
            # Create a dummy dataloader
            logger.warning("Creating dummy train dataloader for menu_handlers compatibility")
            from torch.utils.data import TensorDataset, DataLoader
            import torch
            dummy_dataset = TensorDataset(torch.zeros(10, 5), torch.zeros(10))
            self._dataloaders["train"] = DataLoader(dummy_dataset, batch_size=2)
        
        return self._dataloaders
    
    def initialize_optimizer(self) -> None:
        """
        Initialize the optimizer and scheduler if they haven't been initialized yet.
        This is an adapter method to match the interface expected by menu_handlers.py.
        """
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            if self.model_adapter and self.model_adapter.get_model():
                logger.info("Initializing optimizer")
                self.optimizer = create_optimizer(
                    self.model_adapter.get_model(),
                    self.config
                )
                self.lr_scheduler = create_scheduler(
                    self.optimizer,
                    self.config
                )
            else:
                logger.error("Cannot initialize optimizer: model not initialized")
        else:
            logger.info("Optimizer already initialized")