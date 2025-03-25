"""
Training components for the Quantum Resonance Language Model.
Provides a unified trainer class for model training with checkpoints,
logging, and evaluation.
"""

import os
import time
import math
import logging
import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.utils.device import get_device, move_to_device
from src.data.dataloaders import compute_perplexity

# Get logger
logger = logging.getLogger("quantum_resonance")


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with linear warmup and decay.
    
    Args:
        optimizer: Optimizer to use
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: Last epoch to resume from
        
    Returns:
        LambdaLR: Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with cosine annealing and warmup.
    
    Args:
        optimizer: Optimizer to use
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles
        last_epoch: Last epoch to resume from
        
    Returns:
        LambdaLR: Learning rate scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class EarlyStopping:
    """Early stopping implementation."""
    
    def __init__(
        self,
        patience: int = 3,
        threshold: float = 0.01,
        mode: str = "min"
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement before stopping
            threshold: Minimum change to qualify as improvement
            mode: 'min' for metrics like loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.threshold = threshold
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # Set comparator function
        self.improvement = (
            lambda score, best: score < best - threshold 
            if mode == "min" 
            else score > best + threshold
        )
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.improvement(score, self.best_score):
            # Score improved
            self.best_score = score
            self.counter = 0
        else:
            # Score did not improve
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """Trainer for the Quantum Resonance Language Model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        device: Optional[Union[str, torch.device]] = None,
        output_dir: str = "runs/quantum_resonance",
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LambdaLR] = None,
        max_epochs: int = 10,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        eval_steps: int = 500,
        save_steps: int = 1000,
        logging_steps: int = 10,
        save_total_limit: int = 3,
        save_every_epoch: bool = True,
        use_amp: bool = True,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.01,
        seed: int = 42
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Test data loader
            device: Device to use
            output_dir: Directory to save outputs
            optimizer: Optimizer to use (default: AdamW)
            scheduler: Learning rate scheduler
            max_epochs: Maximum number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay factor
            warmup_steps: Number of warmup steps
            gradient_accumulation_steps: Steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            eval_steps: Steps between evaluations
            save_steps: Steps between model saves
            logging_steps: Steps between logging
            save_total_limit: Maximum number of checkpoints to keep
            save_every_epoch: Whether to save after each epoch
            use_amp: Whether to use automatic mixed precision
            early_stopping_patience: Patience for early stopping
            early_stopping_threshold: Threshold for early stopping
            seed: Random seed
        """
        # Setup device
        self.device = get_device(device)
        self.model = model.to(self.device)
        
        # Data loaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        # Training settings
        self.output_dir = output_dir
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.save_total_limit = save_total_limit
        self.save_every_epoch = save_every_epoch
        self.use_amp = use_amp and torch.cuda.is_available()
        self.seed = seed
        
        # Set up checkpoints directory
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Create optimizer if not provided
        if optimizer is None:
            # Prepare parameters
            param_groups = []
            no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "layer_norm.weight"]
            param_groups = [
                {
                    "params": [p for n, p in model.named_parameters() 
                              if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() 
                              if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(param_groups, lr=learning_rate)
        else:
            self.optimizer = optimizer
        
        # Calculate number of training steps
        self.num_update_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
        self.total_training_steps = self.num_update_steps_per_epoch * max_epochs
        
        # Create scheduler if not provided
        if scheduler is None:
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_training_steps
            )
        else:
            self.scheduler = scheduler
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            threshold=early_stopping_threshold,
            mode="min"
        )
        
        # State tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.metrics_history = {"train": [], "val": [], "test": []}
        
        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Log initialization
        self._log_initialization()
    
    def _log_initialization(self) -> None:
        """Log initialization details."""
        logger.info(f"Trainer initialized with:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Mixed precision: {self.use_amp}")
        logger.info(f"  Max epochs: {self.max_epochs}")
        logger.info(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps: {self.total_training_steps}")
        logger.info(f"  Evaluation interval: {self.eval_steps} steps")
        logger.info(f"  Save interval: {self.save_steps} steps")
        logger.info(f"  Early stopping patience: {self.early_stopping.patience} epochs")
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns:
            Dict[str, Any]: Training statistics and results
        """
        logger.info("Starting training...")
        
        # Training loop
        self.model.train()
        start_time = time.time()
        
        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Training epoch
            epoch_loss = self._train_epoch()
            
            # Validation
            val_loss = self._validate()
            
            # Log epoch metrics
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{self.max_epochs} completed in {epoch_time:.2f}s. "
                        f"Train loss: {epoch_loss:.4f}, Val loss: {val_loss:.4f}, "
                        f"Val PPL: {compute_perplexity(val_loss):.2f}")
            
            # Save checkpoint if requested
            if self.save_every_epoch:
                self._save_checkpoint(epoch, epoch_loss)
            
            # Check early stopping
            if self.early_stopping(val_loss):
                logger.info("Early stopping triggered. Ending training.")
                break
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                self._save_checkpoint(epoch, epoch_loss, best_model_path)
                logger.info(f"New best model saved with val loss: {val_loss:.4f}")
        
        # Final evaluation
        test_results = None
        if self.test_dataloader:
            test_results = self._evaluate(self.test_dataloader, "test")
            logger.info(f"Test perplexity: {compute_perplexity(test_results['loss']):.2f}")
        
        # Compute training time
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.2f} minutes")
        
        # Return training statistics
        stats = {
            "best_val_loss": self.best_val_loss,
            "val_perplexity": compute_perplexity(self.best_val_loss),
            "total_steps": self.global_step,
            "epochs_completed": self.epoch + 1,
            "training_time": total_time
        }
        
        if test_results:
            stats["test_loss"] = test_results["loss"]
            stats["test_perplexity"] = compute_perplexity(test_results["loss"])
        
        return stats
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            float: Average epoch loss
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        
        # Progress bar
        progress_bar = tqdm(
            total=len(self.train_dataloader),
            desc=f"Epoch {self.epoch+1}/{self.max_epochs}",
            disable=logger.level > logging.INFO
        )
        
        for step, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = move_to_device(batch, self.device)
            
            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass with gradient accumulation
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update statistics
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            epoch_steps += 1
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": loss.item() * self.gradient_accumulation_steps,
                "lr": self.scheduler.get_last_lr()[0]
            })
            
            # Log training progress
            if self.global_step % self.logging_steps == 0:
                lr = self.scheduler.get_last_lr()[0]
                logger.info(f"Epoch {self.epoch+1}, Step {step+1}/{len(self.train_dataloader)}, "
                           f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f}, "
                           f"LR: {lr:.6f}")
            
            # Optimizer step on gradient accumulation boundaries
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Update weights
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Update learning rate
                self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Update step count
                self.global_step += 1
                
                # Validation at specified intervals
                if self.val_dataloader and self.global_step % self.eval_steps == 0:
                    val_loss = self._validate()
                    logger.info(f"Validation at step {self.global_step}: Loss {val_loss:.4f}, "
                               f"PPL: {compute_perplexity(val_loss):.2f}")
                    self.model.train()
                
                # Save checkpoint at specified intervals
                if self.global_step % self.save_steps == 0:
                    self._save_checkpoint(self.epoch, epoch_loss / epoch_steps)
        
        # Close progress bar
        progress_bar.close()
        
        # Compute average loss
        avg_epoch_loss = epoch_loss / epoch_steps
        
        return avg_epoch_loss
    
    def _validate(self) -> float:
        """
        Evaluate on the validation set.
        
        Returns:
            float: Validation loss
        """
        if not self.val_dataloader:
            return 0.0
        
        metrics = self._evaluate(self.val_dataloader, "val")
        self.metrics_history["val"].append({
            "step": self.global_step,
            "epoch": self.epoch,
            "loss": metrics["loss"],
            "perplexity": metrics["perplexity"]
        })
        
        return metrics["loss"]
    
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
            Dict[str, float]: Evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating on {split}", disable=logger.level > logging.INFO):
                # Move batch to device
                batch = move_to_device(batch, self.device)
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(**batch)
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                total_loss += loss.item()
                num_batches += 1
        
        # Compute average loss
        avg_loss = total_loss / max(num_batches, 1)
        
        # Compute additional metrics
        metrics = {
            "loss": avg_loss,
            "perplexity": compute_perplexity(avg_loss)
        }
        
        return metrics
    
    def _save_checkpoint(
        self,
        epoch: int,
        loss: float,
        path: Optional[str] = None
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            epoch: Current epoch
            loss: Current loss
            path: Path to save checkpoint to (default: auto-generated)
            
        Returns:
            str: Path to saved checkpoint
        """
        # Use auto-generated path if not specified
        if path is None:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        
        # Get model state dict
        model_state = self.model.state_dict()
        
        # Save checkpoint
        try:
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                loss=loss,
                output_dir=os.path.dirname(path),
                filename=os.path.basename(path)
            )
            logger.info(f"Checkpoint saved to {path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
        
        # Manage number of saved checkpoints
        if self.save_total_limit > 0:
            self._cleanup_checkpoints()
        
        return path
    
    def _cleanup_checkpoints(self) -> None:
        """
        Remove old checkpoints to stay within save_total_limit.
        Keeps the latest checkpoints, plus any checkpoint named 'best_model.pt'.
        """
        if self.save_total_limit <= 0:
            return
        
        # Get all checkpoint files
        import re
        import glob
        
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt"))
        
        # Skip if we don't have too many checkpoints
        if len(checkpoint_files) <= self.save_total_limit:
            return
        
        # Sort by epoch number
        def get_epoch(filename):
            match = re.search(r"checkpoint_epoch_(\d+)\.pt", filename)
            return int(match.group(1)) if match else 0
        
        sorted_checkpoints = sorted(checkpoint_files, key=get_epoch)
        
        # Delete oldest checkpoints
        checkpoints_to_delete = sorted_checkpoints[:-(self.save_total_limit)]
        
        for checkpoint in checkpoints_to_delete:
            try:
                os.remove(checkpoint)
                logger.info(f"Removed old checkpoint: {checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            bool: Whether loading was successful
        """
        try:
            metadata = load_checkpoint(
                model=self.model,
                checkpoint_path=checkpoint_path,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                map_location=self.device
            )
            
            # Update trainer state
            if "epoch" in metadata:
                self.epoch = metadata["epoch"]
            
            # Update global step based on epoch
            self.global_step = self.epoch * self.num_update_steps_per_epoch
            
            # Update best validation loss
            if "metrics" in metadata and "val_loss" in metadata["metrics"]:
                self.best_val_loss = metadata["metrics"]["val_loss"]
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            logger.info(f"Resuming from epoch {self.epoch+1}, global step {self.global_step}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def get_learning_rate(self) -> float:
        """
        Get current learning rate.
        
        Returns:
            float: Current learning rate
        """
        return self.scheduler.get_last_lr()[0]
    
    def get_metrics_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get training metrics history.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Metrics history by split
        """
        return self.metrics_history