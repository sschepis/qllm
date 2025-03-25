"""
Trainer module for Semantic Resonance Language Model.

This module provides the training infrastructure for the model, including
optimization, learning rate scheduling, evaluation, and logging.
"""

import os
import time
import math
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.tensorboard import SummaryWriter

from src.evaluation.metrics import compute_perplexity


class Trainer:
    """
    Trainer class for Semantic Resonance Language Model.
    
    This class handles the training loop, optimization, evaluation,
    checkpointing, and logging.
    """
    
    def __init__(self, model, config, train_dataloader, val_dataloader, 
                 test_dataloader=None, device=None):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            test_dataloader: DataLoader for test data (optional)
            device: Device to train on (defaults to GPU if available)
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        # Set device
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Set up optimizer
        self.optimizer = self._create_optimizer()
        
        # Set up scheduler
        self.scheduler = self._create_scheduler()
        
        # Set up tensorboard writer
        self.output_dir = getattr(config, "output_dir", "runs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "logs"))
        
        # Initialize training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Set up logger
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_optimizer(self):
        """
        Create optimizer for training.
        
        Returns:
            torch.optim.Optimizer: Optimizer
        """
        # Get optimizer parameters with weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]
        
        # Create optimizer
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=1e-8
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """
        Create learning rate scheduler.
        
        Returns:
            torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler
        """
        # Calculate number of training steps
        if hasattr(self.train_dataloader, "dataset"):
            num_training_steps = len(self.train_dataloader) * self.config.max_epochs
        else:
            # If dataloader doesn't have dataset attribute (e.g., IterableDataset)
            num_training_steps = 1000 * self.config.max_epochs
        
        # Create warmup scheduler
        warmup_steps = min(self.config.warmup_steps, num_training_steps // 10)
        
        # Linear warmup followed by cosine decay
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.1
        )
        
        # Chain schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
        
        return scheduler
    
    def save_checkpoint(self, is_best=False):
        """
        Save a checkpoint of the model and training state.
        
        Args:
            is_best (bool): Whether this is the best model so far
        """
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{self.global_step}.pt")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save as best checkpoint if needed
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved as best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint
            
        Returns:
            bool: Whether the checkpoint was successfully loaded
        """
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"Checkpoint {checkpoint_path} does not exist")
            return False
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load training state
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return True
    
    def train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            float: Average training loss
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_iter = 0
        
        # Create progress bar
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}/{self.config.max_epochs}",
            leave=True
        )
        
        # Iterate over batches
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch, return_dict=True)
            loss = outputs["loss"]
            
            # Backward pass
            if self.config.accumulation_steps > 1:
                loss = loss / self.config.accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (epoch_iter + 1) % self.config.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Increment global step
                self.global_step += 1
                
                # Log learning rate
                if self.global_step % 100 == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar("train/learning_rate", lr, self.global_step)
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # Evaluate
                if self.global_step % self.config.eval_steps == 0:
                    eval_loss, eval_ppl = self.evaluate()
                    self.model.train()  # Set back to train mode
                    
                    # Log evaluation metrics
                    self.writer.add_scalar("eval/loss", eval_loss, self.global_step)
                    self.writer.add_scalar("eval/perplexity", eval_ppl, self.global_step)
                    
                    # Save checkpoint if best model
                    if eval_loss < self.best_val_loss:
                        self.best_val_loss = eval_loss
                        self.save_checkpoint(is_best=True)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        "train_loss": f"{epoch_loss / (epoch_iter + 1):.4f}",
                        "eval_loss": f"{eval_loss:.4f}",
                        "eval_ppl": f"{eval_ppl:.2f}"
                    })
            
            # Update epoch loss and iteration counter
            epoch_loss += loss.item() * self.config.accumulation_steps
            epoch_iter += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{epoch_loss / (epoch_iter + 1):.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log training loss
            if epoch_iter % 10 == 0:
                self.writer.add_scalar(
                    "train/loss",
                    loss.item() * self.config.accumulation_steps,
                    self.global_step * self.config.accumulation_steps + epoch_iter
                )
        
        # Calculate average loss
        avg_loss = epoch_loss / epoch_iter
        
        # Log epoch statistics
        self.writer.add_scalar("train/epoch_loss", avg_loss, self.epoch + 1)
        
        return avg_loss
    
    def evaluate(self, dataloader=None):
        """
        Evaluate the model on the validation or test set.
        
        Args:
            dataloader: DataLoader to evaluate on (defaults to validation)
            
        Returns:
            float: Average evaluation loss
            float: Perplexity
        """
        if dataloader is None:
            dataloader = self.val_dataloader
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch, return_dict=True)
                loss = outputs["loss"]
                
                # Count tokens
                mask = batch["attention_mask"]
                num_tokens = mask.sum().item()
                
                # Update statistics
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Calculate average loss and perplexity
        avg_loss = total_loss / total_tokens
        perplexity = compute_perplexity(avg_loss)
        
        return avg_loss, perplexity
    
    def train(self):
        """
        Train the model for the specified number of epochs.
        
        Returns:
            dict: Training statistics
        """
        self.logger.info(f"Starting training on device: {self.device}")
        
        # Log model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model size: {total_params / 1e6:.2f}M parameters "
                        f"({trainable_params / 1e6:.2f}M trainable)")
        
        # Initialize optimizer
        self.optimizer.zero_grad()
        
        # Train for the specified number of epochs
        start_time = time.time()
        for epoch in range(self.epoch, self.config.max_epochs):
            self.epoch = epoch
            
            # Train for one epoch
            epoch_start_time = time.time()
            train_loss = self.train_epoch()
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch statistics
            self.logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs} "
                           f"- Loss: {train_loss:.4f} "
                           f"- Time: {epoch_time:.2f}s")
            
            # Evaluate on validation set
            val_loss, val_ppl = self.evaluate()
            self.logger.info(f"Validation - Loss: {val_loss:.4f} "
                           f"- Perplexity: {val_ppl:.2f}")
            
            # Save checkpoint
            self.save_checkpoint()
            
            # Check if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
        
        # Calculate total training time
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        
        # Evaluate on test set if available
        if self.test_dataloader is not None:
            self.logger.info("Evaluating on test set...")
            test_loss, test_ppl = self.evaluate(self.test_dataloader)
            self.logger.info(f"Test - Loss: {test_loss:.4f} "
                           f"- Perplexity: {test_ppl:.2f}")
            
            return {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
                "test_loss": test_loss,
                "test_perplexity": test_ppl,
                "best_val_loss": self.best_val_loss,
                "training_time": total_time
            }
        
        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
            "best_val_loss": self.best_val_loss,
            "training_time": total_time
        }