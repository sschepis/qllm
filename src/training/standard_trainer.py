"""
Standard trainer for the Quantum Resonance Language Model.

This module provides an implementation of the BaseTrainer specifically
designed for standard language model training.
"""

import os
import torch
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.config.data_config import DataConfig
from src.training.base_trainer import BaseTrainer
from src.model.semantic_resonance_model import SemanticResonanceModel


class StandardTrainer(BaseTrainer):
    """Standard trainer implementation for QLLM."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        output_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the standard trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            data_config: Data configuration
            output_dir: Directory for outputs (default: from training_config)
            logger: Logger instance
        """
        super().__init__(
            model_config=model_config,
            training_config=training_config,
            data_config=data_config,
            output_dir=output_dir,
            logger=logger
        )
        
        # Initialize state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        # Use GradScaler only with CUDA - Fix deprecation warning
        use_grad_scaler = self.training_config.use_mixed_precision and self.device.type == 'cuda'
        if use_grad_scaler:
            self.scaler = torch.amp.GradScaler(device_type='cuda')
        else:
            self.scaler = torch.amp.GradScaler(enabled=False)
    
    def initialize_model(self) -> None:
        """Initialize the model."""
        self.logger.info("Initializing model...")
        
        # Create model instance
        self.model = SemanticResonanceModel(self.model_config)
        self.model.to(self.device)
        
        # Log model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model size: {total_params:,} parameters ({trainable_params:,} trainable)")
    
    def initialize_tokenizer(self) -> None:
        """Initialize the tokenizer."""
        self.logger.info(f"Loading tokenizer: {self.data_config.tokenizer_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.data_config.tokenizer_name)
        
        # Set default pad token if not set
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Update vocab size in model config
        self.model_config.vocab_size = len(self.tokenizer)
        
        # Resize model's token embeddings
        if self.model is not None and hasattr(self.model, "resize_token_embeddings"):
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def initialize_dataloaders(self) -> None:
        """Initialize the dataloaders."""
        self.logger.info("Loading datasets...")
        
        # Use the common dataloaders implementation from BaseTrainer
        super().initialize_dataloaders()
    
    def train_epoch(self) -> Dict[str, Any]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of epoch statistics
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_step = 0
        
        # Create progress bar
        train_iterator = tqdm(
            self.dataloaders["train"],
            desc=f"Epoch {self.current_epoch+1}/{self.training_config.max_epochs}",
            leave=True
        )
        
        for batch_idx, batch in enumerate(train_iterator):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            else:
                # Handle tensor datasets differently
                batch = tuple(t.to(self.device) for t in batch)
                
                # Create input dict for model
                input_ids, attention_mask, labels = batch
                batch = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }
            
            # Forward pass with mixed precision - use proper device type
            with torch.amp.autocast(device_type=self.device.type, enabled=self.training_config.use_mixed_precision):
                outputs = self.model(**batch, return_dict=True)
                loss = outputs["loss"]
                
                # Handle gradient accumulation
                if self.training_config.accumulation_steps > 1:
                    loss = loss / self.training_config.accumulation_steps
            
            # Backward pass
            if self.device.type == 'cuda' and self.training_config.use_mixed_precision:
                # Use scaler only on CUDA
                self.scaler.scale(loss).backward()
            else:
                # Regular backward on MPS/CPU
                loss.backward()
            
            # Update weights if accumulation complete
            if (batch_idx + 1) % self.training_config.accumulation_steps == 0:
                # Clip gradients
                if self.device.type == 'cuda' and self.training_config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.max_grad_norm
                )
                
                # Update weights
                if self.device.type == 'cuda' and self.training_config.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                self.optimizer.zero_grad()
                
                # Update learning rate scheduler
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            
            # Update statistics
            epoch_loss += loss.item() * self.training_config.accumulation_steps
            epoch_step += 1
            self.global_step += 1
            
            # Update progress bar
            lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.training_config.learning_rate
            train_iterator.set_postfix(
                loss=loss.item() * self.training_config.accumulation_steps,
                avg_loss=epoch_loss/epoch_step,
                lr=lr
            )
            
            # Log metrics
            if self.global_step % self.training_config.logging_steps == 0:
                self.logger.info(
                    f"Step {self.global_step}, Loss: {loss.item() * self.training_config.accumulation_steps:.4f}, "
                    f"Average Loss: {epoch_loss/epoch_step:.4f}, LR: {lr:.6f}"
                )
            
            # Evaluate and save checkpoint
            if self.training_config.eval_steps > 0 and self.global_step % self.training_config.eval_steps == 0:
                val_metrics = self.evaluate()
                
                # Log validation metrics
                self.logger.info(
                    f"Step {self.global_step}, Validation Loss: {val_metrics['loss']:.4f}, "
                    f"Validation Perplexity: {val_metrics['perplexity']:.2f}"
                )
                
                # Check for improvement
                if val_metrics["loss"] < self.best_val_loss:
                    improvement = (self.best_val_loss - val_metrics["loss"]) / self.best_val_loss
                    self.logger.info(f"Validation loss improved by {improvement:.2%}")
                    
                    self.best_val_loss = val_metrics["loss"]
                    
                    # Save best model
                    best_model_path = os.path.join(self.output_dir, "best_model.pt")
                    self.save_checkpoint(best_model_path)
                
                # Switch back to training mode
                self.model.train()
            
            # Save checkpoint
            if self.training_config.save_steps > 0 and self.global_step % self.training_config.save_steps == 0:
                checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"step_{self.global_step}.pt")
                self.save_checkpoint(checkpoint_path)
        
        # Calculate average loss for the epoch
        epoch_loss = epoch_loss / epoch_step
        
        return {
            "train_loss": epoch_loss,
            "steps": epoch_step,
            "learning_rate": lr
        }
    
    def evaluate(self, split: str = "validation") -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            split: Dataset split to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        if split not in self.dataloaders:
            self.logger.warning(f"Split '{split}' not found in dataloaders. Available splits: {list(self.dataloaders.keys())}")
            return {"loss": float("inf"), "perplexity": float("inf")}
        
        self.model.eval()
        
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders[split], desc=f"Evaluating ({split})"):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                else:
                    # Handle tensor datasets differently
                    batch = tuple(t.to(self.device) for t in batch)
                    
                    # Create input dict for model
                    input_ids, attention_mask, labels = batch
                    batch = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    }
                
                # Forward pass
                outputs = self.model(**batch, return_dict=True)
                loss = outputs["loss"]
                
                # Update statistics
                total_loss += loss.item()
                total_steps += 1
        
        # Calculate average loss
        avg_loss = total_loss / total_steps
        
        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }