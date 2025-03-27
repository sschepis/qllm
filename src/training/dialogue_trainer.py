"""
Dialogue trainer for the Quantum Resonance Language Model.

This module provides an implementation of the BaseTrainer specifically
designed for training QLLM on dialogue datasets.
"""

import os
import torch
import logging
import numpy as np
import math
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.config.data_config import DataConfig
from src.training.base_trainer import BaseTrainer
from src.model.semantic_resonance_model import SemanticResonanceModel
from src.data.dataloaders import get_appropriate_dataloaders


class DialogueTrainer(BaseTrainer):
    """Dialogue trainer implementation for QLLM."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        output_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the dialogue trainer.
        
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
        
        # NaN prevention: Set tighter gradient clipping
        self.max_grad_norm = getattr(self.training_config, "max_grad_norm", 1.0)
        
        # NaN prevention: More robust learning rate
        self.min_learning_rate = 1e-8
        
        # Use GradScaler only with CUDA - Fix deprecation warning
        use_grad_scaler = self.training_config.use_mixed_precision and self.device.type == 'cuda'
        if use_grad_scaler:
            self.scaler = torch.amp.GradScaler(device_type='cuda')
        else:
            self.scaler = torch.amp.GradScaler(enabled=False)
    
    def initialize_model(self) -> None:
        """Initialize the model."""
        self.logger.info("Initializing dialogue model...")
        
        # Create model instance with any dialogue-specific configurations
        self.model = SemanticResonanceModel(self.model_config)
        self.model.to(self.device)
        
        # Log model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model size: {total_params:,} parameters ({trainable_params:,} trainable)")
    
    def initialize_tokenizer(self) -> None:
        """Initialize the tokenizer."""
        self.logger.info(f"Loading tokenizer: {self.data_config.tokenizer_name}")
        
        # Load tokenizer with dialogue-specific configurations
        self.tokenizer = AutoTokenizer.from_pretrained(self.data_config.tokenizer_name)
        
        # Set default pad token if not set
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Ensure we have the necessary special tokens for dialogue
        special_tokens = {
            "additional_special_tokens": []
        }
        
        if not self.tokenizer.sep_token:
            special_tokens["sep_token"] = "<sep>"
            
        # Add speaker tokens if not present
        required_tokens = ["<system>", "<user>", "<assistant>"]
        missing_tokens = [token for token in required_tokens if token not in self.tokenizer.get_vocab()]
        
        if missing_tokens:
            self.logger.info(f"Adding missing tokens to tokenizer: {missing_tokens}")
            special_tokens["additional_special_tokens"].extend(missing_tokens)
            
        # Resize tokenizer if needed
        if special_tokens["additional_special_tokens"] or "sep_token" in special_tokens:
            self.tokenizer.add_special_tokens(special_tokens)
            
            # Update vocab size in model config
            self.model_config.vocab_size = len(self.tokenizer)
            
            # Resize model's token embeddings
            if self.model is not None and hasattr(self.model, "resize_token_embeddings"):
                self.model.resize_token_embeddings(len(self.tokenizer))
    
    def initialize_dataloaders(self) -> None:
        """Initialize the dataloaders using appropriate dataset loader."""
        self.logger.info("Loading dialogue datasets...")
        
        # Check if we need to set dataset_name to ensure dialogue-specific processing
        if not hasattr(self.data_config, "dataset_name") or not self.data_config.dataset_name:
            self.logger.info("No dataset name specified, using 'daily_dialog' as default for dialogue training")
            self.data_config.dataset_name = "daily_dialog"
        
        # Ensure system prompt is set for dialogue
        if not hasattr(self.data_config, "system_prompt") or not self.data_config.system_prompt:
            self.logger.info("No system prompt specified, using default")
            setattr(self.data_config, "system_prompt", "You are a helpful assistant.")
        
        # Use the appropriate dataloader based on configuration
        # Explicitly pass batch_size and eval_batch_size from training_config
        self.dataloaders = get_appropriate_dataloaders(
            data_config=self.data_config,
            tokenizer=self.tokenizer,
            batch_size=self.training_config.batch_size,
            eval_batch_size=getattr(self.training_config, "eval_batch_size", self.training_config.batch_size),
            num_workers=getattr(self.data_config, "preprocessing_num_workers", 4)
        )
        
        # Log dataset sizes
        self.logger.info(f"Dataset sizes:")
        for split, dataloader in self.dataloaders.items():
            self.logger.info(f"  {split}: {len(dataloader.dataset)} examples, {len(dataloader)} batches")
    
    def train_epoch(self) -> Dict[str, Any]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of epoch statistics
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_step = 0
        nan_count = 0
        
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
                
                # Create input dict for model - with error handling
                try:
                    if len(batch) >= 3:
                        input_ids, attention_mask, labels = batch[:3]
                        batch = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "labels": labels
                        }
                    elif len(batch) == 2:
                        # Only input_ids and attention_mask available
                        input_ids, attention_mask = batch
                        batch = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask
                        }
                    else:
                        self.logger.warning(f"Unexpected batch format with {len(batch)} elements. Skipping.")
                        continue
                except Exception as e:
                    self.logger.warning(f"Error unpacking batch: {e}. Skipping.")
                    continue
            
            # Ensure required keys are present
            if "input_ids" not in batch or "attention_mask" not in batch:
                self.logger.warning("Batch is missing required keys. Skipping.")
                continue
                
            # If no labels are present, add dummy labels using input_ids
            if "labels" not in batch:
                self.logger.warning("No labels found in batch. Using input_ids as labels for language modeling.")
                batch["labels"] = batch["input_ids"].clone()
            
            # NaN prevention: Check if all labels are -100
            if (batch["labels"] == -100).all():
                self.logger.warning("All labels are -100! Skipping batch to avoid NaN loss.")
                continue
                
            # Forward pass with mixed precision - use proper device type
            with torch.amp.autocast(device_type=self.device.type, enabled=self.training_config.use_mixed_precision):
                outputs = self.model(**batch, return_dict=True)
                loss = outputs["loss"]
                
                # NaN prevention: Check for NaN loss and skip this batch if needed
                if torch.isnan(loss).item() or torch.isinf(loss).item():
                    nan_count += 1
                    self.logger.warning(f"Detected NaN/Inf loss at step {self.global_step}. Skipping batch.")
                    
                    # Skip this batch completely
                    if nan_count > 5:  # Too many NaNs in a row, alert the user
                        self.logger.error(f"Too many NaN losses ({nan_count} consecutive). Check your data and model configuration.")
                    continue
                
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
                # Clip gradients - NaN prevention: Always clip gradients
                if self.device.type == 'cuda' and self.training_config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                # NaN prevention: Check for NaN gradients
                valid_gradients = True
                for param in self.model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            valid_gradients = False
                            self.logger.warning(f"NaN/Inf gradients detected at step {self.global_step}. Skipping update.")
                            break
                
                if valid_gradients:
                    # NaN prevention: Always clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
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
            
            # Reset NaN counter since we had a successful batch
            nan_count = 0
            
            # Update statistics
            loss_value = loss.item() * self.training_config.accumulation_steps
            epoch_loss += loss_value
            epoch_step += 1
            self.global_step += 1
            
            # Update progress bar
            lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.training_config.learning_rate
            lr = max(lr, self.min_learning_rate)  # NaN prevention: enforce minimum learning rate
            
            train_iterator.set_postfix(
                loss=loss_value,
                avg_loss=epoch_loss/epoch_step,
                lr=lr
            )
            
            # Log metrics
            if self.global_step % self.training_config.logging_steps == 0:
                self.logger.info(
                    f"Step {self.global_step}, Loss: {loss_value:.4f}, "
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
        if epoch_step > 0:
            avg_epoch_loss = epoch_loss / epoch_step
        else:
            avg_epoch_loss = float('nan')
            self.logger.warning("No valid steps in epoch! All batches were skipped.")
        
        return {
            "train_loss": avg_epoch_loss,
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
                
                # NaN prevention: Skip batch if all labels are -100
                if (batch["labels"] == -100).all():
                    continue
                
                # Forward pass
                outputs = self.model(**batch, return_dict=True)
                loss = outputs["loss"]
                
                # Skip NaN losses
                if torch.isnan(loss).item() or torch.isinf(loss).item():
                    self.logger.warning(f"Skipping NaN/Inf loss in evaluation")
                    continue
                
                # Update statistics
                total_loss += loss.item()
                total_steps += 1
        
        # Calculate average loss
        if total_steps > 0:
            avg_loss = total_loss / total_steps
        else:
            self.logger.warning("No valid steps in evaluation! All batches were skipped.")
            avg_loss = float('inf')
        
        # Calculate perplexity (with protection against overflow)
        if avg_loss > 20:
            # Very high loss would cause overflow in exp
            perplexity = float('inf')
        else:
            try:
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
            except OverflowError:
                perplexity = float('inf')
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }