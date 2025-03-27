"""
Verbose trainer for the Quantum Resonance Language Model.

This module provides an implementation of the BaseTrainer specifically
designed for training QLLM with detailed verbose output and logging.
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
from src.utils.logging import TrainingLogger


class VerboseTrainer(BaseTrainer):
    """Verbose trainer implementation for QLLM with detailed logging."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        output_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the verbose trainer.
        
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
        self.scaler = torch.cuda.amp.GradScaler(enabled=training_config.use_mixed_precision)
        
        # Create enhanced training logger
        self.training_logger = TrainingLogger(logger=self.logger, log_interval=1)
        
        # Enable verbose mode for model if possible
        self.model_config.verbose = True
        
        # Create detailed logging directory
        self.log_dir = os.path.join(self.output_dir, "detailed_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Open a detailed training log file
        self.detailed_log_file = open(os.path.join(self.log_dir, "verbose_training.log"), "w")
        self.detailed_log_file.write("=== QLLM Verbose Training Log ===\n\n")
        self.detailed_log_file.write(f"Model Configuration:\n{str(model_config)}\n\n")
        self.detailed_log_file.write(f"Training Configuration:\n{str(training_config)}\n\n")
        self.detailed_log_file.write(f"Data Configuration:\n{str(data_config)}\n\n")
        self.detailed_log_file.write("=== Training Start ===\n\n")
        self.detailed_log_file.flush()
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'detailed_log_file') and self.detailed_log_file:
            self.detailed_log_file.close()
    
    def initialize_model(self) -> None:
        """Initialize the model with verbose logging."""
        self.logger.info("Initializing model with verbose logging...")
        
        # Create model instance with verbose flag
        self.model = SemanticResonanceModel(self.model_config)
        self.model.to(self.device)
        
        # Log model size and structure
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model size: {total_params:,} parameters ({trainable_params:,} trainable)")
        
        # Log detailed model structure
        self.detailed_log_file.write(f"Model Architecture:\n{str(self.model)}\n\n")
        self.detailed_log_file.write(f"Total Parameters: {total_params:,}\n")
        self.detailed_log_file.write(f"Trainable Parameters: {trainable_params:,}\n\n")
        
        # Log parameter sizes by layer
        self.detailed_log_file.write("Parameter Sizes by Layer:\n")
        for name, param in self.model.named_parameters():
            self.detailed_log_file.write(f"  {name}: {param.numel():,} parameters, shape: {list(param.shape)}\n")
        
        self.detailed_log_file.write("\n")
        self.detailed_log_file.flush()
    
    def initialize_tokenizer(self) -> None:
        """Initialize the tokenizer with verbose logging."""
        self.logger.info(f"Loading tokenizer: {self.data_config.tokenizer_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.data_config.tokenizer_name)
        
        # Set default pad token if not set
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Update vocab size in model config
        self.model_config.vocab_size = len(self.tokenizer)
        
        # Log tokenizer information
        self.detailed_log_file.write(f"Tokenizer: {self.data_config.tokenizer_name}\n")
        self.detailed_log_file.write(f"Vocabulary Size: {len(self.tokenizer)}\n")
        self.detailed_log_file.write(f"Special Tokens: {self.tokenizer.special_tokens_map}\n\n")
        self.detailed_log_file.flush()
        
        # If model is already initialized, resize embedding layer
        if self.model is not None and hasattr(self.model, "resize_token_embeddings"):
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def initialize_dataloaders(self) -> None:
        """Initialize the dataloaders with verbose logging."""
        self.logger.info("Loading datasets with verbose logging...")
        
        # Import dataloader function
        from src.data.dataloaders import get_wikitext_dataloaders
        
        # Create dataloaders
        self.dataloaders = get_wikitext_dataloaders(
            tokenizer=self.tokenizer,
            batch_size=self.training_config.batch_size,
            eval_batch_size=self.training_config.eval_batch_size,
            max_length=self.data_config.max_length,
            stride=self.data_config.stride,
            num_workers=self.data_config.preprocessing_num_workers,
            cache_dir=self.data_config.cache_dir
        )
        
        # Log dataset sizes
        self.logger.info(f"Dataset sizes:")
        self.detailed_log_file.write("Dataset Information:\n")
        
        for split, dataloader in self.dataloaders.items():
            size = len(dataloader.dataset)
            batches = len(dataloader)
            self.logger.info(f"  {split}: {size} examples ({batches} batches)")
            self.detailed_log_file.write(f"  {split}: {size} examples ({batches} batches)\n")
        
        # Sample some data for logging
        self.detailed_log_file.write("\nData Sample:\n")
        try:
            sample_batch = next(iter(self.dataloaders["train"]))
            sample_input_ids = sample_batch["input_ids"][0]
            sample_text = self.tokenizer.decode(sample_input_ids)
            
            self.detailed_log_file.write(f"Sample text (truncated to 200 chars):\n")
            self.detailed_log_file.write(f"{sample_text[:200]}...\n\n")
            
            self.detailed_log_file.write(f"Sample shape: {sample_batch['input_ids'].shape}\n")
            self.detailed_log_file.write(f"Sample input_ids (first 20 tokens): {sample_input_ids[:20].tolist()}\n\n")
        except:
            self.detailed_log_file.write("Could not sample data\n\n")
        
        self.detailed_log_file.flush()

    def initialize_optimizer(self) -> None:
        """Initialize the optimizer and learning rate scheduler with verbose logging."""
        super().initialize_optimizer()
        
        # Log optimizer settings
        self.detailed_log_file.write(f"Optimizer: {self.training_config.optimizer}\n")
        self.detailed_log_file.write(f"Learning Rate: {self.training_config.learning_rate}\n")
        self.detailed_log_file.write(f"Weight Decay: {self.training_config.weight_decay}\n")
        
        if self.lr_scheduler:
            self.detailed_log_file.write(f"Learning Rate Scheduler: {self.training_config.lr_scheduler}\n")
            self.detailed_log_file.write(f"Warmup Steps: {self.training_config.warmup_steps}\n")
        
        self.detailed_log_file.write("\n")
        self.detailed_log_file.flush()
    
    def train_epoch(self) -> Dict[str, Any]:
        """
        Train for one epoch with verbose logging.
        
        Returns:
            Dictionary of epoch statistics
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_step = 0
        
        # Start epoch logging
        self.training_logger.start_epoch(self.current_epoch, self.training_config.max_epochs)
        self.detailed_log_file.write(f"=== Epoch {self.current_epoch+1}/{self.training_config.max_epochs} ===\n\n")
        
        # Log memory usage
        self.training_logger.log_memory(f"Start of epoch {self.current_epoch+1}:")
        
        # Create progress bar
        train_iterator = tqdm(
            self.dataloaders["train"],
            desc=f"Epoch {self.current_epoch+1}/{self.training_config.max_epochs}",
            leave=True
        )
        
        # Save gradients periodically
        grad_log_steps = min(100, len(train_iterator) // 10)
        
        for batch_idx, batch in enumerate(train_iterator):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.training_config.use_mixed_precision):
                outputs = self.model(**batch, return_dict=True)
                loss = outputs["loss"]
                
                # Handle gradient accumulation
                if self.training_config.accumulation_steps > 1:
                    loss = loss / self.training_config.accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Log detailed metrics
            if batch_idx % 20 == 0:
                # Log a sample of predictions vs targets
                try:
                    input_ids = batch["input_ids"][0]
                    labels = batch["labels"][0]
                    
                    # Get model predictions
                    with torch.no_grad():
                        logits = outputs["logits"][0]
                        predictions = torch.argmax(logits, dim=-1)
                    
                    # Convert to text
                    input_text = self.tokenizer.decode(input_ids)
                    pred_text = self.tokenizer.decode(predictions)
                    
                    # Write to detailed log
                    self.detailed_log_file.write(f"Batch {batch_idx}, Step {self.global_step} Sample:\n")
                    self.detailed_log_file.write(f"  Input: {input_text[:100]}...\n")
                    self.detailed_log_file.write(f"  Prediction: {pred_text[:100]}...\n")
                    self.detailed_log_file.write(f"  Loss: {loss.item()}\n\n")
                    self.detailed_log_file.flush()
                except:
                    pass
            
            # Update weights if accumulation complete
            if (batch_idx + 1) % self.training_config.accumulation_steps == 0:
                # Log gradients
                if batch_idx % grad_log_steps == 0:
                    # Save gradient statistics
                    grad_norms = []
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm(2).item()
                            grad_norms.append((name, grad_norm))
                    
                    # Write to detailed log
                    if grad_norms:
                        self.detailed_log_file.write(f"Gradient norms at step {self.global_step}:\n")
                        for name, norm in sorted(grad_norms, key=lambda x: x[1], reverse=True)[:10]:
                            self.detailed_log_file.write(f"  {name}: {norm:.6f}\n")
                        self.detailed_log_file.write("\n")
                        self.detailed_log_file.flush()
                
                # Clip gradients
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.max_grad_norm
                )
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
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
            
            # Log to training logger
            self.training_logger.log_batch(
                self.current_epoch, 
                batch_idx,
                self.training_config.batch_size,
                len(train_iterator),
                loss.item() * self.training_config.accumulation_steps,
                lr
            )
            
            # Evaluate and save checkpoint
            if self.training_config.eval_steps > 0 and self.global_step % self.training_config.eval_steps == 0:
                # Log memory usage
                self.training_logger.log_memory(f"Before validation (step {self.global_step}):")
                
                # Run evaluation
                val_metrics = self.evaluate()
                
                # Log validation metrics
                self.training_logger.log_validation(val_metrics)
                
                # Write to detailed log
                self.detailed_log_file.write(f"Validation at step {self.global_step}:\n")
                for key, value in val_metrics.items():
                    self.detailed_log_file.write(f"  {key}: {value}\n")
                self.detailed_log_file.write("\n")
                self.detailed_log_file.flush()
                
                # Check for improvement
                if val_metrics["loss"] < self.best_val_loss:
                    improvement = (self.best_val_loss - val_metrics["loss"]) / self.best_val_loss
                    self.logger.info(f"Validation loss improved by {improvement:.2%}")
                    
                    self.best_val_loss = val_metrics["loss"]
                    
                    # Save best model
                    best_model_path = os.path.join(self.output_dir, "best_model.pt")
                    self.save_checkpoint(best_model_path)
                    self.training_logger.log_checkpoint(best_model_path)
                
                # Switch back to training mode
                self.model.train()
            
            # Save checkpoint
            if self.training_config.save_steps > 0 and self.global_step % self.training_config.save_steps == 0:
                checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"step_{self.global_step}.pt")
                self.save_checkpoint(checkpoint_path)
                self.training_logger.log_checkpoint(checkpoint_path)
        
        # Calculate average loss for the epoch
        epoch_loss = epoch_loss / epoch_step
        
        # End epoch logging
        self.training_logger.end_epoch(self.current_epoch, epoch_loss)
        
        # Log memory usage
        self.training_logger.log_memory(f"End of epoch {self.current_epoch+1}:")
        
        return {
            "train_loss": epoch_loss,
            "steps": epoch_step,
            "learning_rate": lr
        }
    
    def evaluate(self, split: str = "validation") -> Dict[str, Any]:
        """
        Evaluate the model with verbose logging.
        
        Args:
            split: Dataset split to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        if split not in self.dataloaders:
            self.logger.warning(f"Split '{split}' not found in dataloaders. Available splits: {list(self.dataloaders.keys())}")
            return {"loss": float("inf"), "perplexity": float("inf")}
        
        self.model.eval()
        self.logger.info(f"Evaluating model on {split} split...")
        
        total_loss = 0.0
        total_steps = 0
        
        # Collect detailed metrics
        token_losses = []
        prediction_samples = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloaders[split], desc=f"Evaluating ({split})")):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch, return_dict=True)
                loss = outputs["loss"]
                
                # Update statistics
                total_loss += loss.item()
                total_steps += 1
                
                # Collect detailed metrics (sample-based)
                if batch_idx % 50 == 0 and len(prediction_samples) < 5:
                    try:
                        # Get model predictions
                        logits = outputs["logits"][0]
                        predictions = torch.argmax(logits, dim=-1)
                        
                        # Get input and labels
                        input_ids = batch["input_ids"][0]
                        labels = batch["labels"][0]
                        
                        # Convert to text
                        input_text = self.tokenizer.decode(input_ids)
                        pred_text = self.tokenizer.decode(predictions)
                        
                        # Add to samples
                        prediction_samples.append({
                            "input": input_text[:200] + "...",
                            "prediction": pred_text[:200] + "...",
                            "loss": loss.item()
                        })
                    except:
                        pass
        
        # Calculate average loss
        avg_loss = total_loss / total_steps
        
        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Prepare metrics dictionary
        metrics = {
            "loss": avg_loss,
            "perplexity": perplexity
        }
        
        # Log detailed evaluation results
        self.detailed_log_file.write(f"Detailed Evaluation Results ({split}):\n")
        self.detailed_log_file.write(f"  Loss: {avg_loss:.6f}\n")
        self.detailed_log_file.write(f"  Perplexity: {perplexity:.6f}\n\n")
        
        # Log prediction samples
        if prediction_samples:
            self.detailed_log_file.write("Prediction Samples:\n")
            for i, sample in enumerate(prediction_samples):
                self.detailed_log_file.write(f"Sample {i+1}:\n")
                self.detailed_log_file.write(f"  Input: {sample['input']}\n")
                self.detailed_log_file.write(f"  Prediction: {sample['prediction']}\n")
                self.detailed_log_file.write(f"  Loss: {sample['loss']:.6f}\n\n")
        
        self.detailed_log_file.flush()
        
        return metrics
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save a checkpoint with verbose logging.
        
        Args:
            path: Path to save the checkpoint to
        """
        super().save_checkpoint(path)
        
        # Log detailed checkpoint information
        self.detailed_log_file.write(f"Saved checkpoint to {path}\n")
        self.detailed_log_file.write(f"  Epoch: {self.current_epoch+1}/{self.training_config.max_epochs}\n")
        self.detailed_log_file.write(f"  Global Step: {self.global_step}\n")
        self.detailed_log_file.write(f"  Best Validation Loss: {self.best_val_loss:.6f}\n\n")
        self.detailed_log_file.flush()