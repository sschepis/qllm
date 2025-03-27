"""
Base trainer for the Quantum Resonance Language Model.

This module provides a base trainer class that defines common functionality
for different trainer implementations.
"""

import os
import torch
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.config.data_config import DataConfig
from src.utils.device import get_device


class BaseTrainer:
    """Base trainer implementation for QLLM."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        output_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the base trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            data_config: Data configuration
            output_dir: Directory for outputs (default: from training_config)
            logger: Logger instance
        """
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        
        # Set output directory
        self.output_dir = output_dir or training_config.output_dir
        if not self.output_dir:
            # Default output directory
            self.output_dir = os.path.join("runs", "quantum_resonance")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set logger
        self.logger = logger or logging.getLogger("qllm_trainer")
        
        # Save configuration
        self.save_config()
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.lr_scheduler = None
        self.dataloaders = {}
        
        # Training state
        self.current_epoch = 0
        self.last_checkpoint_path = None
        
        # Set device
        self.device = self._get_device()
        self.logger.info(f"Using device: {self.device}")
    
    def _get_device(self) -> torch.device:
        """
        Get the appropriate device.
        
        Returns:
            Device to use
        """
        return get_device(self.training_config.device)
    
    def save_config(self) -> None:
        """Save configuration to the output directory."""
        # Create config dictionary
        config = {
            "model": self.model_config.to_dict(),
            "training": self.training_config.to_dict(),
            "data": self.data_config.to_dict()
        }
        
        # Save config
        import json
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    def initialize_model(self) -> None:
        """Initialize the model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement initialize_model.")
    
    def initialize_tokenizer(self) -> None:
        """Initialize the tokenizer. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement initialize_tokenizer.")
    
    def initialize_dataloaders(self) -> None:
        """Initialize the dataloaders. To be implemented by subclasses."""
        # Import the get_appropriate_dataloaders function here to avoid circular imports
        from src.data.dataloaders import get_appropriate_dataloaders
        
        # Explicitly pass batch_size and eval_batch_size from training_config
        self.dataloaders = get_appropriate_dataloaders(
            data_config=self.data_config,
            tokenizer=self.tokenizer,
            batch_size=self.training_config.batch_size,
            eval_batch_size=getattr(self.training_config, "eval_batch_size", self.training_config.batch_size),
            num_workers=self.data_config.preprocessing_num_workers
        )
        
        # Log dataset sizes
        self.logger.info(f"Dataset sizes:")
        for split, dataloader in self.dataloaders.items():
            self.logger.info(f"  {split}: {len(dataloader.dataset)} examples, {len(dataloader)} batches")
    
    def initialize_optimizer(self) -> None:
        """Initialize the optimizer and learning rate scheduler."""
        # Check if model is initialized
        if self.model is None:
            raise ValueError("Model must be initialized before optimizer.")
        
        # Set optimizer
        if self.training_config.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                momentum=0.9,
                weight_decay=self.training_config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.training_config.optimizer}")
        
        # Set learning rate scheduler
        if self.training_config.lr_scheduler.lower() == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config.max_epochs * len(self.dataloaders.get("train", [])),
                eta_min=1e-6
            )
        elif self.training_config.lr_scheduler.lower() == "linear":
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.training_config.max_epochs * len(self.dataloaders.get("train", []))
            )
        elif self.training_config.lr_scheduler.lower() == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.training_config.max_epochs // 3,
                gamma=0.1
            )
        elif self.training_config.lr_scheduler.lower() == "none":
            self.lr_scheduler = None
        else:
            self.lr_scheduler = None
    
    def train_epoch(self) -> Dict[str, Any]:
        """
        Train for one epoch. To be implemented by subclasses.
        
        Returns:
            Dictionary of epoch statistics
        """
        raise NotImplementedError("Subclasses must implement train_epoch.")
    
    def evaluate(self, split: str = "validation") -> Dict[str, Any]:
        """
        Evaluate the model. To be implemented by subclasses.
        
        Args:
            split: Dataset split to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate.")
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the model for a specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train (default: from training_config)
            
        Returns:
            Dictionary of training statistics
        """
        # Set number of epochs
        num_epochs = num_epochs or self.training_config.max_epochs
        
        # Check if model, tokenizer, and optimizer are initialized
        if self.model is None:
            self.initialize_model()
        if self.tokenizer is None:
            self.initialize_tokenizer()
        if not self.dataloaders:
            self.initialize_dataloaders()
        if self.optimizer is None:
            self.initialize_optimizer()
        
        # Train for each epoch
        train_stats = {}
        for epoch in range(num_epochs):
            self.logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
            
            # Train epoch
            epoch_stats = self.train_epoch()
            train_stats[f"epoch_{epoch+1}"] = epoch_stats
            
            # Evaluate model
            eval_stats = self.evaluate()
            train_stats[f"epoch_{epoch+1}_eval"] = eval_stats
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {epoch_stats.get('train_loss', 'N/A')}, "
                f"Validation Loss: {eval_stats.get('loss', 'N/A')}, "
                f"Validation Perplexity: {eval_stats.get('perplexity', 'N/A')}"
            )
            
            # Save checkpoint if configured
            if self.training_config.save_every_epoch:
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
                self.save_checkpoint(checkpoint_path)
        
        # Save final model
        self.save_checkpoint(os.path.join(self.output_dir, "final_model.pt"))
        
        return train_stats
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save a checkpoint.
        
        Args:
            path: Path to save checkpoint to
        """
        # Ensure model is initialized
        if self.model is None:
            raise ValueError("Model must be initialized before saving checkpoint.")
        
        # Create checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "epoch": getattr(self, "current_epoch", 0),
            "model_config": self.model_config.to_dict(),
            "training_config": self.training_config.to_dict(),
            "data_config": self.data_config.to_dict()
        }
        
        # Save checkpoint (create directory if needed)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        if self.training_config.disable_optimizer_saving:
            # Save only model weights
            torch.save(checkpoint["model_state_dict"], path)
        else:
            # Save full checkpoint
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load a checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        # Ensure model is initialized
        if self.model is None:
            raise ValueError("Model must be initialized before loading checkpoint.")
        
        # Load checkpoint
        checkpoint = torch.load(
            path,
            map_location=self.device
        )
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Full checkpoint
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
            # Load optimizer if available
            if self.optimizer and "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Load epoch
            if "epoch" in checkpoint:
                self.current_epoch = checkpoint["epoch"]
        else:
            # Model weights only
            self.model.load_state_dict(checkpoint)
            
            # Set default epoch
            self.current_epoch = 0