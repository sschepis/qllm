#!/usr/bin/env python
"""
Example script demonstrating the enhanced training system.

This script shows how to use the enhanced training system with
different model types and training strategies.
"""

import os
import sys
import argparse
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

# Add the parent directory to the path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.config.data_config import DataConfig

from src.training import (
    get_trainer,
    create_trainer_for_model_type,
    EnhancedTrainer
)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log")
        ]
    )
    return logging.getLogger("quantum_resonance")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a language model using the enhanced training system")
    
    # Model type arguments
    parser.add_argument("--model-type", type=str, default="standard",
                        choices=["standard", "dialogue", "multimodal"],
                        help="Type of model to train")
    
    # Training strategy arguments
    parser.add_argument("--training-strategy", type=str, default="standard",
                        choices=["standard", "finetune"],
                        help="Training strategy to use")
    
    # Configuration file arguments
    parser.add_argument("--model-config", type=str, default=None,
                        help="Path to model configuration file")
    parser.add_argument("--training-config", type=str, default=None,
                        help="Path to training configuration file")
    parser.add_argument("--data-config", type=str, default=None,
                        help="Path to data configuration file")
    
    # Training arguments
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for outputs")
    parser.add_argument("--max-epochs", type=int, default=3,
                        help="Maximum number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay factor")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Number of warmup steps")
    
    # Extension arguments
    parser.add_argument("--enable-extensions", type=str, nargs="+", default=[],
                        help="List of extensions to enable")
    
    # Device arguments
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., 'cuda', 'cpu')")
    
    # Checkpoint arguments
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Automatically resume from the latest checkpoint")
    
    # Mixed precision arguments
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Use mixed precision training")
    
    # Override default configs
    parser.add_argument("--override-model", type=str, nargs="+", default=[],
                        help="Override model config (key=value pairs)")
    parser.add_argument("--override-training", type=str, nargs="+", default=[],
                        help="Override training config (key=value pairs)")
    parser.add_argument("--override-data", type=str, nargs="+", default=[],
                        help="Override data config (key=value pairs)")
    
    return parser.parse_args()


def load_config_from_file(file_path, config_class):
    """Load configuration from a file."""
    if file_path is None:
        return config_class()
    
    import json
    try:
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        return config_class(**config_dict)
    except Exception as e:
        print(f"Error loading config from {file_path}: {e}")
        return config_class()


def apply_overrides(config, overrides):
    """Apply key=value overrides to a configuration."""
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        
        # Try to convert value to appropriate type
        try:
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            elif "." in value and all(part.isdigit() for part in value.split(".", 1)):
                value = float(value)
        except Exception:
            pass
        
        # Set attribute
        setattr(config, key, value)
    
    return config


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    # Load configurations from files
    model_config = load_config_from_file(args.model_config, ModelConfig)
    training_config = load_config_from_file(args.training_config, TrainingConfig)
    data_config = load_config_from_file(args.data_config, DataConfig)
    
    # Apply command-line overrides to configurations
    model_config = apply_overrides(model_config, args.override_model)
    training_config = apply_overrides(training_config, args.override_training)
    data_config = apply_overrides(data_config, args.override_data)
    
    # Set command-line argument values in configurations
    if args.max_epochs:
        training_config.max_epochs = args.max_epochs
    if args.batch_size:
        training_config.batch_size = args.batch_size
    if args.learning_rate:
        training_config.learning_rate = args.learning_rate
    if args.weight_decay:
        training_config.weight_decay = args.weight_decay
    if args.warmup_steps:
        training_config.warmup_steps = args.warmup_steps
    if args.device:
        training_config.device = args.device
    if args.mixed_precision:
        training_config.use_mixed_precision = True
    if args.auto_resume:
        training_config.auto_resume = True
    
    # Set model type and training strategy
    training_config.model_type = args.model_type
    training_config.training_strategy = args.training_strategy
    
    # Set enabled extensions
    if args.enable_extensions:
        training_config.enabled_extensions = args.enable_extensions
    
    # Create trainer
    trainer = create_trainer_for_model_type(
        model_type=args.model_type,
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        output_dir=args.output_dir,
        logger=logger
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        trainer.checkpoint_manager.load_checkpoint(
            trainer.model,
            trainer.optimizer,
            trainer.scheduler,
            path=args.resume_from
        )
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    
    # Save the final model
    final_model_path = os.path.join(trainer.output_dir, "final_model.pt")
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Training completed. Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()