#!/usr/bin/env python3
"""
Train a QLLM model using the Daily Dialog dataset.

This script demonstrates how to train a model using the Daily Dialog dataset
with memory and multimodal extensions enabled.
"""

import os
import logging
import torch
from typing import Dict, Any
import argparse
import time

from src.data.loaders.daily_dialog_loader import DailyDialogLoader
from src.model.semantic_resonance_model_with_extensions import SemanticResonanceModelWithExtensions
from src.training.trainers.dialogue_trainer import DialogueTrainer
from src.model.extensions.extension_config import ExtensionConfig
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train QLLM model on Daily Dialog dataset")
    
    # Model configuration
    parser.add_argument(
        "--model-size",
        type=str,
        default="mini",
        choices=["mini", "small", "medium", "large"],
        help="Size/configuration of the model to train"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/daily_dialog",
        help="Directory to save model checkpoints and logs"
    )
    
    # Training configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation (if None, use training batch size)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=3,
        help="Maximum number of dialogue history turns"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt to prepend to dialogues"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching datasets and models"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading"
    )
    
    # Extension configuration
    parser.add_argument(
        "--disable-memory",
        action="store_true",
        help="Disable memory extension"
    )
    parser.add_argument(
        "--disable-multimodal",
        action="store_true",
        help="Disable multimodal extension"
    )
    parser.add_argument(
        "--disable-quantum",
        action="store_true",
        help="Disable quantum extension"
    )
    
    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for training (auto, cuda, cpu, mps)"
    )
    
    return parser.parse_args()

def get_device(device_config):
    """Get the appropriate PyTorch device with robust fallbacks."""
    # Initialize to CPU for safety
    device = "cpu"
    
    try:
        if device_config != "auto":
            # User specified a device, try to use it
            if device_config == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
            elif device_config == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
                logger.warning("MPS requested but not available. Falling back to CPU.")
            else:
                try:
                    # Try to create the device
                    torch.device(device_config)
                    device = device_config
                except Exception as e:
                    logger.warning(f"Could not use device '{device_config}': {e}")
        else:
            # Auto-detect
            if torch.cuda.is_available():
                try:
                    # Try creating a tensor on CUDA to verify it works
                    test_tensor = torch.zeros(1).cuda()
                    test_tensor.cpu()  # Free the memory
                    device = "cuda"
                    logger.info("CUDA is available and working")
                except Exception as e:
                    logger.warning(f"CUDA reports as available but has error: {e}")
                    logger.warning("CUDA will not be used")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    # Verify MPS is working
                    test_tensor = torch.zeros(1).to("mps")
                    test_tensor.cpu()  # Free the memory
                    device = "mps"
                    logger.info("MPS (Apple Silicon) is available and working")
                except Exception as e:
                    logger.warning(f"MPS reports as available but has error: {e}")
    except Exception as e:
        logger.warning(f"Error during device detection: {e}")
        logger.warning("Using CPU as fallback")
    
    # Log the final device choice
    if device == "cpu":
        logger.info("Using CPU for training")
    
    return torch.device(device)

def main():
    """Main training function."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up file handler for logging
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "train.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create tokenizer first (needed for vocab size)
    logger.info("Creating tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Set padding token - GPT2 tokenizer doesn't have one by default
    logger.info("Setting up padding token for tokenizer")
    if tokenizer.pad_token is None:
        # Use the EOS token as the padding token
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    # Configure model
    model_config = ModelConfig()
    model_config.model_size = args.model_size
    model_config.max_seq_length = args.max_length
    hidden_dims = {
        "mini": 768,
        "small": 1024,
        "medium": 1536,
        "large": 2048
    }
    model_config.hidden_dim = hidden_dims[args.model_size]
    
    # CRITICAL: Set the vocabulary size to match the tokenizer
    model_config.vocab_size = len(tokenizer)
    logger.info(f"Setting vocab_size to match tokenizer: {model_config.vocab_size}")
    
    # Configure extensions
    ext_config = ExtensionConfig()
    ext_config.memory.enabled = not args.disable_memory
    ext_config.multimodal.enabled = not args.disable_multimodal
    ext_config.quantum.enabled = not args.disable_quantum
    
    # Set padding token - GPT2 tokenizer doesn't have one by default
    logger.info("Setting up padding token for tokenizer")
    if tokenizer.pad_token is None:
        # Use the EOS token as the padding token
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    # Log configuration
    logger.info(f"Training model with size: {args.model_size}")
    logger.info(f"Memory extension: {'Disabled' if args.disable_memory else 'Enabled'}")
    logger.info(f"Multimodal extension: {'Disabled' if args.disable_multimodal else 'Enabled'}")
    logger.info(f"Quantum extension: {'Disabled' if args.disable_quantum else 'Enabled'}")
    
    # Get the Daily Dialog dataset
    logger.info("Loading Daily Dialog dataset")
    daily_dialog_loader = DailyDialogLoader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        system_prompt=args.system_prompt
    )
    
    # Load dataset and create dataloaders
    dataloaders = daily_dialog_loader.load()
    train_dataloader = dataloaders["train"]
    eval_dataloader = dataloaders.get("validation")
    
    logger.info(f"Loaded {len(train_dataloader)} training batches")
    if eval_dataloader:
        logger.info(f"Loaded {len(eval_dataloader)} validation batches")
    
    # Create model
    logger.info(f"Creating model with {model_config.hidden_dim} hidden dimensions")
    model_kwargs = model_config.to_dict()
    model_kwargs["extension_config"] = ext_config.to_dict()
    model = SemanticResonanceModelWithExtensions(model_kwargs)
    # Don't move model to device here - let the trainer handle it
    
    # Configure training
    training_config = TrainingConfig()
    training_config.max_epochs = args.epochs
    training_config.learning_rate = args.learning_rate
    training_config.batch_size = args.batch_size
    training_config.output_dir = args.output_dir
    training_config.model_type = "dialogue"
    training_config.device = "cpu"  # Force CPU to avoid CUDA errors
    logger.info(f"Setting training device explicitly to CPU")
    
    # Create trainer
    trainer = DialogueTrainer(
        model=model,
        train_dataloader=train_dataloader,
        training_config=training_config,
        model_config=model_config,
        eval_dataloader=eval_dataloader,
        max_history_turns=args.max_history_turns
    )
    
    # Start training
    logger.info("Starting training")
    start_time = time.time()
    training_results = trainer.train()
    training_duration = time.time() - start_time
    
    # Log results
    logger.info(f"Training completed in {training_duration:.2f} seconds")
    logger.info(f"Final loss: {training_results.get('final_loss', 'N/A')}")
    logger.info(f"Best metric: {training_results.get('best_metric', 'N/A')}")
    
    logger.info(f"Model saved to {args.output_dir}")
    return 0

if __name__ == "__main__":
    main()