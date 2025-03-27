#!/usr/bin/env python3
"""
Dialogue Training Script for Semantic Resonance Model with Extensions.

This script trains a Semantic Resonance Model with Extensions (multimodal, memory, quantum)
on dialogue data with continuous learning capabilities. It allows for ongoing improvement
through feedback and conversation data.
"""

import os
import argparse
import json
import torch
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

from transformers import AutoTokenizer

from src.config import ModelConfig, TrainingConfig, DataConfig, get_default_configs
from src.model.semantic_resonance_model_with_extensions import SemanticResonanceModelWithExtensions
from src.model.extensions.extension_config import ExtensionConfig
from src.data.dialogue_dataset import get_dialogue_dataloaders
from src.data.function_calling_dataset import get_function_calling_dataloaders, get_default_function_definitions
from src.training.trainer import Trainer
from src.training.continuous_learning import ContinuousLearningManager, ContinuousLearningConfig
from src.training.checkpoint import find_latest_checkpoint
from src.utils.device import get_device
from src.utils.logging import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Dialogue-based Semantic Resonance Model with Extensions")
    
    # Model configuration
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    
    # Prime Hilbert Encoder settings
    parser.add_argument("--primes", type=int, nargs="+", default=[7, 11, 13, 17, 19], 
                        help="Prime numbers for subspace decomposition")
    parser.add_argument("--base_dim", type=int, default=768, 
                        help="Base embedding dimension")
    
    # Resonance Block settings
    parser.add_argument("--max_iterations", type=int, default=10, 
                        help="Maximum iterations for resonance attention")
    parser.add_argument("--entropy_threshold", type=float, default=0.1, 
                        help="Entropy threshold for halting")
    parser.add_argument("--use_prime_mask", action="store_true", 
                        help="Use prime resonance mask for feed-forward layers")
    
    # Extension settings
    parser.add_argument("--enable_extensions", action="store_true", 
                        help="Enable all extensions")
    parser.add_argument("--enable_multimodal", action="store_true", 
                        help="Enable multimodal extension")
    parser.add_argument("--enable_memory", action="store_true", 
                        help="Enable memory extension")
    parser.add_argument("--enable_quantum", action="store_true", 
                        help="Enable quantum extension")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum epochs")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # Data configuration
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Tokenizer name")
    parser.add_argument("--dataset_name", type=str, default="daily_dialog",
                        help="Dataset name (daily_dialog, convai2, empathetic_dialogues)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to custom dialogue data (JSON format)")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Cache directory")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="System prompt to prepend to all conversations")
    
    # Function calling configuration
    parser.add_argument("--enable_function_calling", action="store_true",
                        help="Enable function calling capabilities")
    parser.add_argument("--function_defs_path", type=str, default=None,
                        help="Path to JSON file with function definitions")
    parser.add_argument("--json_format_probability", type=float, default=0.5,
                        help="Probability of formatting responses as JSON")
    
    # Continuous learning configuration
    parser.add_argument("--continuous_learning", action="store_true", 
                        help="Enable continuous learning")
    parser.add_argument("--learning_mode", type=str, default="adaptive", 
                        choices=["adaptive", "scheduled", "feedback_driven"],
                        help="Continuous learning mode")
    parser.add_argument("--min_samples_for_update", type=int, default=50, 
                        help="Minimum samples for model update")
    parser.add_argument("--memory_buffer_size", type=int, default=2000, 
                        help="Memory buffer size for learning history")
    parser.add_argument("--update_steps", type=int, default=500, 
                        help="Steps per update during continuous learning")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="runs/dialogue_model", 
                        help="Output directory")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    
    # Hardware configuration
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to train on (default: cuda if available)")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of workers for data loading")
    
    # Resuming training
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--resume_continuous_learning", action="store_true",
                        help="Resume continuous learning state if resuming from checkpoint")
    parser.add_argument("--auto_resume", action="store_true",
                        help="Automatically resume from latest checkpoint if available")
    
    return parser.parse_args()


def create_model_config_from_args(args):
    """Create model configuration from command line arguments."""
    # Get number of heads
    num_heads = args.num_heads
    
    # Customize primes to make their sum divisible by num_heads
    # Default primes [7, 11, 13, 17, 19] sum to 67, which isn't divisible by most head counts
    # Let's choose primes that sum to a value divisible by num_heads
    if num_heads == 12:
        # Choose primes that sum to 72 (divisible by 12)
        primes = [3, 5, 7, 11, 13, 17, 16]  # 72 total
        print(f"Using custom primes {primes} with sum 72 (divisible by {num_heads})")
    elif num_heads == 8:
        # Choose primes that sum to 64 (divisible by 8)
        primes = [3, 5, 7, 11, 13, 11, 14]  # 64 total
        print(f"Using custom primes {primes} with sum 64 (divisible by {num_heads})")
    elif num_heads == 6:
        # Choose primes that sum to 48 (divisible by 6)
        primes = [5, 7, 11, 13, 12]  # 48 total
        print(f"Using custom primes {primes} with sum 48 (divisible by {num_heads})")
    elif num_heads == 4:
        # Choose primes that sum to 48 (divisible by 4)
        primes = [11, 13, 10, 14]  # 48 total
        print(f"Using custom primes {primes} with sum 48 (divisible by {num_heads})")
    else:
        # For other head counts, create a new set of primes that work
        total = 0
        primes = []
        prime_candidates = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        
        # Start with standard primes
        for p in prime_candidates:
            primes.append(p)
            total += p
            if total % num_heads == 0 and len(primes) >= 3:
                break
            
        # If we don't have enough primes yet, add numbers that make the total divisible
        if total % num_heads != 0:
            remainder = num_heads - (total % num_heads)
            primes.append(remainder)
            total += remainder
        
        print(f"Using custom primes {primes} with sum {total} (divisible by {num_heads})")
    
    # Create model config with adjusted primes
    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=num_heads,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout,
        primes=primes,  # Use our custom primes
        base_dim=args.base_dim,
        max_iterations=args.max_iterations,
        entropy_threshold=args.entropy_threshold,
        use_prime_mask=args.use_prime_mask
    )
    
    # Create extension config
    extension_config = ExtensionConfig()
    extension_config.extensions_enabled = args.enable_extensions
    extension_config.enable_multimodal = args.enable_multimodal or args.enable_extensions
    extension_config.enable_memory = args.enable_memory or args.enable_extensions
    extension_config.enable_quantum = args.enable_quantum or args.enable_extensions
    
    # Configure multimodal extension
    if extension_config.enable_multimodal:
        extension_config.multimodal_config = {
            "vision_model": "resnet50",
            "use_spatial_features": True,
            "fusion_type": "film",
            "vision_primes": [23, 29, 31, 37],
            "fusion_heads": 6
        }
    
    # Configure memory extension
    if extension_config.enable_memory:
        extension_config.memory_config = {
            "memory_size": 1000,
            "entity_dim": 256,
            "relation_dim": 128,
            "max_entities": 10000,
            "max_relations": 50000
        }
    
    # Configure quantum extension
    if extension_config.enable_quantum:
        extension_config.quantum_config = {
            "pattern_type": "harmonic",
            "base_sparsity": 0.8,
            "mask_type": "binary"
        }
    
    return model_config, extension_config


def create_training_config_from_args(args):
    """Create training configuration from command line arguments."""
    # Create training config
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        warmup_steps=args.warmup_steps,
        accumulation_steps=args.accumulation_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        device=args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Add output directory to training config
    training_config.output_dir = args.output_dir
    
    return training_config


def create_continuous_learning_config_from_args(args):
    """Create continuous learning configuration from command line arguments."""
    # Create continuous learning config
    cl_config = ContinuousLearningConfig(
        learning_mode=args.learning_mode,
        min_samples_for_update=args.min_samples_for_update,
        memory_buffer_size=args.memory_buffer_size,
        # Other settings with default values
        use_memory_extension=args.enable_memory or args.enable_extensions,
        knowledge_persistence=True,
        quantum_adaptation=args.enable_quantum or args.enable_extensions
    )
    
    return cl_config


def save_config(config, output_dir, filename="config.json"):
    """Save configuration to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    def convert_to_serializable(obj):
        """Convert an object to a JSON-serializable format."""
        if hasattr(obj, '__dict__'):
            # Extract object attributes
            return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()
                    if not k.startswith('_') and not callable(v)}
        elif isinstance(obj, torch.device):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        else:
            # Return the object as is if it's a basic type
            return obj
    
    # Convert config to serializable dictionary
    if hasattr(config, '__dict__'):
        config_dict = convert_to_serializable(config)
    else:
        config_dict = config
    
    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(config_dict, f, indent=2)


def init_speaker_tokens(tokenizer):
    """Initialize speaker tokens in the tokenizer."""
    speaker_tokens = {
        "system": "<|system|>",
        "user": "<|user|>",
        "assistant": "<|assistant|>",
        "end": "<|end|>"
    }
    
    # Add special tokens to tokenizer
    special_tokens = list(speaker_tokens.values())
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    return speaker_tokens


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    
    # Check if we should auto-resume from any existing run in the base directory
    if args.auto_resume and not args.resume_from:
        # Look for existing runs in the base output directory
        runs = [d for d in os.listdir(args.output_dir)
                if os.path.isdir(os.path.join(args.output_dir, d)) and d.startswith("dialogue_model_")]
        
        if runs:
            # Sort runs by creation time (newest first)
            runs.sort(reverse=True)
            latest_run = os.path.join(args.output_dir, runs[0])
            
            # Check if this run has any checkpoints
            checkpoint_dir = os.path.join(latest_run, "checkpoints")
            if os.path.exists(checkpoint_dir):
                latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
                if latest_checkpoint:
                    logger.info(f"Auto-resuming from latest run: {latest_run}")
                    logger.info(f"Latest checkpoint: {latest_checkpoint}")
                    output_dir = latest_run
                    args.resume_from = latest_checkpoint
    
    # Only create a new directory with timestamp if not resuming
    if not args.resume_from:
        output_dir = os.path.join(args.output_dir, f"dialogue_model_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logger
    log_file = os.path.join(output_dir, "training.log")
    logger = setup_logger("train_dialogue", log_file)
    logger.info(f"Output directory: {output_dir}")
    
    # Create configuration objects
    model_config, extension_config = create_model_config_from_args(args)
    training_config = create_training_config_from_args(args)
    cl_config = create_continuous_learning_config_from_args(args) if args.continuous_learning else None
    
    # Override output directory
    training_config.output_dir = output_dir
    
    # Save configurations
    save_config(model_config, output_dir, "model_config.json")
    save_config(training_config, output_dir, "training_config.json")
    if extension_config:
        save_config(extension_config, output_dir, "extension_config.json")
    if cl_config:
        save_config(cl_config, output_dir, "continuous_learning_config.json")
    
    # Set device
    device_str = args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize speaker tokens
    speaker_tokens = init_speaker_tokens(tokenizer)
    
    # Update vocabulary size in model config
    model_config.vocab_size = len(tokenizer)
    
    # Load datasets
    if args.enable_function_calling:
        logger.info("Loading dialogue datasets with function calling capabilities...")
        
        # Load function definitions from file or use defaults
        if args.function_defs_path and os.path.exists(args.function_defs_path):
            with open(args.function_defs_path, 'r') as f:
                function_definitions = json.load(f)
                logger.info(f"Loaded {len(function_definitions)} function definitions from {args.function_defs_path}")
        else:
            function_definitions = get_default_function_definitions()
            logger.info(f"Using {len(function_definitions)} default function definitions")
            
            # Save default function definitions for reference
            func_defs_file = os.path.join(output_dir, "function_definitions.json")
            with open(func_defs_file, 'w') as f:
                json.dump(function_definitions, f, indent=2)
            logger.info(f"Saved default function definitions to {func_defs_file}")
        
        # Log function names
        func_names = [f["name"] for f in function_definitions]
        logger.info(f"Training with functions: {', '.join(func_names)}")
        
        # Get dataloaders with function calling support
        dataloaders = get_function_calling_dataloaders(
            tokenizer=tokenizer,
            function_definitions=function_definitions,
            data_path=args.data_path,
            dataset_name=args.dataset_name,
            batch_size=training_config.batch_size,
            max_length=model_config.max_seq_length,
            speaker_tokens=speaker_tokens,
            system_prompt=args.system_prompt,
            num_workers=args.num_workers,
            cache_dir=args.cache_dir,
            json_format_probability=args.json_format_probability
        )
    else:
        logger.info("Loading standard dialogue datasets...")
        dataloaders = get_dialogue_dataloaders(
            tokenizer=tokenizer,
            data_path=args.data_path,
            dataset_name=args.dataset_name,
            batch_size=training_config.batch_size,
            max_length=model_config.max_seq_length,
            speaker_tokens=speaker_tokens,
            system_prompt=args.system_prompt,
            num_workers=args.num_workers,
            cache_dir=args.cache_dir
        )
    
    # Initialize model
    logger.info("Initializing model...")
    model = SemanticResonanceModelWithExtensions(model_config, extension_config)
    model.to(device)
    
    # If using pretrained embeddings, resize the embedding layer
    # This ensures compatibility with the tokenizer's vocabulary
    if model.encoder.base_embedding.weight.shape[0] != len(tokenizer):
        old_shape = model.encoder.base_embedding.weight.shape[1]
        model.encoder.base_embedding = torch.nn.Embedding(
            len(tokenizer),
            old_shape
        )
        logger.info(f"Resized embedding layer to match tokenizer: {len(tokenizer)}")
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model size: {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["validation"],
        test_dataloader=dataloaders["test"],
        device=device_str,  # Pass the string, not the device object
        output_dir=output_dir,
        max_epochs=training_config.max_epochs,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_steps=training_config.warmup_steps,
        gradient_accumulation_steps=training_config.accumulation_steps,
        eval_steps=training_config.eval_steps,
        save_steps=training_config.save_steps
    )
    
    # Initialize continuous learning manager if enabled
    cl_manager = None
    if args.continuous_learning:
        logger.info("Initializing continuous learning manager...")
        cl_manager = ContinuousLearningManager(
            model=model,
            tokenizer=tokenizer,
            base_dataloader=dataloaders["train"],
            config=cl_config,
            output_dir=os.path.join(output_dir, "continuous_learning")
        )
    
    # Auto-resume from latest checkpoint if requested
    if args.auto_resume and not args.resume_from:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                logger.info(f"Auto-resuming from latest checkpoint: {latest_checkpoint}")
                args.resume_from = latest_checkpoint
            else:
                logger.info("No checkpoints found for auto-resume. Starting fresh training.")
        else:
            logger.info("No checkpoints directory found for auto-resume. Starting fresh training.")
    
    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming from checkpoint {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
        
        # Restore continuous learning state if requested
        if args.resume_continuous_learning and cl_manager:
            cl_dir = os.path.join(os.path.dirname(args.resume_from), "continuous_learning")
            if os.path.exists(cl_dir):
                logger.info(f"Resuming continuous learning state from {cl_dir}")
                cl_manager.load_state(cl_dir)
    
    # Train the model
    logger.info("Starting training...")
    
    # If continuous learning is enabled, use a different training loop
    if args.continuous_learning and cl_manager:
        train_with_continuous_learning(
            trainer=trainer,
            cl_manager=cl_manager,
            dataloaders=dataloaders,
            args=args,
            logger=logger
        )
    else:
        # Standard training
        train_stats = trainer.train()
        
        # Print training statistics
        logger.info("\nTraining completed!")
        logger.info(f"Best validation loss: {train_stats['best_val_loss']:.4f}")
        logger.info(f"Final validation perplexity: {train_stats['val_perplexity']:.2f}")
        
        if "test_perplexity" in train_stats:
            logger.info(f"Test perplexity: {train_stats['test_perplexity']:.2f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    trainer.save_checkpoint(os.path.join(final_model_path, "model.pt"))
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Model saved to {final_model_path}")
    
    # Save continuous learning state if enabled
    if args.continuous_learning and cl_manager:
        cl_manager.save_state()
        logger.info("Continuous learning state saved.")
    
    logger.info("All done!")


def train_with_continuous_learning(trainer, cl_manager, dataloaders, args, logger):
    """
    Train model with continuous learning approach.
    
    This training loop alternates between standard training epochs and updates
    from the continuous learning manager based on feedback.
    
    Args:
        trainer: Trainer instance
        cl_manager: ContinuousLearningManager instance
        dataloaders: Dictionary of dataloaders
        args: Command line arguments
        logger: Logger instance
    """
    best_val_perplexity = float('inf')
    epochs_without_improvement = 0
    max_epochs_without_improvement = 3
    
    # Initial evaluation
    val_metrics = trainer.evaluate(dataloaders["validation"])
    logger.info(f"Initial validation - Loss: {val_metrics['loss']:.4f}, "
               f"Perplexity: {val_metrics['perplexity']:.2f}")
    
    # Record initial performance
    cl_manager.history.record_performance(val_metrics)
    
    # Training loop with continuous learning
    for epoch in range(args.max_epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.max_epochs}")
        
        # Standard training for one epoch
        epoch_stats = trainer.train_epoch(dataloaders["train"])
        
        # Evaluate on validation set
        val_metrics = trainer.evaluate(dataloaders["validation"])
        
        logger.info(f"Epoch {epoch+1} complete - Train loss: {epoch_stats['train_loss']:.4f}, "
                   f"Validation loss: {val_metrics['loss']:.4f}, "
                   f"Validation perplexity: {val_metrics['perplexity']:.2f}")
        
        # Record performance
        cl_manager.history.record_performance(val_metrics)
        
        # Check for improvement
        if val_metrics['perplexity'] < best_val_perplexity:
            improvement = (best_val_perplexity - val_metrics['perplexity']) / best_val_perplexity
            logger.info(f"Validation perplexity improved by {improvement:.2%}")
            
            best_val_perplexity = val_metrics['perplexity']
            epochs_without_improvement = 0
            
            # Save best model
            trainer.save_checkpoint(os.path.join(args.output_dir, "best_model.pt"))
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epochs")
        
        # Check if we should stop training
        if epochs_without_improvement >= max_epochs_without_improvement:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
        
        # Perform continuous learning update
        if epoch > 0:  # Skip first epoch to allow model to stabilize
            logger.info("Performing continuous learning update...")
            
            # Generate some synthetic feedback for demonstration
            # In a real scenario, this would come from user interactions
            synthetic_feedback = generate_synthetic_feedback(
                model=trainer.model,
                tokenizer=trainer.tokenizer,
                dataloader=dataloaders["validation"],
                count=args.min_samples_for_update,
                device=trainer.device
            )
            
            # Add feedback to learning history
            for feedback in synthetic_feedback:
                cl_manager.add_feedback(feedback)
            
            # Update model based on feedback
            update_info = cl_manager.update_model(
                validation_dataloader=dataloaders["validation"],
                max_steps=args.update_steps
            )
            
            if update_info:
                logger.info(f"Update completed - Loss: {update_info.get('loss', 'N/A')}, "
                           f"Improved: {update_info.get('improved', False)}")
        
        # Save continuous learning state
        cl_manager.save_state()
    
    # Final evaluation on test set
    test_metrics = trainer.evaluate(dataloaders["test"])
    logger.info(f"\nFinal test metrics - Loss: {test_metrics['loss']:.4f}, "
               f"Perplexity: {test_metrics['perplexity']:.2f}")
    
    # Get learning statistics
    learning_stats = cl_manager.get_learning_stats()
    logger.info("\nContinuous Learning Statistics:")
    logger.info(f"Total entries: {learning_stats['entries_count']}")
    logger.info(f"Feedback entries: {learning_stats['feedback_count']}")
    logger.info(f"Updates performed: {learning_stats['updates_count']}")
    
    # Record final performance
    cl_manager.history.record_performance(test_metrics)
    cl_manager.save_state()


def generate_synthetic_feedback(model, tokenizer, dataloader, count=10, device="cpu"):
    """
    Generate synthetic feedback for demonstration purposes.
    
    In a real-world scenario, this would be replaced by actual user feedback.
    
    Args:
        model: The model to generate responses
        tokenizer: Tokenizer for processing text
        dataloader: DataLoader to sample examples from
        count: Number of feedback items to generate
        device: Device to run inference on
        
    Returns:
        List of synthetic feedback data
    """
    model.eval()
    feedback = []
    
    # Sample batches from dataloader
    sampled_examples = []
    for batch in dataloader:
        batch_size = len(batch["input_ids"])
        for i in range(batch_size):
            example = {
                "input_ids": batch["input_ids"][i],
                "attention_mask": batch["attention_mask"][i],
                "labels": batch["labels"][i]
            }
            sampled_examples.append(example)
            if len(sampled_examples) >= count:
                break
        if len(sampled_examples) >= count:
            break
    
    # Generate synthetic feedback for each example
    for example in sampled_examples[:count]:
        # Convert example to conversation format
        conversation = []
        
        # Get the input text
        input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=False)
        
        # Extract turns based on special tokens
        turns = []
        current_role = None
        current_content = ""
        
        for token in input_text.split():
            if token in ["<|system|>", "<|user|>", "<|assistant|>"]:
                # Save previous turn if exists
                if current_role and current_content.strip():
                    turns.append({"role": current_role, "content": current_content.strip()})
                    current_content = ""
                
                # Update current role
                if token == "<|system|>":
                    current_role = "system"
                elif token == "<|user|>":
                    current_role = "user"
                elif token == "<|assistant|>":
                    current_role = "assistant"
            elif token == "<|end|>":
                # Save completed turn
                if current_role and current_content.strip():
                    turns.append({"role": current_role, "content": current_content.strip()})
                    current_content = ""
                current_role = None
            elif current_role:
                current_content += " " + token
        
        # Create synthetic feedback by modifying the assistant responses
        feedback_turns = []
        for turn in turns:
            if turn["role"] == "assistant":
                # Create a slightly modified version as "feedback"
                content = turn["content"]
                improved_content = content + " [additional synthetic response]"
                feedback_turns.append({"role": turn["role"], "content": improved_content})
            else:
                feedback_turns.append(turn)
        
        feedback.append({
            "conversations": feedback_turns,
            "original": turns,
            "metadata": {"synthetic": True}
        })
    
    return feedback


if __name__ == "__main__":
    main()