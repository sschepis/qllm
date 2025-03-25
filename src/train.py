#!/usr/bin/env python3
"""
Main training script for Semantic Resonance Language Model.

This script handles the complete training pipeline, including data loading,
model initialization, training, evaluation, and model saving.
"""

import os
import argparse
import json
import torch
from transformers import AutoTokenizer

from src.config import ModelConfig, TrainingConfig, DataConfig, get_default_configs
from src.model.semantic_resonance_model import SemanticResonanceModel
from src.data.wikitext_dataset import get_wikitext_dataloaders
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model
from src.utils.compression import compress_model, compare_models


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Semantic Resonance Language Model")
    
    # Model configuration
    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
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
    
    # HCW settings
    parser.add_argument("--enable_hcw", action="store_true", 
                       help="Enable Homomorphic Computational Wrapper")
    parser.add_argument("--memory_size", type=int, default=1000, 
                       help="Memory size for HCW")
    parser.add_argument("--memory_key_dim", type=int, default=128, 
                       help="Memory key dimension for HCW")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum epochs")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # Data configuration
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Tokenizer name")
    parser.add_argument("--dataset_name", type=str, default="wikitext-103-raw-v1", help="Dataset name")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Cache directory")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="runs/semantic_resonance", 
                       help="Output directory")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    
    # Compression
    parser.add_argument("--compress_model", action="store_true", 
                       help="Apply compression after training")
    parser.add_argument("--compression_threshold", type=float, default=0.8, 
                       help="Threshold for compression")
    
    # Hardware configuration
    parser.add_argument("--device", type=str, default=None, 
                       help="Device to train on (default: cuda if available)")
    parser.add_argument("--num_workers", type=int, default=4, 
                       help="Number of workers for data loading")
    
    # Resuming training
    parser.add_argument("--resume_from", type=str, default=None, 
                       help="Path to checkpoint to resume training from")
    
    return parser.parse_args()


def create_config_from_args(args):
    """Create configuration objects from command line arguments."""
    # Create model config
    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout,
        primes=args.primes,
        base_dim=args.base_dim,
        max_iterations=args.max_iterations,
        entropy_threshold=args.entropy_threshold,
        use_prime_mask=args.use_prime_mask,
        enable_hcw=args.enable_hcw,
        memory_size=args.memory_size,
        memory_key_dim=args.memory_key_dim
    )
    
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
    
    # Create data config
    data_config = DataConfig(
        dataset_name=args.dataset_name,
        tokenizer_name=args.tokenizer_name,
        cache_dir=args.cache_dir,
        preprocessing_num_workers=args.num_workers
    )
    
    # Add output directory to training config
    training_config.output_dir = args.output_dir
    
    return model_config, training_config, data_config


def save_config(config, output_dir, filename="config.json"):
    """Save configuration to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    config_dict = {k: v for k, v in config.__dict__.items() 
                  if not k.startswith('_') and not callable(v)}
    
    # Convert any non-serializable types
    for key, value in config_dict.items():
        if isinstance(value, torch.device):
            config_dict[key] = str(value)
    
    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(config_dict, f, indent=2)


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    
    # Create configuration objects
    model_config, training_config, data_config = create_config_from_args(args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configurations
    save_config(model_config, args.output_dir, "model_config.json")
    save_config(training_config, args.output_dir, "training_config.json")
    save_config(data_config, args.output_dir, "data_config.json")
    
    # Set device
    device = torch.device(training_config.device)
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Update vocabulary size in model config
    model_config.vocab_size = len(tokenizer)
    
    # Load datasets
    print("Loading datasets...")
    dataloaders = get_wikitext_dataloaders(
        tokenizer=tokenizer,
        batch_size=training_config.batch_size,
        max_length=data_config.max_length,
        stride=data_config.stride,
        num_workers=data_config.preprocessing_num_workers,
        cache_dir=data_config.cache_dir
    )
    
    # Initialize model
    print("Initializing model...")
    model = SemanticResonanceModel(model_config)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["validation"],
        test_dataloader=dataloaders["test"],
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming from checkpoint {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Train the model
    print("Starting training...")
    train_stats = trainer.train()
    
    # Print training statistics
    print("\nTraining completed!")
    print(f"Best validation loss: {train_stats['best_val_loss']:.4f}")
    print(f"Final validation perplexity: {train_stats['val_perplexity']:.2f}")
    
    if "test_perplexity" in train_stats:
        print(f"Test perplexity: {train_stats['test_perplexity']:.2f}")
    
    # Apply compression if specified
    if args.compress_model:
        print("\nApplying model compression...")
        compression_config = {
            "method": "both",
            "primes": model_config.primes,
            "threshold": args.compression_threshold,
            "mask_type": "mod"
        }
        
        compressed_model, compression_ratio = compress_model(model, compression_config)
        
        print(f"Compression ratio: {compression_ratio:.2f}x")
        
        # Compare models
        comparison = compare_models(model, compressed_model)
        for key, value in comparison.items():
            print(f"{key}: {value}")
        
        # Evaluate compressed model
        print("\nEvaluating compressed model...")
        compressed_metrics = evaluate_model(
            compressed_model, 
            dataloaders["validation"],
            device
        )
        
        print(f"Compressed model validation perplexity: {compressed_metrics['perplexity']:.2f}")
        
        # Save compressed model
        compressed_path = os.path.join(args.output_dir, "compressed_model")
        os.makedirs(compressed_path, exist_ok=True)
        
        compressed_model.save_pretrained(compressed_path)
        tokenizer.save_pretrained(compressed_path)
        
        print(f"Compressed model saved to {compressed_path}")


if __name__ == "__main__":
    main()