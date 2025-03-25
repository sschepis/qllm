#!/usr/bin/env python3
"""
Main entry point for the Semantic Resonance Language Model.

This script provides a unified interface for training, evaluating,
compressing, and generating text with the model.
"""

import os
import sys
import argparse
import json
import torch
from transformers import AutoTokenizer

from src.config import ModelConfig, TrainingConfig, DataConfig, get_default_configs
from src.model.semantic_resonance_model import SemanticResonanceModel
from src.data.wikitext_dataset import get_wikitext_dataloaders
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model
from src.utils.compression import compress_model, compare_models, load_compressed_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Semantic Resonance Language Model")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval", "compress", "generate"],
                        help="Operation mode")
    
    # Common arguments
    parser.add_argument("--config_dir", type=str, default=None, 
                        help="Directory containing configuration files")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint or directory")
    parser.add_argument("--output_dir", type=str, default="runs/semantic_resonance", 
                        help="Output directory")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (default: cuda if available)")
    
    # Training arguments (used with --mode=train)
    parser.add_argument("--resume", action="store_true", 
                        help="Resume training from checkpoint")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum epochs")
    
    # Model configuration (used with --mode=train if no config_dir)
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--primes", type=int, nargs="+", default=[7, 11, 13, 17, 19], 
                       help="Prime numbers for subspace decomposition")
    
    # Evaluation arguments (used with --mode=eval)
    parser.add_argument("--eval_split", type=str, default="validation",
                        choices=["train", "validation", "test"],
                        help="Dataset split to evaluate on")
    
    # Compression arguments (used with --mode=compress)
    parser.add_argument("--compression_threshold", type=float, default=0.8, 
                        help="Threshold for compression")
    parser.add_argument("--compression_method", type=str, default="both",
                        choices=["mask", "prune", "both"],
                        help="Compression method")
    
    # Text generation arguments (used with --mode=generate)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation")
    parser.add_argument("--max_length", type=int, default=100, 
                        help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k filtering parameter")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling parameter")
    
    return parser.parse_args()


def load_configs(config_dir):
    """
    Load configuration from files.
    
    Args:
        config_dir (str): Directory containing configuration files
        
    Returns:
        tuple: (model_config, training_config, data_config)
    """
    # Load model configuration
    with open(os.path.join(config_dir, "model_config.json"), 'r') as f:
        model_config_dict = json.load(f)
    
    # Load training configuration
    with open(os.path.join(config_dir, "training_config.json"), 'r') as f:
        training_config_dict = json.load(f)
    
    # Load data configuration
    with open(os.path.join(config_dir, "data_config.json"), 'r') as f:
        data_config_dict = json.load(f)
    
    # Create configuration objects
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    
    # Set attributes from loaded configs
    for key, value in model_config_dict.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
    
    for key, value in training_config_dict.items():
        if hasattr(training_config, key):
            setattr(training_config, key, value)
    
    for key, value in data_config_dict.items():
        if hasattr(data_config, key):
            setattr(data_config, key, value)
    
    return model_config, training_config, data_config


def create_default_configs(args):
    """
    Create default configurations from command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (model_config, training_config, data_config)
    """
    # Get default configurations
    configs = get_default_configs()
    model_config = configs["model"]
    training_config = configs["training"]
    data_config = configs["data"]
    
    # Update configurations from arguments
    if args.hidden_dim:
        model_config.hidden_dim = args.hidden_dim
    
    if args.num_layers:
        model_config.num_layers = args.num_layers
    
    if args.num_heads:
        model_config.num_heads = args.num_heads
    
    if args.primes:
        model_config.primes = args.primes
    
    if args.batch_size:
        training_config.batch_size = args.batch_size
    
    if args.learning_rate:
        training_config.learning_rate = args.learning_rate
    
    if args.max_epochs:
        training_config.max_epochs = args.max_epochs
    
    if args.output_dir:
        training_config.output_dir = args.output_dir
    
    # Set device
    if args.device:
        training_config.device = args.device
    else:
        training_config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return model_config, training_config, data_config


def get_tokenizer(tokenizer_name):
    """
    Get a tokenizer for the specified name.
    
    Args:
        tokenizer_name (str): Name of the tokenizer
        
    Returns:
        transformers.PreTrainedTokenizer: Tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_path, config=None, device=None):
    """
    Load a model from the specified path.
    
    Args:
        model_path (str): Path to the model
        config (ModelConfig, optional): Model configuration
        device (torch.device, optional): Device to load the model on
        
    Returns:
        SemanticResonanceModel: Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if model_path is a directory or a file
    if os.path.isdir(model_path):
        # Load model from directory
        model = SemanticResonanceModel.from_pretrained(model_path)
    else:
        # Load model from checkpoint
        if config is None:
            raise ValueError("Config must be provided when loading from checkpoint")
        
        model = SemanticResonanceModel(config)
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if checkpoint contains model state dict directly or within a dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    return model


def train_model(args, model_config, training_config, data_config):
    """
    Train a model with the specified configurations.
    
    Args:
        args: Command line arguments
        model_config: Model configuration
        training_config: Training configuration
        data_config: Data configuration
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configurations
    with open(os.path.join(args.output_dir, "model_config.json"), 'w') as f:
        json.dump({k: v for k, v in model_config.__dict__.items() 
                  if not k.startswith('_') and not callable(v)}, f, indent=2)
    
    with open(os.path.join(args.output_dir, "training_config.json"), 'w') as f:
        config_dict = {k: v for k, v in training_config.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        # Convert device to string
        if 'device' in config_dict and isinstance(config_dict['device'], torch.device):
            config_dict['device'] = str(config_dict['device'])
        json.dump(config_dict, f, indent=2)
    
    with open(os.path.join(args.output_dir, "data_config.json"), 'w') as f:
        json.dump({k: v for k, v in data_config.__dict__.items() 
                  if not k.startswith('_') and not callable(v)}, f, indent=2)
    
    # Set device
    device = torch.device(training_config.device)
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(data_config.tokenizer_name)
    
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
    if args.resume and args.model_path:
        print(f"Loading model from {args.model_path}")
        model = load_model(args.model_path, model_config, device)
    else:
        print("Initializing new model...")
        model = SemanticResonanceModel(model_config)
        model.to(device)
    
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
    if args.resume and args.model_path:
        print(f"Resuming training from checkpoint {args.model_path}")
        if os.path.isdir(args.model_path):
            checkpoint_path = os.path.join(args.model_path, "checkpoints/best_model.pt")
            if os.path.exists(checkpoint_path):
                trainer.load_checkpoint(checkpoint_path)
            else:
                print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
        else:
            trainer.load_checkpoint(args.model_path)
    
    # Train the model
    print("Starting training...")
    train_stats = trainer.train()
    
    # Print training statistics
    print("\nTraining completed!")
    print(f"Best validation loss: {train_stats['best_val_loss']:.4f}")
    print(f"Final validation perplexity: {train_stats['val_perplexity']:.2f}")
    
    if "test_perplexity" in train_stats:
        print(f"Test perplexity: {train_stats['test_perplexity']:.2f}")
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Model and tokenizer saved to {args.output_dir}")


def evaluate_loaded_model(args, model, data_config):
    """
    Evaluate a loaded model.
    
    Args:
        args: Command line arguments
        model: Loaded model
        data_config: Data configuration
        
    Returns:
        dict: Evaluation metrics
    """
    # Set device
    device = torch.device(args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(data_config.tokenizer_name)
    
    # Load datasets
    print("Loading dataset...")
    dataloaders = get_wikitext_dataloaders(
        tokenizer=tokenizer,
        batch_size=args.batch_size if args.batch_size else 16,
        max_length=data_config.max_length,
        stride=data_config.stride,
        num_workers=data_config.preprocessing_num_workers,
        cache_dir=data_config.cache_dir
    )
    
    # Evaluate model
    print(f"Evaluating model on {args.eval_split} split...")
    metrics = evaluate_model(model, dataloaders[args.eval_split], device)
    
    # Print metrics
    print("\nEvaluation results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    return metrics


def compress_loaded_model(args, model, model_config):
    """
    Compress a loaded model and save it.
    
    Args:
        args: Command line arguments
        model: Loaded model
        model_config: Model configuration
        
    Returns:
        SemanticResonanceModel: Compressed model
    """
    print("\nApplying model compression...")
    compression_config = {
        "method": args.compression_method,
        "primes": model_config.primes,
        "threshold": args.compression_threshold,
        "mask_type": "mod"
    }
    
    compressed_model, compression_ratio = compress_model(model, compression_config)
    
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Compare models
    comparison = compare_models(model, compressed_model)
    for key, value in comparison.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Save compressed model
    compressed_path = os.path.join(args.output_dir, "compressed_model")
    os.makedirs(compressed_path, exist_ok=True)
    
    compressed_model.save_pretrained(compressed_path)
    
    print(f"Compressed model saved to {compressed_path}")
    
    return compressed_model


def generate_text(args, model, tokenizer=None):
    """
    Generate text with the specified model.
    
    Args:
        args: Command line arguments
        model: Loaded model
        tokenizer: Tokenizer (if None, will be loaded from model path)
    """
    # Set device
    device = torch.device(args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load tokenizer if not provided
    if tokenizer is None:
        if os.path.isdir(args.model_path):
            tokenizer_path = args.model_path
        else:
            tokenizer_path = os.path.dirname(args.model_path)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except:
            # If tokenizer not found, use default
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Get prompt
    if args.prompt is None:
        prompt = input("Enter a prompt: ")
    else:
        prompt = args.prompt
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    print(f"\nGenerating text with temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}...")
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=1.2,
        )
    
    # Decode output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("\nGenerated text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)


def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load or create configurations
    if args.config_dir and os.path.exists(args.config_dir):
        model_config, training_config, data_config = load_configs(args.config_dir)
    else:
        model_config, training_config, data_config = create_default_configs(args)
    
    # Execute according to the specified mode
    if args.mode == "train":
        train_model(args, model_config, training_config, data_config)
    
    elif args.mode == "eval":
        if args.model_path is None:
            print("Error: --model_path must be specified for evaluation")
            sys.exit(1)
        
        # Load model
        model = load_model(args.model_path, model_config, device)
        
        # Evaluate model
        evaluate_loaded_model(args, model, data_config)
    
    elif args.mode == "compress":
        if args.model_path is None:
            print("Error: --model_path must be specified for compression")
            sys.exit(1)
        
        # Load model
        model = load_model(args.model_path, model_config, device)
        
        # Compress model
        compressed_model = compress_loaded_model(args, model, model_config)
        
        # Save tokenizer with compressed model
        if os.path.isdir(args.model_path):
            tokenizer_path = args.model_path
        else:
            tokenizer_path = os.path.dirname(args.model_path)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer.save_pretrained(os.path.join(args.output_dir, "compressed_model"))
        except:
            print("Warning: Could not find tokenizer to save with compressed model")
    
    elif args.mode == "generate":
        if args.model_path is None:
            print("Error: --model_path must be specified for text generation")
            sys.exit(1)
        
        # Load model
        model = load_model(args.model_path, model_config, device)
        
        # Load tokenizer
        if os.path.isdir(args.model_path):
            tokenizer_path = args.model_path
        else:
            tokenizer_path = os.path.dirname(args.model_path)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except:
            # If tokenizer not found, use default
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            print("Warning: Using default tokenizer as none was found with the model")
        
        # Generate text
        generate_text(args, model, tokenizer)
    
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        sys.exit(1)


if __name__ == "__main__":
    main()