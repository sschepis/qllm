"""
Command line argument parsing for the Quantum Resonance Language Model.
Provides a unified interface for parsing arguments for training, evaluation,
compression, and generation.
"""

import os
import argparse
from typing import Optional, Dict, Any, List, Tuple


def create_base_parser() -> argparse.ArgumentParser:
    """
    Create a base argument parser with common arguments.
    
    Returns:
        argparse.ArgumentParser: Base parser
    """
    parser = argparse.ArgumentParser(
        description="Quantum Resonance Language Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic configuration
    parser.add_argument("--config_dir", type=str, default=None,
                        help="Directory containing configuration files")
    parser.add_argument("--output_dir", type=str, default="runs/quantum_resonance",
                        help="Output directory for model and logs")
    
    # Mode and device settings
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval", "compress", "generate"],
                        help="Operation mode")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: cuda if available)")
    
    # Checkpoint handling
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--ignore_checkpoints", action="store_true",
                        help="Ignore existing checkpoints and start fresh")
    
    # Logging settings
    parser.add_argument("--log_level", type=str, default="info",
                        choices=["debug", "verbose", "info", "warning", "error"],
                        help="Logging level")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file (defaults to output_dir/training.log)")
    
    return parser


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add model architecture arguments to a parser.
    
    Args:
        parser: Parser to add arguments to
    """
    model_group = parser.add_argument_group("Model Configuration")
    
    # Core architecture
    model_group.add_argument("--hidden_dim", type=int, default=768,
                            help="Size of hidden dimensions")
    model_group.add_argument("--num_layers", type=int, default=4,
                            help="Number of transformer layers")
    model_group.add_argument("--num_heads", type=int, default=12,
                            help="Number of attention heads")
    model_group.add_argument("--ff_dim", type=int, default=3072,
                            help="Size of feedforward layer")
    
    # Resonance settings
    model_group.add_argument("--max_iterations", type=int, default=10,
                            help="Maximum number of resonance iterations")
    model_group.add_argument("--resonance_epsilon", type=float, default=0.1,
                            help="Convergence threshold for entropy")
    model_group.add_argument("--resonance_momentum", type=float, default=0.2,
                            help="Momentum for resonance updates")
    
    # Phase modulation
    model_group.add_argument("--use_phase_modulation", type=bool, default=True,
                            help="Enable phase modulation in resonance attention")
    model_group.add_argument("--phase_factor", type=float, default=0.1,
                            help="Scaling factor for phase modulation")
    
    # Quantum properties
    model_group.add_argument("--primes", type=int, nargs="+", 
                            default=[7, 11, 13, 17, 19],
                            help="Prime numbers for quantum subspace decomposition")
    
    # Regularization
    model_group.add_argument("--dropout", type=float, default=0.1,
                            help="Dropout probability")
    model_group.add_argument("--attention_dropout", type=float, default=0.1,
                            help="Attention dropout probability")


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add training arguments to a parser.
    
    Args:
        parser: Parser to add arguments to
    """
    train_group = parser.add_argument_group("Training Configuration")
    
    # Basic training settings
    train_group.add_argument("--batch_size", type=int, default=32,
                            help="Training batch size")
    train_group.add_argument("--eval_batch_size", type=int, default=32,
                            help="Evaluation batch size")
    train_group.add_argument("--max_epochs", type=int, default=10,
                            help="Maximum number of training epochs")
    train_group.add_argument("--seed", type=int, default=42,
                            help="Random seed for reproducibility")
    
    # Optimization settings
    train_group.add_argument("--learning_rate", type=float, default=5e-5,
                            help="Peak learning rate for training")
    train_group.add_argument("--weight_decay", type=float, default=0.01,
                            help="Weight decay coefficient")
    train_group.add_argument("--warmup_steps", type=int, default=1000,
                            help="Learning rate warmup steps")
    train_group.add_argument("--max_grad_norm", type=float, default=1.0,
                            help="Gradient clipping norm")
    
    # Trainer selection
    train_group.add_argument("--trainer_type", type=str, default="enhanced",
                            choices=["standard", "dialogue", "enhanced", "base"],
                            help="Type of trainer to use")
    
    # Training schedule
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=1,
                            help="Number of steps to accumulate gradients")
    train_group.add_argument("--eval_steps", type=int, default=500,
                            help="Steps between evaluations")
    train_group.add_argument("--save_steps", type=int, default=1000,
                            help="Steps between model saves")
    
    # Mixed precision
    train_group.add_argument("--mixed_precision", action="store_true",
                            help="Enable mixed precision training")
    
    # Checkpointing
    train_group.add_argument("--save_total_limit", type=int, default=3,
                            help="Maximum number of checkpoints to keep")
    train_group.add_argument("--disable_optimizer_saving", action="store_true",
                            help="Don't save optimizer state in checkpoints")
    train_group.add_argument("--save_every_epoch", action="store_true",
                            help="Save checkpoint after every epoch")
    
    # Early stopping
    train_group.add_argument("--early_stopping_patience", type=int, default=3,
                            help="Patience for early stopping")
    train_group.add_argument("--early_stopping_threshold", type=float, default=0.01,
                            help="Threshold for early stopping improvement")


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add data processing arguments to a parser.
    
    Args:
        parser: Parser to add arguments to
    """
    data_group = parser.add_argument_group("Data Configuration")
    
    # Dataset settings
    data_group.add_argument("--dataset_name", type=str, default="wikitext",
                           help="Dataset name or path")
    data_group.add_argument("--dataset_config_name", type=str, default="wikitext-103-raw-v1",
                           help="Dataset configuration name")
    data_group.add_argument("--cache_dir", type=str, default=".cache",
                           help="Cache directory for datasets and models")
    
    # Input files
    data_group.add_argument("--train_file", type=str, default=None,
                           help="Path to training data file")
    data_group.add_argument("--validation_file", type=str, default=None,
                           help="Path to validation data file")
    data_group.add_argument("--test_file", type=str, default=None,
                           help="Path to test data file")
    
    # Tokenization settings
    data_group.add_argument("--tokenizer_name", type=str, default="gpt2",
                           help="Tokenizer name or path")
    data_group.add_argument("--max_length", type=int, default=512,
                           help="Maximum sequence length")
    data_group.add_argument("--stride", type=int, default=128,
                           help="Stride for sliding window tokenization")
    
    # Preprocessing settings
    data_group.add_argument("--preprocessing_num_workers", type=int, default=4,
                           help="Number of preprocessing workers")
    data_group.add_argument("--overwrite_cache", action="store_true",
                           help="Overwrite the cached datasets")


def add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add evaluation arguments to a parser.
    
    Args:
        parser: Parser to add arguments to
    """
    eval_group = parser.add_argument_group("Evaluation Configuration")
    
    eval_group.add_argument("--eval_split", type=str, default="validation",
                           choices=["train", "validation", "test"],
                           help="Dataset split to evaluate on")
    eval_group.add_argument("--metrics", type=str, nargs="+",
                           default=["loss", "perplexity", "accuracy"],
                           help="Metrics to evaluate")
    eval_group.add_argument("--compute_full_metrics", action="store_true",
                           help="Compute extended metrics (slower)")


def add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add text generation arguments to a parser.
    
    Args:
        parser: Parser to add arguments to
    """
    gen_group = parser.add_argument_group("Generation Configuration")
    
    gen_group.add_argument("--prompt", type=str, default=None,
                          help="Text prompt for generation")
    gen_group.add_argument("--max_length", type=int, default=200,
                          help="Maximum generation length")
    gen_group.add_argument("--min_length", type=int, default=10,
                          help="Minimum generation length")
    gen_group.add_argument("--temperature", type=float, default=0.7,
                          help="Sampling temperature")
    gen_group.add_argument("--top_k", type=int, default=50,
                          help="Top-k sampling parameter")
    gen_group.add_argument("--top_p", type=float, default=0.9,
                          help="Nucleus sampling parameter")
    gen_group.add_argument("--repetition_penalty", type=float, default=1.2,
                          help="Repetition penalty")
    gen_group.add_argument("--num_return_sequences", type=int, default=1,
                          help="Number of sequences to generate")


def add_compression_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add model compression arguments to a parser.
    
    Args:
        parser: Parser to add arguments to
    """
    comp_group = parser.add_argument_group("Compression Configuration")
    
    comp_group.add_argument("--compression_method", type=str, default="both",
                           choices=["mask", "prune", "quantize", "both"],
                           help="Compression method to use")
    comp_group.add_argument("--compression_threshold", type=float, default=0.8,
                           help="Threshold for compression")
    comp_group.add_argument("--mask_type", type=str, default="mod",
                           choices=["mod", "topk", "random"],
                           help="Type of masking to apply")
    comp_group.add_argument("--compare_performance", action="store_true",
                           help="Compare performance before and after compression")


def create_parser_for_mode(mode: str) -> argparse.ArgumentParser:
    """
    Create a parser with arguments specific to a mode.
    
    Args:
        mode: Mode to create parser for ("train", "eval", "compress", "generate")
        
    Returns:
        argparse.ArgumentParser: Mode-specific parser
    """
    parser = create_base_parser()
    
    # Add arguments common to all modes
    add_model_arguments(parser)
    
    # Add mode-specific arguments
    if mode == "train":
        add_training_arguments(parser)
        add_data_arguments(parser)
    elif mode == "eval":
        add_evaluation_arguments(parser)
        add_data_arguments(parser)
    elif mode == "compress":
        add_compression_arguments(parser)
    elif mode == "generate":
        add_generation_arguments(parser)
    
    return parser


def infer_mode_from_args() -> str:
    """
    Infer the mode from command line arguments.
    
    Returns:
        str: Inferred mode
    """
    import sys
    
    # Check if mode is explicitly specified
    for i, arg in enumerate(sys.argv):
        if arg == "--mode" and i+1 < len(sys.argv):
            return sys.argv[i+1]
        elif arg.startswith("--mode="):
            return arg.split("=")[1]
    
    # Try to infer from script name
    script_name = os.path.basename(sys.argv[0]).lower()
    if "train" in script_name:
        return "train"
    elif "eval" in script_name:
        return "eval"
    elif "compress" in script_name:
        return "compress"
    elif "generate" in script_name:
        return "generate"
    
    # Default to train
    return "train"


def parse_args(mode: Optional[str] = None) -> argparse.Namespace:
    """
    Parse command line arguments for the specified mode.
    
    Args:
        mode: Mode to parse arguments for (inferred if None)
        
    Returns:
        argparse.Namespace: Parsed arguments
    """
    if mode is None:
        mode = infer_mode_from_args()
    
    parser = create_parser_for_mode(mode)
    return parser.parse_args()


def extract_config_args(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Extract model, training and data arguments from parsed arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Tuple: (model_args, training_args, data_args)
    """
    # Create dictionaries for each config type
    model_args = {}
    training_args = {}
    data_args = {}
    
    # Define which arguments belong to which config
    model_arg_names = {
        "hidden_dim", "num_layers", "num_heads", "ff_dim", "max_iterations",
        "resonance_epsilon", "resonance_momentum", "use_phase_modulation",
        "phase_factor", "primes", "dropout", "attention_dropout"
    }
    
    training_arg_names = {
        "batch_size", "eval_batch_size", "max_epochs", "seed", "learning_rate",
        "weight_decay", "warmup_steps", "max_grad_norm", "gradient_accumulation_steps",
        "eval_steps", "save_steps", "mixed_precision", "save_total_limit",
        "disable_optimizer_saving", "save_every_epoch", "early_stopping_patience",
        "early_stopping_threshold", "output_dir", "device", "trainer_type"
    }
    
    data_arg_names = {
        "dataset_name", "dataset_config_name", "cache_dir", "train_file",
        "validation_file", "test_file", "tokenizer_name", "max_length",
        "stride", "preprocessing_num_workers", "overwrite_cache"
    }
    
    # Extract arguments
    for key, value in vars(args).items():
        if key in model_arg_names:
            model_args[key] = value
        elif key in training_arg_names:
            training_args[key] = value
        elif key in data_arg_names:
            data_args[key] = value
    
    return model_args, training_args, data_args


def get_log_level(level_str: str) -> int:
    """
    Convert a log level string to the corresponding logging level.
    
    Args:
        level_str: Log level string
        
    Returns:
        int: Logging level
    """
    import logging
    
    levels = {
        "debug": logging.DEBUG,
        "verbose": 15,  # Custom level between DEBUG and INFO
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR
    }
    
    return levels.get(level_str.lower(), logging.INFO)