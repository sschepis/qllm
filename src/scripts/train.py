#!/usr/bin/env python
"""
Training script for QLLM models with extension support.

This script provides a comprehensive training pipeline for QLLM models,
supporting dialogue capabilities, function calling, memory extensions,
and multimodal operations. It can train both mini-sized and large models.
"""

import os
import sys
import argparse
import logging
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Union

# Add the project root directory to the Python path so we can import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import torch
import numpy as np
from torch.utils.data import DataLoader

# Import configuration modules
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.config.data_config import DataConfig
from src.model.extensions.extension_config import (
    ExtensionConfig,
    MultimodalConfig,
    MemoryConfig,
    QuantumConfig
)

# Import training modules
from src.training.unified_trainer import UnifiedTrainer
from src.training.trainers.dialogue_trainer import DialogueTrainer
from src.training.trainers.function_call_trainer import FunctionCallTrainer
from src.training.trainer_factory import TrainerFactory

# Import data modules
from src.data import (
    create_text_dataloader,
    create_dialogue_dataloader,
    create_function_calling_dataloader,
    create_dataloader_from_config
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("qllm_train.log")
    ]
)
logger = logging.getLogger("qllm.train")


def parse_arguments():
    """Parse command line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train a QLLM model with extensions")
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model-size", 
        type=str, 
        default="mini", 
        choices=["mini", "small", "medium", "large", "xlarge"],
        help="Size of the model to train"
    )
    model_group.add_argument(
        "--model-type", 
        type=str, 
        default="semantic_resonance_with_extensions",
        choices=["semantic_resonance", "semantic_resonance_with_extensions"],
        help="Type of model to train"
    )
    model_group.add_argument(
        "--vocab-size", 
        type=int, 
        default=50257,
        help="Vocabulary size for the model"
    )
    model_group.add_argument(
        "--max-seq-length", 
        type=int, 
        default=1024,
        help="Maximum sequence length for the model"
    )
    model_group.add_argument(
        "--checkpoint-path", 
        type=str, 
        default=None,
        help="Path to a checkpoint to load the model from"
    )
    
    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--training-mode", 
        type=str, 
        default="unified",
        choices=["text", "dialogue", "function_call", "unified"],
        help="Training mode to use"
    )
    train_group.add_argument(
        "--batch-size", 
        type=int, 
        default=16,
        help="Batch size for training"
    )
    train_group.add_argument(
        "--eval-batch-size", 
        type=int, 
        default=8,
        help="Batch size for evaluation"
    )
    train_group.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of epochs to train for"
    )
    train_group.add_argument(
        "--learning-rate", 
        type=float, 
        default=5e-5,
        help="Learning rate for training"
    )
    train_group.add_argument(
        "--weight-decay", 
        type=float, 
        default=0.01,
        help="Weight decay for training"
    )
    train_group.add_argument(
        "--warmup-steps", 
        type=int, 
        default=500,
        help="Number of warmup steps for the learning rate scheduler"
    )
    train_group.add_argument(
        "--gradient-accumulation-steps", 
        type=int, 
        default=1,
        help="Number of gradient accumulation steps"
    )
    train_group.add_argument(
        "--save-steps", 
        type=int, 
        default=1000,
        help="Save checkpoint every X steps"
    )
    train_group.add_argument(
        "--eval-steps", 
        type=int, 
        default=1000,
        help="Evaluate every X steps"
    )
    train_group.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    train_group.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use for training (auto, cuda, cpu, mps)"
    )
    train_group.add_argument(
        "--fp16", 
        action="store_true",
        help="Use mixed precision training"
    )
    train_group.add_argument(
        "--output-dir", 
        type=str, 
        default="runs/qllm",
        help="Directory to save checkpoints and logs"
    )
    
    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--train-data", 
        type=str, 
        required=True,
        help="Path to training data file or directory"
    )
    data_group.add_argument(
        "--eval-data", 
        type=str, 
        default=None,
        help="Path to evaluation data file or directory"
    )
    data_group.add_argument(
        "--dataset-type", 
        type=str, 
        default="auto",
        choices=["auto", "text", "dialogue", "function_calling"],
        help="Type of dataset to use"
    )
    data_group.add_argument(
        "--max-history-turns", 
        type=int, 
        default=3,
        help="Maximum number of history turns for dialogue data"
    )
    data_group.add_argument(
        "--function-schema-path", 
        type=str, 
        default=None,
        help="Path to function schema JSON file for function calling"
    )
    
    # Extension configuration
    ext_group = parser.add_argument_group("Extension Configuration")
    ext_group.add_argument(
        "--enable-extensions", 
        action="store_true",
        help="Enable model extensions"
    )
    ext_group.add_argument(
        "--enable-memory", 
        action="store_true",
        help="Enable memory extension"
    )
    ext_group.add_argument(
        "--enable-multimodal", 
        action="store_true",
        help="Enable multimodal extension"
    )
    ext_group.add_argument(
        "--enable-quantum", 
        action="store_true",
        help="Enable quantum extension"
    )
    ext_group.add_argument(
        "--disable-memory",
        action="store_true",
        help="Disable memory extension when using semantic_resonance_with_extensions model"
    )
    ext_group.add_argument(
        "--disable-multimodal",
        action="store_true",
        help="Disable multimodal extension when using semantic_resonance_with_extensions model"
    )
    ext_group.add_argument(
        "--disable-quantum",
        action="store_true",
        help="Disable quantum extension when using semantic_resonance_with_extensions model"
    )
    ext_group.add_argument(
        "--extension-config",
        type=str,
        default=None,
        help="Path to extension configuration JSON file"
    )
    
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_config: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        device_config: Device configuration string
        
    Returns:
        PyTorch device
    """
    # Use specified device if provided
    if device_config is not None and device_config != "auto":
        return torch.device(device_config)
    
    # Auto-detect device
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_model_config(args) -> ModelConfig:
    """
    Create a model configuration from arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        ModelConfig: Model configuration
    """
    # Set model dimensions based on model size
    model_dims = {
        "mini": {
            "hidden_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "intermediate_dim": 3072
        },
        "small": {
            "hidden_dim": 1024,
            "num_layers": 16,
            "num_heads": 16,
            "intermediate_dim": 4096
        },
        "medium": {
            "hidden_dim": 1280,
            "num_layers": 24,
            "num_heads": 20,
            "intermediate_dim": 5120
        },
        "large": {
            "hidden_dim": 1600,
            "num_layers": 32,
            "num_heads": 25,
            "intermediate_dim": 6400
        },
        "xlarge": {
            "hidden_dim": 2048,
            "num_layers": 40,
            "num_heads": 32,
            "intermediate_dim": 8192
        }
    }
    
    # Get dimensions for selected model size
    dims = model_dims.get(args.model_size, model_dims["mini"])
    
    # Create model config
    config_dict = {
        "model_type": args.model_type,
        "model_size": args.model_size,
        "vocab_size": args.vocab_size,
        "max_seq_length": args.max_seq_length,
        "checkpoint_path": args.checkpoint_path,
        "hidden_dim": dims["hidden_dim"],
        "num_layers": dims["num_layers"],
        "num_heads": dims["num_heads"],
        "intermediate_dim": dims["intermediate_dim"],
        "dropout": 0.1,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "tokenizer_name": "gpt2"
    }
    
    # Create ModelConfig instance
    return ModelConfig.from_dict(config_dict)


def create_training_config(args) -> TrainingConfig:
    """
    Create a training configuration from arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        TrainingConfig: Training configuration
    """
    config_dict = {
        "training_strategy": args.training_mode,
        "model_type": args.training_mode,  # Use training mode as model type too
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "accumulation_steps": args.gradient_accumulation_steps,
        "max_grad_norm": 1.0,
        "fp16": args.fp16,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "logging_steps": 100,
        "device": args.device,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "max_seq_length": args.max_seq_length,
        "num_workers": 4
    }
    
    # Create TrainingConfig instance
    return TrainingConfig.from_dict(config_dict)


def create_data_config(args) -> DataConfig:
    """
    Create a data configuration from arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        DataConfig: Data configuration
    """
    # Determine dataset type/mode if auto
    dataset_type = args.dataset_type
    dialogue_mode = False
    
    if dataset_type == "auto":
        # Try to infer from training mode
        if args.training_mode == "dialogue":
            dataset_type = "dialogue"
            dialogue_mode = True
        elif args.training_mode == "function_call":
            dataset_type = "function_calling"
        else:
            # Default to text
            dataset_type = "text"
    elif dataset_type == "dialogue":
        dialogue_mode = True
    
    # Load function schema if needed
    function_schema = None
    if args.function_schema_path and (dataset_type == "function_calling" or args.training_mode == "function_call"):
        try:
            with open(args.function_schema_path, 'r') as f:
                function_schema = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading function schema from {args.function_schema_path}: {e}")
            logger.warning("Falling back to default function schema")
    
    config_dict = {
        "train_file": args.train_data,
        "validation_file": args.eval_data,
        "dataset_name": dataset_type,  # Use dataset_type as the name
        "max_length": args.max_seq_length,
        "dialogue_mode": dialogue_mode,
        "add_special_tokens": True,
        "dataloader_num_workers": 4
    }
    
    # Add dialogue-specific settings if needed
    if dialogue_mode:
        config_dict.update({
            "input_column_name": "input",
            "response_column_name": "response",
            # Additional dialogue parameters
            "tokenizer_config": {
                "max_history_turns": args.max_history_turns
            }
        })
    
    # Add function schema if available
    if function_schema:
        # Store function schema in dataset_config
        if "dataset_config" not in config_dict:
            config_dict["dataset_config"] = {}
        config_dict["dataset_config"]["function_schema"] = function_schema
    
    # Create DataConfig instance
    return DataConfig.from_dict(config_dict)


def create_extension_config(args) -> ExtensionConfig:
    """
    Create an extension configuration from arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        ExtensionConfig: Extension configuration
    """
    # Start with default extension config
    ext_config = ExtensionConfig()
    
    # Set global extension settings
    # Enable extensions explicitly via flag or implicitly if using extension model
    ext_config.extensions_enabled = args.enable_extensions or args.model_type == "semantic_resonance_with_extensions"
    ext_config.default_device = get_device(args.device).type
    
    # Set up multimodal extension if enabled via flag or implicitly with extension model
    if args.enable_multimodal or (args.model_type == "semantic_resonance_with_extensions" and not args.disable_multimodal):
        ext_config.multimodal.enabled = True
        
    # Set up memory extension if enabled via flag or implicitly with extension model
    if args.enable_memory or (args.model_type == "semantic_resonance_with_extensions" and not args.disable_memory):
        ext_config.memory.enabled = True
        
    # Set up quantum extension if enabled via flag or implicitly with extension model
    if args.enable_quantum or (args.model_type == "semantic_resonance_with_extensions" and not args.disable_quantum):
        ext_config.quantum.enabled = True
    
    # Load extension config from file if provided
    if args.extension_config:
        try:
            with open(args.extension_config, 'r') as f:
                file_config = json.load(f)
                
            # Update with file config
            if "multimodal" in file_config:
                for key, value in file_config["multimodal"].items():
                    if hasattr(ext_config.multimodal, key):
                        setattr(ext_config.multimodal, key, value)
            
            if "memory" in file_config:
                for key, value in file_config["memory"].items():
                    if hasattr(ext_config.memory, key):
                        setattr(ext_config.memory, key, value)
            
            if "quantum" in file_config:
                for key, value in file_config["quantum"].items():
                    if hasattr(ext_config.quantum, key):
                        setattr(ext_config.quantum, key, value)
            
            # Set feature flags if provided
            if "feature_flags" in file_config:
                ext_config.feature_flags.update(file_config["feature_flags"])
                
        except Exception as e:
            logger.warning(f"Error loading extension config from {args.extension_config}: {e}")
            logger.warning("Falling back to default extension config")
    
    return ext_config


def load_or_create_model(model_config: ModelConfig, ext_config: ExtensionConfig, device: torch.device, model_type: str) -> torch.nn.Module:
    """
    Load or create a model based on configuration.
    
    Args:
        model_config: Model configuration
        ext_config: Extension configuration
        device: Device to place model on
        model_type: Type of model to create
        
    Returns:
        Initialized model
    """
    logger.info(f"Initializing {model_type} model ({model_config.hidden_dim} hidden dimensions)")
    
    # Check if loading from checkpoint
    checkpoint_path = getattr(model_config, "checkpoint_path", None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        try:
            model = torch.load(checkpoint_path, map_location=device)
            return model
        except Exception as e:
            logger.error(f"Error loading model from checkpoint: {e}")
            logger.info("Falling back to creating new model")
    
    # Create new model based on model type
    try:
        if model_type == "semantic_resonance":
            # Import appropriate model class
            from src.model.semantic_resonance_model import SemanticResonanceModel
            
            # Create model
            model = SemanticResonanceModel(
                model_size="mini",  # Use fixed size
                vocab_size=model_config.vocab_size,
                max_seq_length=model_config.max_seq_length,
                **model_config.get_additional_params()
            )
            
        elif model_type == "semantic_resonance_with_extensions":
            # Import appropriate model class
            from src.model.semantic_resonance_model_with_extensions import SemanticResonanceModelWithExtensions
            
            # Add extension config to model config
            model_config_dict = model_config.to_dict()
            model_config_dict["extension_config"] = ext_config.to_dict()
            
            # Create model
            model = SemanticResonanceModelWithExtensions(model_config_dict)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Move model to device
        model = model.to(device)
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise


def prepare_dataloaders(
    data_config: DataConfig,
    training_config: TrainingConfig,
    tokenizer: Any
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Prepare training and evaluation dataloaders.
    
    Args:
        data_config: Data configuration
        training_config: Training configuration
        tokenizer: Tokenizer for processing text
        
    Returns:
        Tuple of (train_dataloader, eval_dataloader)
    """
    logger.info("Preparing dataloaders")
    
    # Get the dataset type from dataset_name
    dataset_type = data_config.dataset_name or "text"
    
    # Get file paths
    train_file = data_config.train_file
    val_file = data_config.validation_file
    
    # Get parameters for dataloaders
    max_length = data_config.max_length
    batch_size = data_config.batch_size or training_config.batch_size
    num_workers = data_config.dataloader_num_workers
    
    # Get dialogue-specific parameters
    dialogue_mode = data_config.dialogue_mode
    max_history_turns = None
    separate_input_response = True
    
    if dialogue_mode and data_config.tokenizer_config:
        max_history_turns = data_config.tokenizer_config.get("max_history_turns", 3)
    
    # Get function schema if available
    function_schema = None
    if data_config.dataset_config and "function_schema" in data_config.dataset_config:
        function_schema = data_config.dataset_config.get("function_schema")
    
    # Determine dataloader type based on data configuration
    try:
        if dataset_type == "text":
            train_dataloader = create_text_dataloader(
                data_path=train_file,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                shuffle=True,
                num_workers=num_workers
            )
            
            # Create evaluation dataloader if eval data path is provided
            eval_dataloader = None
            if val_file:
                eval_dataloader = create_text_dataloader(
                    data_path=val_file,
                    tokenizer=tokenizer,
                    batch_size=training_config.eval_batch_size or batch_size,
                    max_length=max_length,
                    shuffle=False,
                    num_workers=num_workers
                )
                
        elif dataset_type == "dialogue":
            train_dataloader = create_dialogue_dataloader(
                data_path=train_file,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                shuffle=True,
                num_workers=num_workers,
                max_history_turns=max_history_turns,
                separate_input_response=separate_input_response
            )
            
            # Create evaluation dataloader if eval data path is provided
            eval_dataloader = None
            if val_file:
                eval_dataloader = create_dialogue_dataloader(
                    data_path=val_file,
                    tokenizer=tokenizer,
                    batch_size=training_config.eval_batch_size or batch_size,
                    max_length=max_length,
                    shuffle=False,
                    num_workers=num_workers,
                    max_history_turns=max_history_turns,
                    separate_input_response=separate_input_response
                )
                
        elif dataset_type == "function_calling":
            train_dataloader = create_function_calling_dataloader(
                data_path=train_file,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                shuffle=True,
                num_workers=num_workers,
                function_schema=function_schema
            )
            
            # Create evaluation dataloader if eval data path is provided
            eval_dataloader = None
            if val_file:
                eval_dataloader = create_function_calling_dataloader(
                    data_path=val_file,
                    tokenizer=tokenizer,
                    batch_size=training_config.eval_batch_size or batch_size,
                    max_length=max_length,
                    shuffle=False,
                    num_workers=num_workers,
                    function_schema=function_schema
                )
                
        else:
            # Use generic dataloader creation from config
            train_dataloader = create_dataloader_from_config(
                config={
                    "dataset_name": dataset_type,
                    "train_file": train_file,
                    "max_length": max_length,
                    "dialogue_mode": dialogue_mode
                },
                tokenizer=tokenizer,
                batch_size=batch_size
            )
            
            # Create evaluation dataloader if eval data path is provided
            eval_dataloader = None
            if val_file:
                eval_config = {
                    "dataset_name": dataset_type,
                    "train_file": val_file,
                    "max_length": max_length,
                    "dialogue_mode": dialogue_mode
                }
                
                eval_dataloader = create_dataloader_from_config(
                    config=eval_config,
                    tokenizer=tokenizer,
                    batch_size=training_config.eval_batch_size or batch_size
                )
        
        return train_dataloader, eval_dataloader
        
    except Exception as e:
        logger.error(f"Error preparing dataloaders: {e}")
        raise


def train(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: DataConfig,
    ext_config: ExtensionConfig,
    model_type: str = "semantic_resonance_with_extensions"
) -> Dict[str, Any]:
    """
    Train a model with the given configurations.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        data_config: Data configuration
        ext_config: Extension configuration
        model_type: Type of model to train
        ext_config: Extension configuration
        
    Returns:
        Dictionary with training results
    """
    # Set up random seed for reproducibility
    setup_seed(training_config.seed)
    
    # Get device
    device = get_device(training_config.device)
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = load_or_create_model(model_config, ext_config, device, model_type)
    
    # Get tokenizer from model if available, otherwise create new one
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        logger.info("Creating new tokenizer")
        from transformers import AutoTokenizer, GPT2TokenizerFast
        
        # Use GPT2 as default tokenizer since it's widely compatible
        try:
            logger.info("Loading GPT2 tokenizer")
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            # Set the max sequence length
            tokenizer.model_max_length = model_config.max_seq_length
        except Exception as e:
            logger.error(f"Failed to load GPT2 tokenizer: {e}")
            raise
    
    # Prepare dataloaders
    train_dataloader, eval_dataloader = prepare_dataloaders(
        data_config, training_config, tokenizer
    )
    
    # Create trainer based on training mode
    logger.info(f"Creating trainer for mode: {training_config.model_type}")
    
    # Additional kwargs for special trainers
    trainer_kwargs = {}
    
    # Add dialogue-specific arguments
    if training_config.model_type == "dialogue":
        # Get max_history_turns from tokenizer_config if available
        max_history_turns = 3  # Default value
        separate_input_response = True  # Default value
        
        if data_config.tokenizer_config and "max_history_turns" in data_config.tokenizer_config:
            max_history_turns = data_config.tokenizer_config["max_history_turns"]
            
        # Update trainer kwargs
        trainer_kwargs.update({
            "max_history_turns": max_history_turns,
            "separate_input_response": separate_input_response
        })
    
    # Add function-call-specific arguments
    elif training_config.model_type == "function_call":
        trainer_kwargs.update({
            "function_schema": data_config.function_schema
        })
    
    # Create trainer
    if training_config.model_type == "dialogue":
        trainer = DialogueTrainer(
            model=model,
            train_dataloader=train_dataloader,
            training_config=training_config,
            model_config=model_config,
            data_config=data_config,
            eval_dataloader=eval_dataloader,
            **trainer_kwargs
        )
    elif training_config.model_type == "function_call":
        trainer = FunctionCallTrainer(
            model=model,
            train_dataloader=train_dataloader,
            training_config=training_config,
            model_config=model_config,
            data_config=data_config,
            eval_dataloader=eval_dataloader,
            **trainer_kwargs
        )
    else:
        # Use UnifiedTrainer for other modes
        trainer = UnifiedTrainer(
            model=model,
            train_dataloader=train_dataloader,
            training_config=training_config,
            model_config=model_config,
            data_config=data_config,
            eval_dataloader=eval_dataloader,
            training_type=training_config.model_type,
            **trainer_kwargs
        )
    
    # Create output directory if it doesn't exist
    os.makedirs(training_config.output_dir, exist_ok=True)
    
    # Save configurations
    config_path = os.path.join(training_config.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump({
            "model": model_config.to_dict(),
            "training": training_config.to_dict(),
            "data": data_config.to_dict(),
            "extensions": ext_config.to_dict()
        }, f, indent=2)
    
    logger.info(f"Saved configuration to {config_path}")
    
    # Run training
    logger.info("Starting training")
    start_time = time.time()
    training_results = trainer.train()
    training_duration = time.time() - start_time
    
    # Update training results
    training_results["training_duration"] = training_duration
    
    # Log training results
    logger.info(f"Training completed in {training_duration:.2f} seconds")
    logger.info(f"Best metric: {training_results.get('best_metric', 'N/A')}")
    
    # Save training results
    results_path = os.path.join(training_config.output_dir, "training_results.json")
    try:
        with open(results_path, 'w') as f:
            # Convert any tensor values to floats
            serializable_results = {}
            for k, v in training_results.items():
                if isinstance(v, torch.Tensor):
                    serializable_results[k] = v.item()
                elif isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                    serializable_results[k] = v
                else:
                    serializable_results[k] = str(v)
            
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Saved training results to {results_path}")
    except Exception as e:
        logger.error(f"Error saving training results: {e}")
    
    return training_results


def main():
    """Main function for the training script."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create configurations
    model_config = create_model_config(args)
    training_config = create_training_config(args)
    data_config = create_data_config(args)
    ext_config = create_extension_config(args)
    
    # Create output directory if it doesn't exist
    os.makedirs(training_config.output_dir, exist_ok=True)
    
    # Set up file logging
    file_handler = logging.FileHandler(
        os.path.join(training_config.output_dir, "train.log")
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)
    
    # Log configurations
    logger.info("Starting QLLM training with the following configurations:")
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Hidden dimensions: {model_config.hidden_dim}")
    logger.info(f"Training mode: {training_config.training_strategy}")
    logger.info(f"Dataset name: {data_config.dataset_name}")
    logger.info(f"Dialogue mode: {data_config.dialogue_mode}")
    logger.info(f"Extensions enabled: {ext_config.extensions_enabled}")
    if ext_config.extensions_enabled:
        logger.info(f"  - Multimodal: {ext_config.multimodal.enabled}")
        logger.info(f"  - Memory: {ext_config.memory.enabled}")
        logger.info(f"  - Quantum: {ext_config.quantum.enabled}")
    
    # Run training
    try:
        results = train(model_config, training_config, data_config, ext_config, args.model_type)
        logger.info("Training completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())