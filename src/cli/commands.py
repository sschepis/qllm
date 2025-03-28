"""
Command implementations for the Quantum Resonance Language Model.
Provides handler functions for the different CLI commands:
train, evaluate, compress, and generate.
"""

import os
import logging
import torch
from typing import Dict, Any, Optional, Tuple

from src.utils.config import ModelConfig, TrainingConfig, DataConfig, GenerationConfig
from src.utils.config import get_default_configs, merge_configs, save_configs
from src.utils.logging import setup_logger, get_default_logger
from src.utils.device import get_device, print_device_info
from src.model.semantic_resonance_model import SemanticResonanceModel
from src.data.dataloaders import get_tokenizer, get_wikitext_dataloaders
from src.training.checkpoint import load_checkpoint, find_latest_checkpoint
from src.training import TrainerFactory, get_trainer
from src.training.model_adapters import StandardModelAdapter

# Get logger
logger = logging.getLogger("quantum_resonance")


def setup_environment(
    config_dir: Optional[str],
    output_dir: str,
    model_args: Dict[str, Any] = None,
    training_args: Dict[str, Any] = None,
    data_args: Dict[str, Any] = None,
    log_level: str = "info",
    trainer_type: str = "enhanced"
) -> Tuple[Dict[str, Any], str]:
    """
    Set up the environment for commands.
    
    Args:
        config_dir: Directory containing configuration files
        output_dir: Directory to save outputs
        model_args: Model arguments from CLI
        training_args: Training arguments from CLI
        data_args: Data arguments from CLI
        log_level: Logging level
        
    Returns:
        Tuple[Dict[str, Any], str]: Configuration dictionary and device
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, "quantum_resonance.log")
    logger = setup_logger(
        name="quantum_resonance",
        log_file=log_file,
        log_level=getattr(logging, log_level.upper())
    )
    
    # Get default configurations
    configs = get_default_configs()
    
    # Load configurations from file if provided
    if config_dir and os.path.exists(config_dir):
        logger.info(f"Loading configurations from {config_dir}")
        
        # For each config file that might exist
        config_files = {
            "model": "model_config.json",
            "training": "training_config.json",
            "data": "data_config.json",
            "generation": "generation_config.json"
        }
        
        for config_key, filename in config_files.items():
            filepath = os.path.join(config_dir, filename)
            if os.path.exists(filepath):
                logger.info(f"Loading {config_key} configuration from {filepath}")
                # Update config object directly
                from src.utils.config import load_dataclass_from_json
                config_class = type(configs[config_key])
                configs[config_key] = load_dataclass_from_json(config_class, filepath)
    
    # Update configurations from CLI arguments
    if model_args:
        for key, value in model_args.items():
            if hasattr(configs["model"], key):
                setattr(configs["model"], key, value)
    
    if training_args:
        for key, value in training_args.items():
            if hasattr(configs["training"], key):
                setattr(configs["training"], key, value)
    
    if data_args:
        for key, value in data_args.items():
            if hasattr(configs["data"], key):
                setattr(configs["data"], key, value)
    
    # Set output directory in training config
    configs["training"].output_dir = output_dir
    
    # Set trainer type in training config
    setattr(configs["training"], "trainer_type", trainer_type)
    
    # Save updated configurations
    save_configs(configs, output_dir)
    
    # Set up device
    device = get_device(configs["training"].device)
    print_device_info(device)
    
    # Set random seed
    torch.manual_seed(configs["training"].seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs["training"].seed)
    
    return configs, device


def train_command(args: Dict[str, Any]) -> int:
    """
    Run the training command.
    
    Args:
        args: Command line arguments
        
    Returns:
        int: Exit code
    """
    # Extract arguments
    config_dir = args.get("config_dir")
    output_dir = args.get("output_dir", "runs/quantum_resonance")
    checkpoint_path = args.get("checkpoint_path")
    ignore_checkpoints = args.get("ignore_checkpoints", False)
    log_level = args.get("log_level", "info")
    trainer_type = args.get("trainer_type", "enhanced")
    
    # Extract model/training/data args
    from src.cli.arg_parsing import extract_config_args
    model_args, training_args, data_args = extract_config_args(args)
    
    # Setup environment
    configs, device = setup_environment(
        config_dir=config_dir,
        output_dir=output_dir,
        model_args=model_args,
        training_args=training_args,
        data_args=data_args,
        log_level=log_level,
        trainer_type=trainer_type
    )
    
    # Get configurations
    model_config = configs["model"]
    training_config = configs["training"]
    data_config = configs["data"]
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {data_config.tokenizer_name}")
    tokenizer = get_tokenizer(data_config.tokenizer_name)
    
    # Update model config with vocab size
    model_config.vocab_size = len(tokenizer)
    
    # Create dataloaders
    logger.info("Loading datasets...")
    dataloaders = get_wikitext_dataloaders(
        tokenizer=tokenizer,
        batch_size=training_config.batch_size,
        eval_batch_size=training_config.eval_batch_size,
        max_length=data_config.max_length,
        stride=data_config.stride,
        num_workers=data_config.preprocessing_num_workers,
        cache_dir=data_config.cache_dir
    )
    
    # Check for existing checkpoint
    if not ignore_checkpoints and not checkpoint_path:
        checkpoint_path = find_latest_checkpoint(output_dir)
        if checkpoint_path:
            logger.info(f"Found latest checkpoint: {checkpoint_path}")
    
    # Initialize model
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        # Create empty model first
        model = SemanticResonanceModel(model_config)
        model.to(device)
        
        # Load checkpoint (handles strict=False and warnings)
        load_checkpoint(model, checkpoint_path, map_location=device)
    else:
        logger.info("Initializing new model...")
        model = SemanticResonanceModel(model_config)
        model.to(device)
    
    # Initialize trainer using factory
    trainer_factory = TrainerFactory()
    
    # Determine trainer type from config
    trainer_type = getattr(training_config, "trainer_type", "enhanced")
    logger.info(f"Creating {trainer_type} trainer")
    
    # Create trainer
    trainer = trainer_factory.create_trainer(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        output_dir=output_dir,
        logger=logger
    )
    
    # Set model in the trainer
    trainer.model = model
    
    # Initialize components
    trainer.initialize_tokenizer()
    trainer.tokenizer = tokenizer  # Ensure tokenizer is set
    
    # Set dataloaders
    trainer.dataloaders = dataloaders
    
    # Initialize optimizer
    trainer.initialize_optimizer()
    
    # Load checkpoint into trainer if specified
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading training state from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
    
    # Train model
    logger.info("Starting training...")
    
    # With EnhancedTrainer we need to set the model first, then initialize other components
    if hasattr(trainer, 'model_adapter'):
        # For EnhancedTrainer, just set the model in the model adapter
        if trainer.model_adapter is not None:
            trainer.model_adapter.set_model(model)
        else:
            # Create a model adapter if needed
            from src.training.model_adapters import StandardModelAdapter
            # Ensure model_config has tokenizer info from data_config
            model_config.extra_model_params['tokenizer_name'] = data_config.tokenizer_name
            model_adapter = StandardModelAdapter(model_config, training_config)
            model_adapter.set_model(model)
            if hasattr(trainer, 'set_model_adapter'):
                trainer.set_model_adapter(model_adapter)
    else:
        # For other trainers, just set the model directly
        trainer.model = model
    
    # Initialize remaining components
    if hasattr(trainer, 'initialize_tokenizer'):
        trainer.tokenizer = tokenizer
    
    # Make sure dataloaders are set
    if hasattr(trainer, 'dataloaders'):
        trainer.dataloaders = dataloaders
    
    # Train
    train_stats = trainer.train()
    
    # Print training statistics
    logger.info("\nTraining completed!")
    logger.info(f"Best validation loss: {train_stats['best_val_loss']:.4f}")
    logger.info(f"Final validation perplexity: {train_stats['val_perplexity']:.2f}")
    
    if "test_perplexity" in train_stats:
        logger.info(f"Test perplexity: {train_stats['test_perplexity']:.2f}")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Model and tokenizer saved to {output_dir}")
    
    return 0  # Success


def evaluate_command(args: Dict[str, Any]) -> int:
    """
    Run the evaluation command.
    
    Args:
        args: Command line arguments
        
    Returns:
        int: Exit code
    """
    # Extract arguments
    config_dir = args.get("config_dir")
    checkpoint_path = args.get("checkpoint_path")
    output_dir = args.get("output_dir", "runs/quantum_resonance")
    eval_split = args.get("eval_split", "validation")
    compute_full_metrics = args.get("compute_full_metrics", False)
    log_level = args.get("log_level", "info")
    trainer_type = args.get("trainer_type", "enhanced")
    
    # Extract model/data args
    from src.cli.arg_parsing import extract_config_args
    model_args, training_args, data_args = extract_config_args(args)
    
    # Setup environment
    configs, device = setup_environment(
        config_dir=config_dir,
        output_dir=output_dir,
        model_args=model_args,
        training_args=training_args,
        data_args=data_args,
        log_level=log_level,
        trainer_type=trainer_type
    )
    
    # Get configurations
    model_config = configs["model"]
    data_config = configs["data"]
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {data_config.tokenizer_name}")
    tokenizer = get_tokenizer(data_config.tokenizer_name)
    
    # Update model config with vocab size
    model_config.vocab_size = len(tokenizer)
    
    # Check for checkpoint
    if not checkpoint_path:
        checkpoint_path = find_latest_checkpoint(output_dir)
        if not checkpoint_path:
            logger.error("No checkpoint found. Please specify a checkpoint path.")
            return 1
    
    # Initialize model
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = SemanticResonanceModel(model_config)
    
    # Load checkpoint
    load_checkpoint(model, checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    
    # Create dataloaders
    logger.info("Loading datasets...")
    dataloaders = get_wikitext_dataloaders(
        tokenizer=tokenizer,
        batch_size=args.get("batch_size", 16),
        max_length=data_config.max_length,
        stride=data_config.stride,
        num_workers=data_config.preprocessing_num_workers,
        cache_dir=data_config.cache_dir
    )
    
    # Check if split exists
    if eval_split not in dataloaders:
        logger.error(f"Dataset split '{eval_split}' not found. Available splits: {list(dataloaders.keys())}")
        return 1
    
    # Evaluate model
    logger.info(f"Evaluating model on {eval_split} split...")
    
    # Initialize trainer using factory
    trainer_factory = TrainerFactory()
    
    # Determine trainer type from config
    trainer_type = getattr(training_config, "trainer_type", "enhanced")
    logger.info(f"Creating {trainer_type} trainer for evaluation")
    
    # Create trainer
    trainer = trainer_factory.create_trainer(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        output_dir=output_dir,
        logger=logger
    )
    
    # Set model in the trainer
    if hasattr(trainer, 'model_adapter'):
        # For EnhancedTrainer, just set the model in the model adapter
        if trainer.model_adapter is not None:
            trainer.model_adapter.set_model(model)
        else:
            # Create a model adapter if needed
            from src.training.model_adapters import StandardModelAdapter
            # Ensure model_config has tokenizer info from data_config
            model_config.extra_model_params['tokenizer_name'] = data_config.tokenizer_name
            model_adapter = StandardModelAdapter(model_config, training_config)
            model_adapter.set_model(model)
            if hasattr(trainer, 'set_model_adapter'):
                trainer.set_model_adapter(model_adapter)
    else:
        # For other trainers, just set the model directly
        trainer.model = model
    
    # Set tokenizer
    trainer.tokenizer = tokenizer
    
    # Set dataloaders for evaluation
    trainer.dataloaders = {eval_split: dataloaders[eval_split]}
    
    # Evaluate
    metrics = trainer.evaluate(eval_split)
    
    # Print metrics
    logger.info("\nEvaluation results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")
    
    return 0  # Success


def generate_command(args: Dict[str, Any]) -> int:
    """
    Run the text generation command.
    
    Args:
        args: Command line arguments
        
    Returns:
        int: Exit code
    """
    # Extract arguments
    config_dir = args.get("config_dir")
    checkpoint_path = args.get("checkpoint_path")
    output_dir = args.get("output_dir", "runs/quantum_resonance")
    prompt = args.get("prompt")
    max_length = args.get("max_length", 200)
    temperature = args.get("temperature", 0.7)
    top_k = args.get("top_k", 50)
    top_p = args.get("top_p", 0.9)
    log_level = args.get("log_level", "info")
    
    # Setup environment
    configs, device = setup_environment(
        config_dir=config_dir,
        output_dir=output_dir,
        log_level=log_level
    )
    
    # Get configurations
    model_config = configs["model"]
    generation_config = configs["generation"]
    
    # Update generation config with CLI args
    if max_length is not None:
        generation_config.max_length = max_length
    if temperature is not None:
        generation_config.temperature = temperature
    if top_k is not None:
        generation_config.top_k = top_k
    if top_p is not None:
        generation_config.top_p = top_p
    
    # Get tokenizer
    try:
        if os.path.isdir(checkpoint_path):
            tokenizer_path = checkpoint_path
        else:
            tokenizer_path = os.path.dirname(checkpoint_path)
        
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = get_tokenizer(tokenizer_path)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from checkpoint path: {e}")
        logger.info("Falling back to default tokenizer")
        tokenizer = get_tokenizer("gpt2")
    
    # Update model config with vocab size
    model_config.vocab_size = len(tokenizer)
    
    # Check for checkpoint
    if not checkpoint_path:
        checkpoint_path = find_latest_checkpoint(output_dir)
        if not checkpoint_path:
            logger.error("No checkpoint found. Please specify a checkpoint path.")
            return 1
    
    # Initialize model
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = SemanticResonanceModel(model_config)
    
    # Load checkpoint
    load_checkpoint(model, checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    
    # Get prompt
    if prompt is None:
        prompt = input("Enter a prompt: ")
    
    # Generate text
    logger.info(f"Generating text with temperature={generation_config.temperature}, "
                f"top_k={generation_config.top_k}, top_p={generation_config.top_p}...")
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=generation_config.max_length,
            min_length=generation_config.min_length,
            temperature=generation_config.temperature,
            do_sample=generation_config.do_sample,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            repetition_penalty=generation_config.repetition_penalty,
            num_return_sequences=generation_config.num_return_sequences,
        )
    
    # Decode output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Print generated text
    print("\nGenerated text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)
    
    return 0  # Success


def compress_command(args: Dict[str, Any]) -> int:
    """
    Run the model compression command.
    
    Args:
        args: Command line arguments
        
    Returns:
        int: Exit code
    """
    # Extract arguments
    config_dir = args.get("config_dir")
    checkpoint_path = args.get("checkpoint_path")
    output_dir = args.get("output_dir", "runs/quantum_resonance")
    compression_method = args.get("compression_method", "both")
    compression_threshold = args.get("compression_threshold", 0.8)
    mask_type = args.get("mask_type", "mod")
    compare_performance = args.get("compare_performance", False)
    log_level = args.get("log_level", "info")
    
    # Setup environment
    configs, device = setup_environment(
        config_dir=config_dir,
        output_dir=output_dir,
        log_level=log_level
    )
    
    # Get configurations
    model_config = configs["model"]
    
    # Initialize tokenizer (needed if comparing performance)
    if compare_performance:
        from src.data.dataloaders import get_tokenizer
        tokenizer = get_tokenizer("gpt2")
        model_config.vocab_size = len(tokenizer)
    
    # Check for checkpoint
    if not checkpoint_path:
        checkpoint_path = find_latest_checkpoint(output_dir)
        if not checkpoint_path:
            logger.error("No checkpoint found. Please specify a checkpoint path.")
            return 1
    
    # Initialize model
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = SemanticResonanceModel(model_config)
    
    # Load checkpoint
    load_checkpoint(model, checkpoint_path, map_location=device)
    model.to(device)
    
    # Configure compression
    compression_config = {
        "method": compression_method,
        "primes": model_config.primes,
        "threshold": compression_threshold,
        "mask_type": mask_type
    }
    
    # Apply compression
    logger.info(f"Applying compression with method: {compression_method}, threshold: {compression_threshold}")
    
    # Import compression utilities
    from src.utils.compression import compress_model, compare_models
    
    # Compress model
    compressed_model, compression_ratio = compress_model(model, compression_config)
    
    logger.info(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Compare models if requested
    if compare_performance:
        logger.info("Comparing original and compressed models...")
        comparison = compare_models(model, compressed_model)
        
        for key, value in comparison.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
    
    # Save compressed model
    compressed_path = os.path.join(output_dir, "compressed_model")
    os.makedirs(compressed_path, exist_ok=True)
    
    # Save model state dict
    torch.save(compressed_model.state_dict(), os.path.join(compressed_path, "model.pt"))
    
    # Save model configuration
    from src.utils.config import save_dataclass_to_json
    save_dataclass_to_json(model_config, os.path.join(compressed_path, "model_config.json"))
    
    # Save compression details
    import json
    with open(os.path.join(compressed_path, "compression_info.json"), 'w') as f:
        json.dump({
            "method": compression_method,
            "threshold": compression_threshold,
            "mask_type": mask_type,
            "compression_ratio": compression_ratio,
            "original_checkpoint": checkpoint_path
        }, f, indent=2)
    
    logger.info(f"Compressed model saved to {compressed_path}")
    
    return 0  # Success