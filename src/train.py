"""
Main training script for QLLM.

This module provides a simplified interface for training models,
leveraging the unified training components to streamline the process.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, Optional, List, Union

import torch
import numpy as np

from src.config import load_config, parse_args, validate_config
from src.training import TrainerFactory
from src.data import (
    create_text_dataloader,
    create_dialogue_dataloader,
    create_function_calling_dataloader,
    create_dataloader_from_config
)
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.config.data_config import DataConfig


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("qllm_train.log")
    ]
)
logger = logging.getLogger("qllm.train")


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
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_or_create_model(
    model_config: ModelConfig,
    device: torch.device
) -> torch.nn.Module:
    """
    Load or create a model based on configuration.
    
    Args:
        model_config: Model configuration
        device: Device to place model on
        
    Returns:
        Initialized model
    """
    logger.info(f"Initializing {model_config.model_type} model ({model_config.model_size})")
    
    # Check if loading from checkpoint
    if model_config.checkpoint_path:
        logger.info(f"Loading model from checkpoint: {model_config.checkpoint_path}")
        try:
            model = torch.load(model_config.checkpoint_path, map_location=device)
            return model
        except Exception as e:
            logger.error(f"Error loading model from checkpoint: {e}")
            logger.info("Falling back to creating new model")
    
    # Create new model based on model type
    try:
        if model_config.model_type == "semantic_resonance":
            # Import appropriate model class
            from src.model.semantic_resonance_model import SemanticResonanceModel
            
            # Create model
            model = SemanticResonanceModel(
                model_size=model_config.model_size,
                vocab_size=model_config.vocab_size,
                max_seq_length=model_config.max_seq_length,
                **model_config.get_additional_params()
            )
            
        elif model_config.model_type == "semantic_resonance_with_extensions":
            # Import appropriate model class
            from src.model.semantic_resonance_model_with_extensions import SemanticResonanceModelWithExtensions
            
            # Create model
            model = SemanticResonanceModelWithExtensions(
                model_size=model_config.model_size,
                vocab_size=model_config.vocab_size,
                max_seq_length=model_config.max_seq_length,
                **model_config.get_additional_params()
            )
            
        else:
            raise ValueError(f"Unknown model type: {model_config.model_type}")
        
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
    
    # Determine dataloader type based on data configuration
    try:
        if data_config.dataset_type == "text":
            train_dataloader = create_text_dataloader(
                data_path=data_config.data_path,
                tokenizer=tokenizer,
                batch_size=training_config.batch_size,
                max_length=training_config.max_seq_length,
                shuffle=True,
                num_workers=training_config.num_workers
            )
            
            # Create evaluation dataloader if eval data path is provided
            eval_dataloader = None
            if data_config.eval_data_path:
                eval_dataloader = create_text_dataloader(
                    data_path=data_config.eval_data_path,
                    tokenizer=tokenizer,
                    batch_size=training_config.eval_batch_size or training_config.batch_size,
                    max_length=training_config.max_seq_length,
                    shuffle=False,
                    num_workers=training_config.num_workers
                )
                
        elif data_config.dataset_type == "dialogue":
            train_dataloader = create_dialogue_dataloader(
                data_path=data_config.data_path,
                tokenizer=tokenizer,
                batch_size=training_config.batch_size,
                max_length=training_config.max_seq_length,
                shuffle=True,
                num_workers=training_config.num_workers,
                max_history_turns=data_config.max_history_turns,
                separate_input_response=data_config.separate_input_response
            )
            
            # Create evaluation dataloader if eval data path is provided
            eval_dataloader = None
            if data_config.eval_data_path:
                eval_dataloader = create_dialogue_dataloader(
                    data_path=data_config.eval_data_path,
                    tokenizer=tokenizer,
                    batch_size=training_config.eval_batch_size or training_config.batch_size,
                    max_length=training_config.max_seq_length,
                    shuffle=False,
                    num_workers=training_config.num_workers,
                    max_history_turns=data_config.max_history_turns,
                    separate_input_response=data_config.separate_input_response
                )
                
        elif data_config.dataset_type == "function_calling":
            train_dataloader = create_function_calling_dataloader(
                data_path=data_config.data_path,
                tokenizer=tokenizer,
                batch_size=training_config.batch_size,
                max_length=training_config.max_seq_length,
                shuffle=True,
                num_workers=training_config.num_workers,
                function_schema=data_config.function_schema
            )
            
            # Create evaluation dataloader if eval data path is provided
            eval_dataloader = None
            if data_config.eval_data_path:
                eval_dataloader = create_function_calling_dataloader(
                    data_path=data_config.eval_data_path,
                    tokenizer=tokenizer,
                    batch_size=training_config.eval_batch_size or training_config.batch_size,
                    max_length=training_config.max_seq_length,
                    shuffle=False,
                    num_workers=training_config.num_workers,
                    function_schema=data_config.function_schema
                )
                
        else:
            # Use generic dataloader creation from config
            train_dataloader = create_dataloader_from_config(
                config=data_config.to_dict(),
                tokenizer=tokenizer,
                batch_size=training_config.batch_size
            )
            
            # Create evaluation dataloader if eval data path is provided
            eval_dataloader = None
            if data_config.eval_data_path:
                eval_config = data_config.to_dict()
                eval_config["data_path"] = data_config.eval_data_path
                
                eval_dataloader = create_dataloader_from_config(
                    config=eval_config,
                    tokenizer=tokenizer,
                    batch_size=training_config.eval_batch_size or training_config.batch_size
                )
        
        return train_dataloader, eval_dataloader
        
    except Exception as e:
        logger.error(f"Error preparing dataloaders: {e}")
        raise


def train(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a model with the given configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Dictionary with training results
    """
    # Extract configurations
    model_config = ModelConfig.from_dict(config["model"])
    training_config = TrainingConfig.from_dict(config["training"])
    data_config = DataConfig.from_dict(config["data"])
    
    # Set up random seed for reproducibility
    setup_seed(training_config.seed)
    
    # Get device
    device = get_device(training_config.device)
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = load_or_create_model(model_config, device)
    
    # Get tokenizer from model if available, otherwise create new one
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        logger.info("Creating new tokenizer")
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_name)
        except Exception:
            logger.warning("Failed to load tokenizer, using a basic one")
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast(
                vocab_size=model_config.vocab_size,
                model_max_length=model_config.max_seq_length
            )
    
    # Prepare dataloaders
    train_dataloader, eval_dataloader = prepare_dataloaders(
        data_config, training_config, tokenizer
    )
    
    # Create trainer using factory
    logger.info("Creating trainer")
    trainer = TrainerFactory.create_trainer(
        model=model,
        train_dataloader=train_dataloader,
        training_config=training_config,
        model_config=model_config,
        data_config=data_config,
        eval_dataloader=eval_dataloader,
        trainer_type=training_config.trainer_type
    )
    
    # Run training
    logger.info("Starting training")
    training_results = trainer.train()
    
    # Log training results
    logger.info(f"Training completed in {training_results['training_duration']:.2f} seconds")
    logger.info(f"Best metric: {training_results['best_metric']}")
    
    return training_results


def main():
    """Main entry point for training."""
    # Parse arguments
    config = parse_args()
    
    # Validate configuration
    config = validate_config(config)
    
    # Run training
    try:
        results = train(config)
        logger.info("Training completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())