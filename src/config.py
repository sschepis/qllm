"""
Main configuration module for QLLM.

This module provides a simplified interface to the configuration system,
leveraging the shared configuration components to create and manage
configurations for training, model, and data.
"""

import os
import argparse
import logging
from typing import Dict, Any, Optional, List, Union

from src.config.config_manager import ConfigManager
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.config.data_config import DataConfig


logger = logging.getLogger("qllm.config")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    config_manager = ConfigManager()
    return config_manager.load_config(config_path)


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save the configuration to
    """
    config_manager = ConfigManager()
    config_manager.save_config(config, output_path)


def create_config(
    model_type: str = "semantic_resonance",
    model_size: str = "base",
    data_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    output_dir: str = "./output",
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    max_epochs: int = 10,
    device: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a complete configuration with default values.
    
    Args:
        model_type: Type of model to use
        model_size: Size of the model
        data_path: Path to the data
        dataset_name: Name of the dataset
        output_dir: Output directory
        batch_size: Batch size
        learning_rate: Learning rate
        max_epochs: Maximum number of epochs
        device: Device to use for training
        **kwargs: Additional configuration parameters
        
    Returns:
        Complete configuration dictionary
    """
    # Create model configuration
    model_config = ModelConfig(
        model_type=model_type,
        model_size=model_size,
        **kwargs.get("model_kwargs", {})
    )
    
    # Create training configuration
    training_config = TrainingConfig(
        output_dir=output_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        device=device,
        **kwargs.get("training_kwargs", {})
    )
    
    # Create data configuration
    data_config = DataConfig(
        data_path=data_path,
        dataset_name=dataset_name,
        **kwargs.get("data_kwargs", {})
    )
    
    # Create complete configuration
    config = {
        "model": model_config.to_dict(),
        "training": training_config.to_dict(),
        "data": data_config.to_dict()
    }
    
    # Add any additional configuration sections
    for key, value in kwargs.items():
        if key not in ["model_kwargs", "training_kwargs", "data_kwargs"]:
            config[key] = value
    
    return config


def parse_args() -> Dict[str, Any]:
    """
    Parse command line arguments for configuration.
    
    Returns:
        Dictionary of parsed arguments
    """
    parser = argparse.ArgumentParser(description="QLLM Configuration")
    
    # Configuration file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configuration file"
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model_type", type=str, default="semantic_resonance",
        help="Type of model to use"
    )
    model_group.add_argument(
        "--model_size", type=str, default="base",
        help="Size of the model"
    )
    
    # Data configuration
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data_path", type=str, default=None,
        help="Path to the data"
    )
    data_group.add_argument(
        "--dataset_name", type=str, default=None,
        help="Name of the dataset"
    )
    
    # Training configuration
    training_group = parser.add_argument_group("Training")
    training_group.add_argument(
        "--output_dir", type=str, default="./output",
        help="Output directory"
    )
    training_group.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size"
    )
    training_group.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Learning rate"
    )
    training_group.add_argument(
        "--max_epochs", type=int, default=10,
        help="Maximum number of epochs"
    )
    training_group.add_argument(
        "--device", type=str, default=None,
        help="Device to use for training"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert to dictionary
    args_dict = vars(args)
    
    # Handle config file
    if args.config is not None:
        # Load configuration from file
        file_config = load_config(args.config)
        
        # Update with command line arguments
        for key, value in args_dict.items():
            if key != "config" and value is not None:
                # Determine which section the argument belongs to
                if key.startswith("model_"):
                    # Model configuration
                    section = "model"
                    name = key[6:]  # Remove "model_" prefix
                elif key.startswith("data_"):
                    # Data configuration
                    section = "data"
                    name = key[5:]  # Remove "data_" prefix
                elif key in ["output_dir", "batch_size", "learning_rate", "max_epochs", "device"]:
                    # Training configuration
                    section = "training"
                    name = key
                else:
                    # Other configuration
                    section = None
                    name = key
                
                # Update configuration
                if section is not None:
                    file_config[section][name] = value
                else:
                    file_config[name] = value
        
        return file_config
    else:
        # Create configuration from arguments
        config = create_config(
            model_type=args.model_type,
            model_size=args.model_size,
            data_path=args.data_path,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            device=args.device
        )
        
        return config


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration dictionary
    """
    # Validate model configuration
    if "model" in config:
        model_config = ModelConfig.from_dict(config["model"])
        config["model"] = model_config.to_dict()
    
    # Validate training configuration
    if "training" in config:
        training_config = TrainingConfig.from_dict(config["training"])
        config["training"] = training_config.to_dict()
    
    # Validate data configuration
    if "data" in config:
        data_config = DataConfig.from_dict(config["data"])
        config["data"] = data_config.to_dict()
    
    return config


def setup_config_wizard() -> Dict[str, Any]:
    """
    Run an interactive configuration wizard.
    
    Returns:
        Configuration dictionary
    """
    print("QLLM Configuration Wizard")
    print("========================")
    
    # Model configuration
    print("\nModel Configuration:")
    model_type = input("Model type [semantic_resonance]: ") or "semantic_resonance"
    model_size = input("Model size [base]: ") or "base"
    
    # Data configuration
    print("\nData Configuration:")
    data_path = input("Data path: ")
    dataset_name = input("Dataset name: ")
    
    # Training configuration
    print("\nTraining Configuration:")
    output_dir = input("Output directory [./output]: ") or "./output"
    batch_size = int(input("Batch size [8]: ") or "8")
    learning_rate = float(input("Learning rate [0.0001]: ") or "0.0001")
    max_epochs = int(input("Maximum epochs [10]: ") or "10")
    device = input("Device [auto]: ") or None
    
    # Create configuration
    config = create_config(
        model_type=model_type,
        model_size=model_size,
        data_path=data_path,
        dataset_name=dataset_name,
        output_dir=output_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        device=device
    )
    
    # Ask to save configuration
    save_path = input("\nSave configuration to (leave empty to skip): ")
    if save_path:
        save_config(config, save_path)
        print(f"Configuration saved to {save_path}")
    
    return config


if __name__ == "__main__":
    # Run configuration wizard if executed directly
    config = setup_config_wizard()