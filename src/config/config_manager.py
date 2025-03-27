"""
Configuration manager for the Quantum Resonance Language Model.

This module provides utilities for loading, saving, and managing 
configuration for model training and evaluation.
"""

import os
import json
from typing import Dict, Any, Optional, Union, List, Type

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.config.data_config import DataConfig
from src.config.config_schema import get_schema


class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.schema = get_schema()
    
    def create_default_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Create a default configuration with all sections.
        
        Returns:
            Dictionary with model, training, and data configuration sections
        """
        model_config = ModelConfig().to_dict()
        training_config = TrainingConfig().to_dict()
        data_config = DataConfig().to_dict()
        
        return {
            "model": model_config,
            "training": training_config,
            "data": data_config,
        }
    
    def load_config(self, config_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary with configuration sections
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the configuration file isn't valid JSON
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Ensure all sections exist
        for section in ["model", "training", "data"]:
            if section not in config:
                config[section] = {}
        
        return config
    
    def save_config(self, config: Dict[str, Dict[str, Any]], config_path: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            config: Configuration dictionary to save
            config_path: Path to save the configuration to
            
        Raises:
            ValueError: If the configuration is invalid
        """
        # Validate configuration
        errors = self.schema.validate(config)
        if errors:
            error_msg = "Configuration validation failed:\n\n"
            for error in errors:
                error_msg += f"  - {error}\n"
            raise ValueError(error_msg)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def merge_configs(self, base_config: Dict[str, Dict[str, Any]], 
                    override_config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Merge two configurations, with override_config taking precedence.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override base values
            
        Returns:
            Merged configuration
        """
        merged_config = self.create_default_config()
        
        # First copy base config
        for section in merged_config:
            if section in base_config:
                merged_config[section].update(base_config[section])
        
        # Then override with override_config
        for section in merged_config:
            if section in override_config:
                merged_config[section].update(override_config[section])
        
        return merged_config
    
    def from_command_line_args(self, args: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Create a configuration from command line arguments.
        
        Args:
            args: Dictionary of command line arguments
            
        Returns:
            Configuration dictionary
        """
        config = {
            "model": {},
            "training": {},
            "data": {}
        }
        
        # Map known arguments to their sections
        for arg, value in args.items():
            if value is None:
                continue
                
            # Model arguments
            if arg in ["hidden_dim", "num_layers", "num_heads", "dropout", "max_seq_length"]:
                config["model"][arg] = value
            
            # Training arguments
            elif arg in ["batch_size", "learning_rate", "weight_decay", "max_epochs", 
                        "training_type", "device", "output_dir", "seed"]:
                config["training"][arg] = value
            
            # Data arguments
            elif arg in ["dataset_name", "tokenizer_name", "train_file", "validation_file", 
                        "test_file", "max_length"]:
                config["data"][arg] = value
        
        return config
    
    def to_config_classes(self, config: Dict[str, Dict[str, Any]]) -> Dict[str, Union[ModelConfig, TrainingConfig, DataConfig]]:
        """
        Convert a configuration dictionary to configuration classes.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary with configuration class instances
        """
        model_config = ModelConfig.from_dict(config.get("model", {}))
        training_config = TrainingConfig.from_dict(config.get("training", {}))
        data_config = DataConfig.from_dict(config.get("data", {}))
        
        return {
            "model": model_config,
            "training": training_config,
            "data": data_config
        }
    
    def validate_config(self, config: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Validate a configuration against the schema.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of error messages, empty if validation passes
        """
        return self.schema.validate(config)