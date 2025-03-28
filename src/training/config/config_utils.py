"""
Configuration utilities for the enhanced training system.

This module provides utilities for loading, merging, and validating
training configurations, enabling flexible configuration management.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any, Type
from dataclasses import asdict, is_dataclass

from src.training.config.training_config import EnhancedTrainingConfig
from src.config.training_config import TrainingConfig


# Get logger
logger = logging.getLogger("quantum_resonance")


def load_config_from_file(
    config_path: str
) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine file type from extension
    file_ext = os.path.splitext(config_path)[1].lower()
    
    try:
        if file_ext == ".json":
            with open(config_path, "r") as f:
                config = json.load(f)
        elif file_ext in [".yaml", ".yml"]:
            try:
                import yaml
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML not installed. Install with 'pip install pyyaml'")
        else:
            raise ValueError(f"Unsupported configuration file format: {file_ext}")
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def create_config_from_dict(
    config_dict: Dict[str, Any],
    config_class: Type = EnhancedTrainingConfig
) -> Union[EnhancedTrainingConfig, TrainingConfig]:
    """
    Create a configuration object from a dictionary.
    
    Args:
        config_dict: Configuration dictionary
        config_class: Configuration class to use
        
    Returns:
        Configuration object
    """
    try:
        # Create configuration object
        if config_class == EnhancedTrainingConfig:
            config = EnhancedTrainingConfig.from_dict(config_dict)
        elif config_class == TrainingConfig:
            config = TrainingConfig.from_dict(config_dict)
        else:
            # Try to instantiate the class with the dictionary
            config = config_class(**config_dict)
        
        return config
    
    except Exception as e:
        logger.error(f"Error creating configuration object: {e}")
        raise


def merge_configs(
    base_config: Union[Dict[str, Any], EnhancedTrainingConfig, TrainingConfig],
    override_config: Union[Dict[str, Any], EnhancedTrainingConfig, TrainingConfig]
) -> Dict[str, Any]:
    """
    Merge two configurations.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration dictionary
    """
    # Convert to dictionaries if needed
    base_dict = _config_to_dict(base_config)
    override_dict = _config_to_dict(override_config)
    
    # Deep merge the dictionaries
    return _deep_merge(base_dict, override_dict)


def _config_to_dict(
    config: Union[Dict[str, Any], EnhancedTrainingConfig, TrainingConfig]
) -> Dict[str, Any]:
    """
    Convert a configuration object to a dictionary.
    
    Args:
        config: Configuration object or dictionary
        
    Returns:
        Configuration dictionary
    """
    if isinstance(config, dict):
        return config
    elif hasattr(config, "to_dict"):
        return config.to_dict()
    elif is_dataclass(config):
        return asdict(config)
    else:
        # Try to convert to dict using __dict__
        return {k: v for k, v in config.__dict__.items() if not k.startswith("_")}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary to override base with
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = _deep_merge(result[key], value)
        else:
            # Override or add value
            result[key] = value
    
    return result


def validate_config(
    config: Union[Dict[str, Any], EnhancedTrainingConfig, TrainingConfig]
) -> List[str]:
    """
    Validate a configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Convert to dictionary if needed
    config_dict = _config_to_dict(config)
    
    # Validate required fields
    required_fields = ["batch_size", "learning_rate", "max_epochs"]
    for field in required_fields:
        if field not in config_dict:
            errors.append(f"Missing required field: {field}")
    
    # Validate field types
    if "batch_size" in config_dict and not isinstance(config_dict["batch_size"], int):
        errors.append(f"batch_size must be an integer")
    
    if "learning_rate" in config_dict and not isinstance(config_dict["learning_rate"], (int, float)):
        errors.append(f"learning_rate must be a number")
    
    if "max_epochs" in config_dict and not isinstance(config_dict["max_epochs"], int):
        errors.append(f"max_epochs must be an integer")
    
    # Validate specific constraints
    if "batch_size" in config_dict and config_dict["batch_size"] <= 0:
        errors.append(f"batch_size must be positive")
    
    if "learning_rate" in config_dict and config_dict["learning_rate"] <= 0:
        errors.append(f"learning_rate must be positive")
    
    if "max_epochs" in config_dict and config_dict["max_epochs"] <= 0:
        errors.append(f"max_epochs must be positive")
    
    # Validate optimizer and scheduler
    if "optimizer" in config_dict and not isinstance(config_dict["optimizer"], str):
        errors.append(f"optimizer must be a string")
    
    if "lr_scheduler" in config_dict and not isinstance(config_dict["lr_scheduler"], str):
        errors.append(f"lr_scheduler must be a string")
    
    return errors


def save_config_to_file(
    config: Union[Dict[str, Any], EnhancedTrainingConfig, TrainingConfig],
    config_path: str
) -> None:
    """
    Save a configuration to a file.
    
    Args:
        config: Configuration to save
        config_path: Path to save configuration to
    """
    # Convert to dictionary if needed
    config_dict = _config_to_dict(config)
    
    # Determine file type from extension
    file_ext = os.path.splitext(config_path)[1].lower()
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        if file_ext == ".json":
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        elif file_ext in [".yaml", ".yml"]:
            try:
                import yaml
                with open(config_path, "w") as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML not installed. Install with 'pip install pyyaml'")
        else:
            raise ValueError(f"Unsupported configuration file format: {file_ext}")
        
        logger.info(f"Saved configuration to {config_path}")
    
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")
        raise


def load_config(
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    defaults: Optional[Dict[str, Any]] = None,
    config_class: Type = EnhancedTrainingConfig
) -> Union[EnhancedTrainingConfig, TrainingConfig]:
    """
    Load a configuration from a file or dictionary.
    
    Args:
        config_path: Path to configuration file
        config_dict: Configuration dictionary
        defaults: Default configuration values
        config_class: Configuration class to use
        
    Returns:
        Configuration object
    """
    # Start with defaults
    merged_config = defaults or {}
    
    # Load from file if provided
    if config_path is not None:
        file_config = load_config_from_file(config_path)
        merged_config = merge_configs(merged_config, file_config)
    
    # Override with dictionary if provided
    if config_dict is not None:
        merged_config = merge_configs(merged_config, config_dict)
    
    # Create configuration object
    config = create_config_from_dict(merged_config, config_class)
    
    # Validate configuration
    errors = validate_config(config)
    if errors:
        logger.warning(f"Configuration validation errors: {errors}")
    
    return config


def update_nested_config(
    config: Dict[str, Any],
    path: str,
    value: Any
) -> Dict[str, Any]:
    """
    Update a nested configuration value.
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path to the value (e.g. "model_adapter.max_sequence_length")
        value: New value
        
    Returns:
        Updated configuration dictionary
    """
    # Copy config to avoid modifying the original
    result = config.copy()
    
    # Split path into components
    components = path.split(".")
    
    # Navigate to the nested dictionary
    current = result
    for i, component in enumerate(components[:-1]):
        # Create nested dictionary if it doesn't exist
        if component not in current or not isinstance(current[component], dict):
            current[component] = {}
        
        current = current[component]
    
    # Set the value
    current[components[-1]] = value
    
    return result


def extract_base_config(
    enhanced_config: EnhancedTrainingConfig
) -> TrainingConfig:
    """
    Extract a base TrainingConfig from an EnhancedTrainingConfig.
    
    Args:
        enhanced_config: Enhanced training configuration
        
    Returns:
        Base training configuration
    """
    # Convert to dictionary
    config_dict = _config_to_dict(enhanced_config)
    
    # Remove enhanced components
    enhanced_keys = [
        "model_adapter", "dataset_adapter", "training_strategy",
        "extensions", "optimization", "checkpointing"
    ]
    
    base_dict = {k: v for k, v in config_dict.items() if k not in enhanced_keys}
    
    # Create base configuration
    return create_config_from_dict(base_dict, TrainingConfig)


def get_config_diff(
    config1: Union[Dict[str, Any], EnhancedTrainingConfig, TrainingConfig],
    config2: Union[Dict[str, Any], EnhancedTrainingConfig, TrainingConfig]
) -> Dict[str, Any]:
    """
    Get the differences between two configurations.
    
    Args:
        config1: First configuration
        config2: Second configuration
        
    Returns:
        Dictionary of differences
    """
    # Convert to dictionaries
    dict1 = _config_to_dict(config1)
    dict2 = _config_to_dict(config2)
    
    # Get differences
    diff = {}
    
    # Find keys in dict2 that differ from dict1
    for key, value in dict2.items():
        if key not in dict1:
            diff[key] = {"added": value}
        elif isinstance(value, dict) and isinstance(dict1[key], dict):
            nested_diff = get_config_diff(dict1[key], value)
            if nested_diff:
                diff[key] = nested_diff
        elif dict1[key] != value:
            diff[key] = {"from": dict1[key], "to": value}
    
    # Find keys in dict1 that are not in dict2
    for key, value in dict1.items():
        if key not in dict2:
            diff[key] = {"removed": value}
    
    return diff