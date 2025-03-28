"""
Configuration utilities for QLLM.

This module provides utility functions for loading, saving, and manipulating
configuration files, leveraging the more specialized configuration
implementations in the config module.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union, Type, TypeVar

# Import shared configuration components
from src.config.config_manager import ConfigManager
from src.core.configuration import load_config_file, save_config_file

logger = logging.getLogger("qllm.utils.config")

T = TypeVar('T')


def load_config_file(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from a file (JSON or YAML).
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    # Use the ConfigManager to load the configuration
    config_manager = ConfigManager()
    return config_manager.load_config(filepath)


def save_config_file(config: Dict[str, Any], filepath: str, indent: int = 2) -> None:
    """
    Save configuration to a file (JSON or YAML).
    
    Args:
        config: Configuration dictionary
        filepath: Path to save the configuration to
        indent: JSON indentation level (for JSON files)
    """
    # Determine file type from extension
    _, ext = os.path.splitext(filepath)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Use the ConfigManager to save the configuration
    config_manager = ConfigManager()
    config_manager.save_config(config, filepath)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge override config into base config.
    
    Args:
        base: Base configuration
        override: Override configuration with new values
        
    Returns:
        dict: Merged configuration
    """
    if not override:
        return base.copy()
    
    # Create a new dictionary to avoid modifying the input
    merged = base.copy()
    
    # Merge dictionaries recursively
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override or add the value
            merged[key] = value
    
    return merged


def validate_config_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a configuration against a schema.
    
    Args:
        config: Configuration to validate
        schema: Schema to validate against
        
    Returns:
        dict: Validated configuration
    """
    # Use the ConfigManager to validate the configuration
    config_manager = ConfigManager()
    return config_manager.validate_config(config, schema)


def get_config_path(config_name: str, config_dir: Optional[str] = None) -> str:
    """
    Get the path to a configuration file.
    
    Args:
        config_name: Name of the configuration file
        config_dir: Directory containing configuration files
        
    Returns:
        str: Path to the configuration file
    """
    # Default config directories to search
    default_dirs = [
        os.path.join(os.getcwd(), "configs"),
        os.path.join(os.getcwd(), "config"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"),
    ]
    
    # Add specified directory if provided
    if config_dir:
        search_dirs = [config_dir] + default_dirs
    else:
        search_dirs = default_dirs
    
    # Add extension if not provided
    if not any(config_name.endswith(ext) for ext in [".json", ".yaml", ".yml"]):
        config_name = f"{config_name}.json"
    
    # Search for the configuration file
    for directory in search_dirs:
        config_path = os.path.join(directory, config_name)
        if os.path.exists(config_path):
            return config_path
    
    # File not found in any of the search directories
    raise FileNotFoundError(f"Configuration file '{config_name}' not found in search directories")


def update_nested_dict(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    """
    Update a nested dictionary with a value at the specified key path.
    
    Args:
        d: Dictionary to update
        keys: List of keys forming the path to the value
        value: Value to set
    """
    if not keys:
        return
    
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    
    d[keys[-1]] = value