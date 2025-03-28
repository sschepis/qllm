"""
Configuration strategies for QLLM.

This module implements the Strategy pattern for configuration management,
providing different strategies for loading and saving configurations from
various sources such as JSON, YAML, environment variables, etc.
"""

import os
import json
import yaml
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Type

# Setup logger
logger = logging.getLogger("qllm.config")


class ConfigurationStrategy(ABC):
    """
    Abstract base class for configuration strategies.
    
    This class defines the interface that all configuration strategies must implement.
    Each strategy provides methods for loading and saving configurations from different
    sources.
    """
    
    @abstractmethod
    def load(self, source: Any) -> Dict[str, Any]:
        """
        Load configuration from a source.
        
        Args:
            source: Source to load from (e.g., file path, string)
            
        Returns:
            Dictionary containing configuration data
            
        Raises:
            Exception: If loading fails
        """
        pass
    
    @abstractmethod
    def save(self, config: Dict[str, Any], destination: Any) -> None:
        """
        Save configuration to a destination.
        
        Args:
            config: Configuration data to save
            destination: Destination to save to
            
        Raises:
            Exception: If saving fails
        """
        pass


class JsonConfigStrategy(ConfigurationStrategy):
    """
    Configuration strategy for JSON files.
    
    This strategy handles loading from and saving to JSON files.
    """
    
    def load(self, source: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            source: Path to JSON file
            
        Returns:
            Dictionary containing configuration data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file isn't valid JSON
        """
        if not os.path.exists(source):
            raise FileNotFoundError(f"Configuration file not found: {source}")
        
        try:
            with open(source, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON configuration file {source}: {e}")
            raise
    
    def save(self, config: Dict[str, Any], destination: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            config: Configuration data to save
            destination: Path to save the configuration to
            
        Raises:
            OSError: If directory creation or file writing fails
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        
        with open(destination, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


class YamlConfigStrategy(ConfigurationStrategy):
    """
    Configuration strategy for YAML files.
    
    This strategy handles loading from and saving to YAML files.
    """
    
    def load(self, source: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            source: Path to YAML file
            
        Returns:
            Dictionary containing configuration data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the file isn't valid YAML
        """
        if not os.path.exists(source):
            raise FileNotFoundError(f"Configuration file not found: {source}")
        
        try:
            with open(source, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config if config is not None else {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration file {source}: {e}")
            raise
    
    def save(self, config: Dict[str, Any], destination: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config: Configuration data to save
            destination: Path to save the configuration to
            
        Raises:
            OSError: If directory creation or file writing fails
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        
        with open(destination, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)


class EnvConfigStrategy(ConfigurationStrategy):
    """
    Configuration strategy for environment variables.
    
    This strategy handles loading from environment variables with a specified prefix.
    """
    
    def __init__(self, prefix: str = "QLLM_"):
        """
        Initialize the environment configuration strategy.
        
        Args:
            prefix: Prefix for environment variables to consider
        """
        self.prefix = prefix
    
    def load(self, source: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Args:
            source: Optional parameter, ignored for this strategy
            
        Returns:
            Dictionary containing configuration data from environment variables
        """
        config = {}
        prefix_len = len(self.prefix)
        
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Convert environment variable name to config key
                # Example: QLLM_MODEL_HIDDEN_DIM -> model.hidden_dim
                config_key = key[prefix_len:].lower()
                config_key = config_key.replace("__", ".")
                
                # Try to parse value as JSON for structured values
                try:
                    parsed_value = json.loads(value)
                    self._set_nested_value(config, config_key, parsed_value)
                except json.JSONDecodeError:
                    # Handle common value conversions
                    if value.lower() == "true":
                        self._set_nested_value(config, config_key, True)
                    elif value.lower() == "false":
                        self._set_nested_value(config, config_key, False)
                    elif value.isdigit():
                        self._set_nested_value(config, config_key, int(value))
                    elif self._is_float(value):
                        self._set_nested_value(config, config_key, float(value))
                    else:
                        self._set_nested_value(config, config_key, value)
        
        return config
    
    def save(self, config: Dict[str, Any], destination: Optional[str] = None) -> None:
        """
        Saving to environment variables is not supported.
        
        Args:
            config: Configuration data to save
            destination: Ignored for this strategy
            
        Raises:
            NotImplementedError: This operation is not supported
        """
        raise NotImplementedError(
            "Saving configuration to environment variables is not supported"
        )
    
    def _is_float(self, value: str) -> bool:
        """Check if a string can be converted to a float."""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """
        Set a value in a nested dictionary based on a dotted key.
        
        Args:
            config: Dictionary to modify
            key: Dotted key path (e.g., "model.hidden_dim")
            value: Value to set
        """
        if "." not in key:
            config[key] = value
            return
            
        parts = key.split(".")
        current = config
        
        # Navigate to the nested dictionary
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        
        # Set the value
        current[parts[-1]] = value


class DictConfigStrategy(ConfigurationStrategy):
    """
    Configuration strategy for dictionary objects.
    
    This strategy handles loading from and "saving" to Python dictionaries.
    """
    
    def load(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load configuration from a dictionary.
        
        Args:
            source: Dictionary containing configuration data
            
        Returns:
            Copy of the source dictionary
            
        Raises:
            TypeError: If source is not a dictionary
        """
        if not isinstance(source, dict):
            raise TypeError(f"Expected dictionary, got {type(source).__name__}")
        
        # Return a copy to prevent modifications to the source
        return source.copy()
    
    def save(self, config: Dict[str, Any], destination: Dict[str, Any]) -> None:
        """
        Save configuration to a dictionary.
        
        Args:
            config: Configuration data to save
            destination: Dictionary to update with configuration data
            
        Raises:
            TypeError: If destination is not a dictionary
        """
        if not isinstance(destination, dict):
            raise TypeError(f"Expected dictionary, got {type(destination).__name__}")
        
        # Update the destination dictionary
        destination.update(config)


class ArgsConfigStrategy(ConfigurationStrategy):
    """
    Configuration strategy for command-line arguments.
    
    This strategy handles loading configuration from parsed command-line arguments.
    """
    
    def load(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load configuration from command-line arguments.
        
        Args:
            source: Dictionary of parsed command-line arguments
            
        Returns:
            Dictionary containing configuration data
            
        Raises:
            TypeError: If source is not a dictionary
        """
        if not isinstance(source, dict):
            raise TypeError(f"Expected dictionary of args, got {type(source).__name__}")
        
        # Convert from flat structure to nested configuration
        config = {
            "model": {},
            "training": {},
            "data": {}
        }
        
        # Map known arguments to their sections
        for arg, value in source.items():
            if value is None:
                continue
                
            # Model arguments
            if arg in ["hidden_dim", "num_layers", "num_heads", "dropout", 
                       "max_seq_length", "vocab_size", "tie_word_embeddings"]:
                config["model"][arg] = value
            
            # Training arguments
            elif arg in ["batch_size", "learning_rate", "weight_decay", "max_epochs", 
                         "training_type", "device", "output_dir", "seed", "warmup_steps",
                         "accumulation_steps", "save_steps", "eval_steps"]:
                config["training"][arg] = value
            
            # Data arguments
            elif arg in ["dataset_name", "tokenizer_name", "train_file", "validation_file", 
                         "test_file", "max_length", "preprocessing_num_workers"]:
                config["data"][arg] = value
            
            # Handle remaining arguments
            else:
                # Try to infer the section based on the argument name
                if arg.startswith("model_"):
                    section = "model"
                    key = arg[6:]  # Remove "model_" prefix
                elif arg.startswith("train_") or arg.startswith("training_"):
                    section = "training"
                    key = arg[arg.find("_")+1:]  # Remove prefix
                elif arg.startswith("data_"):
                    section = "data"
                    key = arg[5:]  # Remove "data_" prefix
                else:
                    # Default to training section for unknown arguments
                    section = "training"
                    key = arg
                
                config[section][key] = value
        
        return config
    
    def save(self, config: Dict[str, Any], destination: Any) -> None:
        """
        Saving to command-line arguments is not supported.
        
        Args:
            config: Configuration data to save
            destination: Ignored for this strategy
            
        Raises:
            NotImplementedError: This operation is not supported
        """
        raise NotImplementedError(
            "Saving configuration to command-line arguments is not supported"
        )