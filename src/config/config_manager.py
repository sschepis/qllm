"""
Configuration manager for QLLM.

This module provides utilities for loading, saving, and managing 
configuration for model training and evaluation using the strategy pattern
for flexible configuration handling.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List, Type, TypeVar, Generic
from dataclasses import is_dataclass, asdict

from src.core.configuration import ConfigurationBase
from src.config.config_strategy import (
    ConfigurationStrategy, 
    JsonConfigStrategy,
    YamlConfigStrategy,
    EnvConfigStrategy,
    DictConfigStrategy,
    ArgsConfigStrategy
)
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.config.data_config import DataConfig
from src.config.config_schema import get_schema

# Setup logger
logger = logging.getLogger("qllm.config")

# Type variable for configuration classes
T = TypeVar('T', bound=ConfigurationBase)


class ConfigManager:
    """
    Configuration manager for QLLM.
    
    This class manages loading, validation, and saving of configurations
    using the strategy pattern. It centralizes configuration handling and
    eliminates code duplication across the codebase.
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        # Load the schema
        self.schema = get_schema()
        
        # Register default strategies
        self.strategies = {
            "json": JsonConfigStrategy(),
            "yaml": YamlConfigStrategy(),
            "env": EnvConfigStrategy(),
            "dict": DictConfigStrategy(),
            "args": ArgsConfigStrategy()
        }
    
    def register_strategy(self, name: str, strategy: ConfigurationStrategy) -> None:
        """
        Register a new configuration strategy.
        
        Args:
            name: Name of the strategy
            strategy: Strategy implementation
        """
        self.strategies[name] = strategy
    
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
    
    def load_config(
        self, 
        config_path: str,
        strategy: str = "json"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load configuration from a file using the specified strategy.
        
        Args:
            config_path: Path to the configuration file
            strategy: Name of the strategy to use
            
        Returns:
            Dictionary with configuration sections
            
        Raises:
            ValueError: If the strategy is not registered
            FileNotFoundError: If the configuration file doesn't exist
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown configuration strategy: {strategy}")
        
        # Get the strategy
        strategy_impl = self.strategies[strategy]
        
        # Load the configuration
        try:
            config = strategy_impl.load(config_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            raise
        
        # Ensure all sections exist
        for section in ["model", "training", "data"]:
            if section not in config:
                config[section] = {}
        
        return config
    
    def save_config(
        self, 
        config: Dict[str, Dict[str, Any]], 
        config_path: str,
        strategy: str = "json"
    ) -> None:
        """
        Save configuration to a file using the specified strategy.
        
        Args:
            config: Configuration dictionary to save
            config_path: Path to save the configuration to
            strategy: Name of the strategy to use
            
        Raises:
            ValueError: If the configuration is invalid or the strategy is not registered
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown configuration strategy: {strategy}")
        
        # Validate configuration
        errors = self.schema.validate(config)
        if errors:
            error_msg = "Configuration validation failed:\n\n"
            for error in errors:
                error_msg += f"  - {error}\n"
            raise ValueError(error_msg)
        
        # Get the strategy
        strategy_impl = self.strategies[strategy]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # Save configuration
        try:
            strategy_impl.save(config, config_path)
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
            raise
    
    def merge_configs(
        self, 
        base_config: Dict[str, Dict[str, Any]], 
        override_config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
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
    
    def from_command_line_args(
        self, 
        args: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create a configuration from command line arguments.
        
        Args:
            args: Dictionary of command line arguments
            
        Returns:
            Configuration dictionary
        """
        # Use the Args strategy to convert command line arguments to config
        strategy = self.strategies["args"]
        return strategy.load(args)
    
    def to_config_classes(
        self, 
        config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Union[ModelConfig, TrainingConfig, DataConfig]]:
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
    
    def from_config_classes(
        self,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        data_config: Optional[DataConfig] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert configuration classes to a configuration dictionary.
        
        Args:
            model_config: Model configuration class instance
            training_config: Training configuration class instance
            data_config: Data configuration class instance
            
        Returns:
            Configuration dictionary
        """
        config = {
            "model": {},
            "training": {},
            "data": {}
        }
        
        if model_config is not None:
            config["model"] = model_config.to_dict()
        
        if training_config is not None:
            config["training"] = training_config.to_dict()
        
        if data_config is not None:
            config["data"] = data_config.to_dict()
        
        return config
    
    def validate_config(
        self, 
        config: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Validate a configuration against the schema.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of error messages, empty if validation passes
        """
        return self.schema.validate(config)
    
    def load_config_class(
        self, 
        config_path: str,
        config_class: Type[T],
        section: Optional[str] = None,
        strategy: str = "json"
    ) -> T:
        """
        Load a specific configuration class from a file.
        
        Args:
            config_path: Path to the configuration file
            config_class: Configuration class to instantiate
            section: Section to load from (if None, assumes the whole file is for this class)
            strategy: Name of the strategy to use
            
        Returns:
            Configuration class instance
            
        Raises:
            ValueError: If the strategy is not registered
            FileNotFoundError: If the configuration file doesn't exist
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown configuration strategy: {strategy}")
        
        # Get the strategy
        strategy_impl = self.strategies[strategy]
        
        # Load the configuration
        try:
            config_data = strategy_impl.load(config_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            raise
        
        # Extract section if specified
        if section is not None and section in config_data:
            config_data = config_data[section]
        
        # Convert to configuration class
        if issubclass(config_class, ConfigurationBase):
            return config_class.from_dict(config_data)
        else:
            # For regular classes, create instance and set attributes
            instance = config_class()
            for key, value in config_data.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            return instance
    
    def save_config_class(
        self,
        config_obj: T,
        config_path: str,
        section: Optional[str] = None,
        strategy: str = "json"
    ) -> None:
        """
        Save a configuration class to a file.
        
        Args:
            config_obj: Configuration class instance
            config_path: Path to save the configuration to
            section: Section to save as (if None, saves the whole object)
            strategy: Name of the strategy to use
            
        Raises:
            ValueError: If the strategy is not registered
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown configuration strategy: {strategy}")
        
        # Convert configuration class to dictionary
        if isinstance(config_obj, ConfigurationBase):
            config_dict = config_obj.to_dict()
        elif is_dataclass(config_obj):
            config_dict = asdict(config_obj)
        else:
            config_dict = {k: v for k, v in config_obj.__dict__.items() 
                          if not k.startswith('_')}
        
        # Wrap in section if specified
        if section is not None:
            config_dict = {section: config_dict}
        
        # Get the strategy
        strategy_impl = self.strategies[strategy]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # Save configuration
        try:
            strategy_impl.save(config_dict, config_path)
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
            raise