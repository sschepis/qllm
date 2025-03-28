"""
Unified configuration handling for QLLM.

This module provides a comprehensive configuration system that consolidates
duplicated configuration code found across the codebase. It implements
the Strategy pattern to support different configuration types and sources.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Set, Type, TypeVar, Generic, Protocol, cast
from dataclasses import dataclass, field, asdict, is_dataclass, fields, MISSING
from abc import ABC, abstractmethod


# Setup logger
logger = logging.getLogger("qllm.configuration")


# Type variable for configuration classes
T = TypeVar('T')


class ConfigurationBase:
    """
    Base class for all configuration objects.
    
    This class provides common functionality for configuration classes,
    including serialization, validation, and merging.
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        if is_dataclass(self):
            return asdict(self)
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigurationBase':
        """Create a configuration from a dictionary."""
        if is_dataclass(cls):
            # Filter out unknown fields for dataclasses
            field_names = {f.name for f in fields(cls)}
            filtered_data = {k: v for k, v in data.items() if k in field_names}
            return cls(**filtered_data)
        else:
            # For regular classes, create instance and set attributes
            instance = cls()
            for key, value in data.items():
                if not key.startswith('_'):  # Skip private attributes
                    setattr(instance, key, value)
            return instance
    
    def merge(self, other: Union[Dict[str, Any], 'ConfigurationBase']) -> 'ConfigurationBase':
        """
        Merge another configuration or dictionary into this one.
        
        Args:
            other: Another configuration object or dictionary to merge
            
        Returns:
            The merged configuration (self)
        """
        if isinstance(other, dict):
            update_dict = other
        else:
            update_dict = other.to_dict()
            
        for key, value in update_dict.items():
            if not key.startswith('_'):  # Skip private attributes
                setattr(self, key, value)
                
        return self
    
    def validate(self) -> List[str]:
        """
        Validate the configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # By default, just check types if this is a dataclass
        if is_dataclass(self):
            for field_info in fields(self):
                field_name = field_info.name
                field_value = getattr(self, field_name)
                
                # Skip validation if the field was not provided (using default)
                if field_info.default is not MISSING or field_info.default_factory is not MISSING:
                    if field_value is None:
                        continue
                
                # Validate type if specified
                if field_info.type:
                    expected_type = field_info.type
                    # Handle Optional types
                    if hasattr(expected_type, "__origin__") and expected_type.__origin__ is Union:
                        if type(None) in expected_type.__args__:
                            if field_value is None:
                                continue
                            # Extract non-None type for validation
                            non_none_types = [t for t in expected_type.__args__ if t is not type(None)]
                            if len(non_none_types) == 1:
                                expected_type = non_none_types[0]
                    
                    # Check if value matches expected type
                    if field_value is not None and not isinstance(field_value, expected_type):
                        errors.append(
                            f"Field '{field_name}' has type '{type(field_value).__name__}' but "
                            f"expected '{expected_type.__name__}'"
                        )
        
        return errors
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return str(self.to_dict())


class ConfigurationStrategy(ABC, Generic[T]):
    """
    Strategy interface for loading and saving configurations.
    
    This class defines the interface for different configuration strategies,
    such as loading from JSON, YAML, environment variables, etc.
    """
    
    @abstractmethod
    def load(self, source: str) -> Dict[str, Any]:
        """
        Load configuration from a source.
        
        Args:
            source: Source to load from (e.g., file path, string content)
            
        Returns:
            Dictionary containing configuration data
        """
        pass
    
    @abstractmethod
    def save(self, config: Dict[str, Any], destination: str) -> None:
        """
        Save configuration to a destination.
        
        Args:
            config: Configuration data to save
            destination: Destination to save to (e.g., file path)
        """
        pass


class JsonConfigurationStrategy(ConfigurationStrategy):
    """Configuration strategy for JSON files."""
    
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
        
        with open(source, 'r') as f:
            return json.load(f)
    
    def save(self, config: Dict[str, Any], destination: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            config: Configuration data to save
            destination: Path to save the configuration to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        
        with open(destination, 'w') as f:
            json.dump(config, f, indent=2)


class YamlConfigurationStrategy(ConfigurationStrategy):
    """Configuration strategy for YAML files."""
    
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
        
        with open(source, 'r') as f:
            return yaml.safe_load(f)
    
    def save(self, config: Dict[str, Any], destination: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config: Configuration data to save
            destination: Path to save the configuration to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        
        with open(destination, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


class EnvironmentConfigurationStrategy(ConfigurationStrategy):
    """Configuration strategy for environment variables."""
    
    def __init__(self, prefix: str = "QLLM_"):
        """
        Initialize the environment configuration strategy.
        
        Args:
            prefix: Prefix for environment variables
        """
        self.prefix = prefix
    
    def load(self, source: str = "") -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Args:
            source: Ignored, as we're loading from environment
            
        Returns:
            Dictionary containing configuration data
        """
        config = {}
        prefix_len = len(self.prefix)
        
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Convert environment variable to config key
                config_key = key[prefix_len:].lower()
                
                # Try to parse as JSON for structured values
                try:
                    config[config_key] = json.loads(value)
                except json.JSONDecodeError:
                    # If not valid JSON, use as string
                    config[config_key] = value
        
        return config
    
    def save(self, config: Dict[str, Any], destination: str = "") -> None:
        """
        Save is not applicable for environment variables.
        
        Args:
            config: Configuration data to save
            destination: Ignored for environment variables
            
        Raises:
            NotImplementedError: This operation is not supported
        """
        raise NotImplementedError("Saving to environment variables is not supported")


class DictConfigurationStrategy(ConfigurationStrategy):
    """Configuration strategy for dictionary objects."""
    
    def load(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load configuration from a dictionary.
        
        Args:
            source: Dictionary containing configuration data
            
        Returns:
            Dictionary containing configuration data
            
        Raises:
            TypeError: If source is not a dictionary
        """
        if not isinstance(source, dict):
            raise TypeError(f"Expected dictionary, got {type(source).__name__}")
        
        return source
    
    def save(self, config: Dict[str, Any], destination: Any) -> None:
        """
        Save is not applicable for dictionary objects.
        
        Args:
            config: Configuration data to save
            destination: Ignored for dictionaries
            
        Raises:
            NotImplementedError: This operation is not supported
        """
        raise NotImplementedError("Saving to dictionary is not supported")


class ConfigurationManager:
    """
    Central manager for configuration handling.
    
    This class manages the loading, validation, and saving of configurations
    using different strategies. It centralizes configuration logic that was
    previously duplicated across multiple files.
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        # Register configuration strategies
        self.strategies = {
            "json": JsonConfigurationStrategy(),
            "yaml": YamlConfigurationStrategy(),
            "env": EnvironmentConfigurationStrategy(),
            "dict": DictConfigurationStrategy()
        }
        
        # Default configuration schema validators
        self.validators: Dict[str, Any] = {}
    
    def register_strategy(self, name: str, strategy: ConfigurationStrategy) -> None:
        """
        Register a new configuration strategy.
        
        Args:
            name: Name of the strategy
            strategy: Strategy implementation
        """
        self.strategies[name] = strategy
    
    def register_validator(self, config_type: str, validator: Any) -> None:
        """
        Register a validator for a configuration type.
        
        Args:
            config_type: Configuration type name
            validator: Validator object or function
        """
        self.validators[config_type] = validator
    
    def load_config(
        self, 
        source: Any,
        strategy_name: str = "json",
        config_class: Optional[Type[T]] = None,
        validate: bool = True
    ) -> Union[Dict[str, Any], T]:
        """
        Load configuration from a source using the specified strategy.
        
        Args:
            source: Source to load from (e.g., file path)
            strategy_name: Name of the strategy to use
            config_class: Optional class to instantiate with the loaded config
            validate: Whether to validate the configuration
            
        Returns:
            Configuration dictionary or instance of config_class
            
        Raises:
            ValueError: If the strategy is not registered
            ValidationError: If validation fails
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown configuration strategy: {strategy_name}")
        
        # Get the strategy
        strategy = self.strategies[strategy_name]
        
        # Load the configuration
        config = strategy.load(source)
        
        # Validate if requested
        if validate:
            errors = self._validate_config(config, config_class)
            if errors:
                error_msg = "Configuration validation failed:\n\n"
                for error in errors:
                    error_msg += f"  - {error}\n"
                raise ValueError(error_msg)
        
        # Convert to config class if specified
        if config_class is not None:
            if issubclass(config_class, ConfigurationBase):
                return config_class.from_dict(config)
            else:
                # For regular classes, create instance and set attributes
                instance = config_class()
                for key, value in config.items():
                    if hasattr(instance, key):
                        setattr(instance, key, value)
                return instance
        
        return config
    
    def save_config(
        self,
        config: Union[Dict[str, Any], ConfigurationBase],
        destination: str,
        strategy_name: str = "json"
    ) -> None:
        """
        Save configuration to a destination using the specified strategy.
        
        Args:
            config: Configuration to save
            destination: Destination to save to
            strategy_name: Name of the strategy to use
            
        Raises:
            ValueError: If the strategy is not registered
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown configuration strategy: {strategy_name}")
        
        # Get the strategy
        strategy = self.strategies[strategy_name]
        
        # Convert to dictionary if it's a configuration object
        if isinstance(config, ConfigurationBase):
            config_dict = config.to_dict()
        else:
            config_dict = config
        
        # Save the configuration
        strategy.save(config_dict, destination)
    
    def merge_configs(
        self, 
        base_config: Union[Dict[str, Any], ConfigurationBase],
        override_config: Union[Dict[str, Any], ConfigurationBase]
    ) -> Union[Dict[str, Any], ConfigurationBase]:
        """
        Merge two configurations, with override_config taking precedence.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override base values
            
        Returns:
            Merged configuration
        """
        # Handle different types
        if isinstance(base_config, ConfigurationBase):
            if isinstance(override_config, dict):
                return base_config.merge(override_config)
            else:
                return base_config.merge(override_config.to_dict())
        elif isinstance(override_config, ConfigurationBase):
            # Create a copy of the override config and merge base into it
            result = type(override_config).from_dict(override_config.to_dict())
            # Only update values that don't exist in override
            base_dict = base_config.to_dict() if isinstance(base_config, ConfigurationBase) else base_config
            for key, value in base_dict.items():
                if key not in override_config.to_dict():
                    setattr(result, key, value)
            return result
        else:
            # Both are dictionaries
            result = base_config.copy()
            result.update(override_config)
            return result
    
    def create_config_from_env(
        self,
        config_class: Type[T],
        prefix: str = "QLLM_",
        validate: bool = True
    ) -> T:
        """
        Create a configuration from environment variables.
        
        Args:
            config_class: Class to instantiate with the loaded config
            prefix: Prefix for environment variables
            validate: Whether to validate the configuration
            
        Returns:
            Instance of config_class
        """
        # Use environment strategy with the specified prefix
        env_strategy = EnvironmentConfigurationStrategy(prefix)
        config_dict = env_strategy.load()
        
        # Validate if requested
        if validate:
            errors = self._validate_config(config_dict, config_class)
            if errors:
                error_msg = "Environment configuration validation failed:\n\n"
                for error in errors:
                    error_msg += f"  - {error}\n"
                raise ValueError(error_msg)
        
        # Convert to config class
        if issubclass(config_class, ConfigurationBase):
            return config_class.from_dict(config_dict)
        else:
            # For regular classes, create instance and set attributes
            instance = config_class()
            for key, value in config_dict.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            return instance
    
    def _validate_config(
        self,
        config: Dict[str, Any],
        config_class: Optional[Type[Any]] = None
    ) -> List[str]:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            config_class: Optional class for validation
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Use class validation if available
        if config_class is not None:
            if issubclass(config_class, ConfigurationBase):
                # Create instance and validate
                instance = config_class.from_dict(config)
                class_errors = instance.validate()
                errors.extend(class_errors)
            
            # Use registered validator if available
            config_type = config_class.__name__
            if config_type in self.validators:
                validator = self.validators[config_type]
                validator_errors = validator(config)
                errors.extend(validator_errors)
        
        return errors


# Common configuration classes

@dataclass
class ModelConfig(ConfigurationBase):
    """Model configuration parameters."""
    
    vocab_size: int = 30000
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    max_seq_length: int = 512
    use_cache: bool = True
    tie_word_embeddings: bool = True
    
    # Additional model-specific parameters
    extensions: Optional[Dict[str, Any]] = None
    primes: Optional[List[int]] = None
    base_dim: Optional[int] = None
    max_iterations: Optional[int] = None
    entropy_threshold: Optional[float] = None
    use_prime_mask: bool = False
    enable_hcw: bool = False
    memory_size: Optional[int] = None
    memory_key_dim: Optional[int] = None


@dataclass
class TrainingConfig(ConfigurationBase):
    """Training configuration parameters."""
    
    batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    max_epochs: int = 3
    warmup_steps: int = 0
    accumulation_steps: int = 1
    save_steps: int = 100
    eval_steps: int = 50
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"
    seed: int = 42
    use_mixed_precision: bool = False
    auto_resume: bool = False
    
    # Training strategy and model type
    training_strategy: str = "standard"
    model_type: str = "standard"
    
    # Extensions
    enabled_extensions: List[str] = field(default_factory=list)
    extension_config: Optional[Dict[str, Any]] = None


@dataclass
class DataConfig(ConfigurationBase):
    """Data configuration parameters."""
    
    dataset_name: Optional[str] = None
    tokenizer_name: str = "gpt2"
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    max_length: int = 512
    preprocessing_num_workers: Optional[int] = None
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Dataset-specific parameters
    dataset_config: Optional[Dict[str, Any]] = None
    tokenizer_config: Optional[Dict[str, Any]] = None