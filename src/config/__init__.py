"""
Configuration module for QLLM.

This module provides configuration classes and utilities for managing
model, training, and data configurations using a strategy pattern for
flexible configuration loading and saving.

The config module has been refactored to reduce code duplication and
improve maintainability by leveraging the core configuration utilities.
"""

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.config.data_config import DataConfig
from src.config.config_manager import ConfigManager
from src.config.config_strategy import (
    ConfigurationStrategy,
    JsonConfigStrategy,
    YamlConfigStrategy,
    EnvConfigStrategy,
    DictConfigStrategy
)
from src.config.config_schema import get_schema

__all__ = [
    # Configuration classes
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    
    # Configuration management
    'ConfigManager',
    
    # Configuration strategies
    'ConfigurationStrategy',
    'JsonConfigStrategy',
    'YamlConfigStrategy',
    'EnvConfigStrategy',
    'DictConfigStrategy',
    
    # Schema handling
    'get_schema'
]