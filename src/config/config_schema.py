"""
Configuration schema for QLLM.

This module provides schema validation for configuration files,
ensuring that configurations have the correct structure and values.
It has been simplified to leverage the validation methods in the
config classes themselves.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Union

# Setup logger
logger = logging.getLogger("qllm.config")


class ConfigSchema:
    """
    Configuration schema validator for QLLM.
    
    This class provides validation for configuration dictionaries,
    ensuring they have the correct structure and values.
    """
    
    def __init__(self):
        """Initialize the schema validator."""
        # Required top-level sections
        self.required_sections = ["model", "training", "data"]
        
        # Required fields for each section
        self.required_fields = {
            "model": [],  # No required fields, all have defaults
            "training": [],  # No required fields, all have defaults
            "data": []  # No required fields, all have defaults
        }
        
        # Expected types for common fields
        self.field_types = {
            "model": {
                "vocab_size": int,
                "hidden_dim": int,
                "num_layers": int,
                "num_heads": int,
                "dropout": float,
                "max_seq_length": int,
                "use_cache": bool,
                "tie_word_embeddings": bool,
                "primes": list,
                "base_dim": int,
                "max_iterations": int,
                "entropy_threshold": float,
                "use_prime_mask": bool,
                "enable_hcw": bool,
                "memory_size": int,
                "memory_key_dim": int,
                "extensions": dict
            },
            "training": {
                "batch_size": int,
                "learning_rate": float,
                "weight_decay": float,
                "max_epochs": int,
                "warmup_steps": int,
                "accumulation_steps": int,
                "save_steps": int,
                "eval_steps": int,
                "checkpoint_dir": str,
                "device": str,
                "seed": int,
                "use_mixed_precision": bool,
                "training_strategy": str,
                "model_type": str,
                "enabled_extensions": list,
                "extension_config": dict
            },
            "data": {
                "dataset_name": str,
                "tokenizer_name": str,
                "train_file": str,
                "validation_file": str,
                "test_file": str,
                "max_length": int,
                "preprocessing_num_workers": int,
                "dataloader_num_workers": int,
                "dataloader_pin_memory": bool,
                "dataset_config": dict,
                "tokenizer_config": dict
            }
        }
        
        # Value constraints for certain fields
        self.constraints = {
            "model.dropout": (0.0, 1.0),  # Range (min, max)
            "training.learning_rate": (0.0, None),  # Range (min, no max)
            "training.weight_decay": (0.0, None),  # Range (min, no max)
            "training.save_strategy": {"steps", "epochs", "no"},  # Enum (set of allowed values)
            "training.eval_strategy": {"steps", "epochs", "no"},  # Enum (set of allowed values)
            "data.padding": {"max_length", "do_not_pad", "longest"}  # Enum
        }
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation error messages, empty if valid
        """
        errors = []
        
        # Check that all required sections are present
        for section in self.required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
                continue
            
            if not isinstance(config[section], dict):
                errors.append(f"Section {section} must be a dictionary")
                continue
            
            # Check that all required fields are present
            for field in self.required_fields[section]:
                if field not in config[section]:
                    errors.append(f"Missing required field: {section}.{field}")
        
        # Check field types
        for section_name, section_data in config.items():
            if section_name not in self.field_types:
                continue
                
            if not isinstance(section_data, dict):
                continue
                
            section_types = self.field_types[section_name]
            for field_name, field_value in section_data.items():
                # Skip None values (they're valid for any type)
                if field_value is None:
                    continue
                    
                # Skip fields without type information
                if field_name not in section_types:
                    continue
                    
                expected_type = section_types[field_name]
                
                # Check field type
                if not isinstance(field_value, expected_type):
                    errors.append(
                        f"Field {section_name}.{field_name} has incorrect type: "
                        f"expected {expected_type.__name__}, got {type(field_value).__name__}"
                    )
        
        # Check value constraints
        for constraint_key, constraint_value in self.constraints.items():
            section_name, field_name = constraint_key.split(".")
            
            # Skip if section or field doesn't exist
            if section_name not in config or not isinstance(config[section_name], dict):
                continue
                
            if field_name not in config[section_name] or config[section_name][field_name] is None:
                continue
            
            field_value = config[section_name][field_name]
            
            # Check range constraint
            if isinstance(constraint_value, tuple):
                min_val, max_val = constraint_value
                
                if min_val is not None and field_value < min_val:
                    errors.append(
                        f"Field {constraint_key} value {field_value} is less than minimum {min_val}"
                    )
                
                if max_val is not None and field_value > max_val:
                    errors.append(
                        f"Field {constraint_key} value {field_value} is greater than maximum {max_val}"
                    )
            
            # Check enum constraint
            elif isinstance(constraint_value, set):
                if field_value not in constraint_value:
                    errors.append(
                        f"Field {constraint_key} value '{field_value}' not in allowed values: {constraint_value}"
                    )
        
        return errors


def get_schema() -> ConfigSchema:
    """
    Get a schema validator instance.
    
    Returns:
        ConfigSchema instance
    """
    return ConfigSchema()


class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors."""
    
    def __init__(self, errors: List[str]):
        """
        Initialize the exception.
        
        Args:
            errors: List of validation error messages
        """
        self.errors = errors
        message = f"Configuration validation failed with {len(errors)} errors:\n"
        message += "\n".join(f"- {error}" for error in errors)
        super().__init__(message)


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate a configuration dictionary and raise an exception if invalid.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ConfigValidationError: If the configuration is invalid
    """
    schema = get_schema()
    errors = schema.validate(config)
    if errors:
        raise ConfigValidationError(errors)