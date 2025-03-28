"""
Data configuration for QLLM.

This module provides a simplified data configuration class that extends
the ConfigurationBase class from the core module, reducing code duplication
and ensuring consistent behavior.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union

from src.core.configuration import ConfigurationBase


@dataclass
class DataConfig(ConfigurationBase):
    """
    Data configuration for QLLM.
    
    This class defines parameters for data loading and processing,
    extending the ConfigurationBase with data-specific parameters.
    
    Note: This is a simplified version that relies on the shared ConfigurationBase
    class to reduce code duplication.
    """
    
    # Data source parameters
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Tokenization parameters
    tokenizer_name: str = "gpt2"
    tokenizer_revision: Optional[str] = None
    add_special_tokens: bool = True
    max_length: int = 512
    padding: str = "max_length"  # "max_length", "do_not_pad", or "longest"
    truncation: bool = True
    
    # Dataloader parameters
    batch_size: int = 16
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = False
    
    # Preprocessing parameters
    preprocessing_num_workers: Optional[int] = None
    preprocessing_only: bool = False
    remove_unused_columns: bool = True
    
    # Caching parameters
    cache_dir: Optional[str] = None
    use_cached_datasets: bool = True
    overwrite_cache: bool = False
    
    # Text processing parameters
    text_column_name: str = "text"
    label_column_name: str = "label"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    
    # Special parameters for dialogue datasets
    dialogue_mode: bool = False
    input_column_name: Optional[str] = None
    response_column_name: Optional[str] = None
    
    # Custom datasets parameters
    dataset_config: Optional[Dict[str, Any]] = None
    tokenizer_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Perform validation after initialization."""
        # If dialogue mode is enabled, set column names if not provided
        if self.dialogue_mode:
            if self.input_column_name is None:
                self.input_column_name = "input"
            if self.response_column_name is None:
                self.response_column_name = "response"
    
    def validate(self) -> List[str]:
        """
        Validate data configuration values.
        
        Returns:
            List of validation error messages, empty if valid
        """
        errors = super().validate()
        
        # Check if at least one data source is specified
        if (self.dataset_name is None and 
            self.train_file is None and 
            self.validation_file is None and 
            self.test_file is None):
            errors.append(
                "At least one of dataset_name, train_file, validation_file, or "
                "test_file must be specified"
            )
        
        # Validate tokenizer name
        if not self.tokenizer_name:
            errors.append("tokenizer_name cannot be empty")
        
        # Validate max_length
        if self.max_length <= 0:
            errors.append(f"max_length must be positive, got {self.max_length}")
        
        # Validate padding
        valid_padding = ["max_length", "do_not_pad", "longest"]
        if self.padding not in valid_padding:
            errors.append(
                f"padding must be one of {valid_padding}, got {self.padding}"
            )
        
        # Validate batch_size
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        
        # Validate dataloader_num_workers
        if self.dataloader_num_workers < 0:
            errors.append(
                f"dataloader_num_workers must be non-negative, got {self.dataloader_num_workers}"
            )
        
        # Validate preprocessing_num_workers
        if (self.preprocessing_num_workers is not None and 
            self.preprocessing_num_workers < 0):
            errors.append(
                f"preprocessing_num_workers must be non-negative, got {self.preprocessing_num_workers}"
            )
        
        # Validate dialogue mode settings
        if self.dialogue_mode:
            if not self.input_column_name:
                errors.append("input_column_name must be specified in dialogue_mode")
            if not self.response_column_name:
                errors.append("response_column_name must be specified in dialogue_mode")
        
        return errors
    
    def has_custom_dataset(self) -> bool:
        """
        Check if a custom dataset is specified.
        
        Returns:
            True if custom dataset files are provided
        """
        return (
            self.train_file is not None or
            self.validation_file is not None or
            self.test_file is not None
        )
    
    def has_huggingface_dataset(self) -> bool:
        """
        Check if a Hugging Face dataset is specified.
        
        Returns:
            True if a Hugging Face dataset is specified
        """
        return self.dataset_name is not None
    
    def get_train_file_extension(self) -> Optional[str]:
        """
        Get the extension of the training file.
        
        Returns:
            File extension (e.g., 'json', 'csv') or None if no training file
        """
        if self.train_file is None:
            return None
        
        file_parts = self.train_file.split('.')
        if len(file_parts) > 1:
            return file_parts[-1].lower()
        return None
    
    def get_tokenizer_config(self) -> Dict[str, Any]:
        """
        Get the complete tokenizer configuration.
        
        Returns:
            Dictionary with tokenizer configuration
        """
        config = {
            "padding": self.padding,
            "truncation": self.truncation,
            "max_length": self.max_length,
            "add_special_tokens": self.add_special_tokens
        }
        
        # Add any additional tokenizer config parameters
        if self.tokenizer_config is not None:
            config.update(self.tokenizer_config)
        
        return config
    
    def get_dataset_kwargs(self) -> Dict[str, Any]:
        """
        Get keyword arguments for dataset loading.
        
        Returns:
            Dictionary with dataset loading arguments
        """
        kwargs = {
            "cache_dir": self.cache_dir,
            "use_auth_token": None  # Set to token if needed
        }
        
        # Add dataset config if specified
        if self.dataset_config_name is not None:
            kwargs["name"] = self.dataset_config_name
        
        # Add any additional dataset config parameters
        if self.dataset_config is not None:
            kwargs.update(self.dataset_config)
        
        return kwargs
    
    def get_dataloader_kwargs(self) -> Dict[str, Any]:
        """
        Get keyword arguments for dataloader creation.
        
        Returns:
            Dictionary with dataloader arguments
        """
        return {
            "batch_size": self.batch_size,
            "num_workers": self.dataloader_num_workers,
            "pin_memory": self.dataloader_pin_memory,
            "drop_last": self.dataloader_drop_last
        }