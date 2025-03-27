"""
Configuration utilities for QLLM evaluation.

This module provides helper functions and classes for managing evaluation
configurations and settings.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union


class EvaluationConfig:
    """
    Configuration class for QLLM evaluation.
    
    This class handles the configuration settings for evaluation runs,
    providing defaults and validation.
    """
    
    DEFAULT_EXTENSIONS = ["multimodal", "memory", "quantum"]
    DEFAULT_METRICS = ["perplexity", "parameter_efficiency", "inference_speed"]
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluation configuration.
        
        Args:
            config_dict: Optional dictionary containing configuration settings
        """
        self.config = config_dict or {}
        self._set_defaults()
        self._validate()
    
    def _set_defaults(self):
        """Set default values for missing configuration items."""
        if "extensions_to_evaluate" not in self.config:
            self.config["extensions_to_evaluate"] = self.DEFAULT_EXTENSIONS
        
        if "metrics" not in self.config:
            self.config["metrics"] = self.DEFAULT_METRICS
        
        if "datasets" not in self.config:
            self.config["datasets"] = {}
        
        if "run_ablation_studies" not in self.config:
            self.config["run_ablation_studies"] = True
        
        if "output_dir" not in self.config:
            self.config["output_dir"] = "evaluation_results"
    
    def _validate(self):
        """Validate configuration settings."""
        # Validate extensions
        valid_extensions = set(["multimodal", "memory", "quantum", "baseline", "integrated"])
        for ext in self.config["extensions_to_evaluate"]:
            if ext not in valid_extensions:
                raise ValueError(f"Invalid extension: {ext}. Must be one of {valid_extensions}")
        
        # Validate metrics
        valid_metrics = set([
            "perplexity", "parameter_efficiency", "memory_usage",
            "inference_speed", "generation_diversity"
        ])
        for metric in self.config["metrics"]:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not present
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return self.config
    
    @classmethod
    def from_file(cls, file_path: str) -> 'EvaluationConfig':
        """
        Load configuration from a JSON file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            EvaluationConfig instance
        """
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            file_path: Path to save configuration file
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.config, f, indent=2)


def load_evaluation_config(config_path: str) -> Dict[str, Any]:
    """
    Load evaluation configuration from a file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing evaluation configuration
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_default_config() -> Dict[str, Any]:
    """
    Create a default evaluation configuration.
    
    Returns:
        Dictionary containing default configuration
    """
    return {
        "extensions_to_evaluate": ["multimodal", "memory", "quantum", "baseline", "integrated"],
        "metrics": ["perplexity", "parameter_efficiency", "memory_usage", "inference_speed"],
        "datasets": {
            "text": ["This is a test input for evaluation."],
            "inference_inputs": ["Generate a response about quantum computing."],
            "generation_prompts": ["Complete this thought:"]
        },
        "run_ablation_studies": True,
        "output_dir": "evaluation_results"
    }