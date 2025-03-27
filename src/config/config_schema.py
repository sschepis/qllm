"""
Configuration schema for the Quantum Resonance Language Model.

This module defines the schema for validating configuration files,
including required fields, types, and validation rules.
"""

from typing import List, Dict, Any, Optional
import re


class ConfigSchema:
    """Configuration schema with validation."""
    
    def __init__(self):
        """Initialize the schema."""
        self.schema = {
            "model": {
                "vocab_size": {
                    "type": "int",
                    "range": [1000, 1000000],
                    "default": 50257,
                    "help": "Size of the vocabulary"
                },
                "hidden_dim": {
                    "type": "int",
                    "range": [64, 4096],
                    "default": 768,
                    "help": "Hidden dimension size"
                },
                "num_layers": {
                    "type": "int",
                    "range": [1, 48],
                    "default": 12,
                    "help": "Number of transformer layers"
                },
                "num_heads": {
                    "type": "int",
                    "range": [1, 48],
                    "default": 12,
                    "help": "Number of attention heads"
                },
                "dropout": {
                    "type": "float",
                    "range": [0.0, 0.9],
                    "default": 0.1,
                    "help": "Dropout probability"
                },
                "max_seq_length": {
                    "type": "int",
                    "range": [16, 8192],
                    "default": 1024,
                    "help": "Maximum sequence length"
                },
                "primes": {
                    "type": "list",
                    "default": [23, 29, 31, 37, 41, 43, 47],
                    "help": "Prime numbers for quantum resonance"
                },
                "max_iterations": {
                    "type": "int",
                    "range": [1, 100],
                    "default": 10,
                    "help": "Maximum iterations for resonance convergence"
                },
                "entropy_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.01,
                    "help": "Entropy threshold for convergence"
                },
                "phase_factor": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.5,
                    "help": "Phase factor for quantum adjustment"
                },
                "extensions": {
                    "type": "dict",
                    "default": {},
                    "help": "Model extensions configuration"
                },
                "extra_model_params": {
                    "type": "dict",
                    "default": {},
                    "help": "Additional model parameters"
                }
            },
            "training": {
                "batch_size": {
                    "type": "int",
                    "range": [1, 1024],
                    "default": 16,
                    "help": "Training batch size"
                },
                "eval_batch_size": {
                    "type": "int",
                    "range": [1, 1024],
                    "default": 16,
                    "help": "Evaluation batch size"
                },
                "learning_rate": {
                    "type": "float",
                    "range": [1e-6, 1.0],
                    "default": 5e-5,
                    "help": "Learning rate"
                },
                "weight_decay": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.01,
                    "help": "Weight decay"
                },
                "max_epochs": {
                    "type": "int",
                    "range": [1, 1000],
                    "default": 3,
                    "help": "Maximum number of epochs"
                },
                "training_type": {
                    "type": "str",
                    "choices": ["standard", "dialogue", "verbose"],
                    "default": "standard",
                    "help": "Type of training"
                },
                "learning_mode": {
                    "type": "str",
                    "choices": ["adaptive", "scheduled", "feedback_driven"],
                    "default": "adaptive",
                    "help": "Learning rate mode"
                },
                "device": {
                    "type": "str",
                    "optional": True,
                    "help": "Device to use (cuda, cpu, mps, auto)"
                },
                "use_mixed_precision": {
                    "type": "bool",
                    "default": True,
                    "help": "Whether to use mixed precision training"
                },
                "optimizer": {
                    "type": "str",
                    "choices": ["adamw", "adam", "sgd"],
                    "default": "adamw",
                    "help": "Optimizer to use"
                },
                "max_grad_norm": {
                    "type": "float",
                    "range": [0.1, 100.0],
                    "default": 1.0,
                    "help": "Maximum gradient norm for clipping"
                },
                "accumulation_steps": {
                    "type": "int",
                    "range": [1, 32],
                    "default": 1,
                    "help": "Gradient accumulation steps"
                },
                "lr_scheduler": {
                    "type": "str",
                    "choices": ["linear", "cosine", "step", "constant"],
                    "default": "cosine",
                    "help": "Learning rate scheduler"
                },
                "warmup_steps": {
                    "type": "int",
                    "range": [0, 10000],
                    "default": 0,
                    "help": "Number of warmup steps"
                },
                "logging_steps": {
                    "type": "int",
                    "range": [1, 1000],
                    "default": 10,
                    "help": "Logging frequency (steps)"
                },
                "save_steps": {
                    "type": "int",
                    "range": [0, 10000],
                    "default": 0,
                    "help": "Checkpoint frequency (steps). 0 for epoch-only saving."
                },
                "eval_steps": {
                    "type": "int",
                    "range": [0, 10000],
                    "default": 0,
                    "help": "Evaluation frequency (steps). 0 for epoch-only evaluation."
                },
                "save_every_epoch": {
                    "type": "bool",
                    "default": True,
                    "help": "Whether to save after each epoch"
                },
                "disable_optimizer_saving": {
                    "type": "bool",
                    "default": False,
                    "help": "Whether to disable saving optimizer state"
                },
                "output_dir": {
                    "type": "str",
                    "default": "runs/quantum_resonance",
                    "help": "Output directory"
                },
                "seed": {
                    "type": "int",
                    "range": [0, 9999],
                    "default": 42,
                    "help": "Random seed"
                },
                "extra_training_params": {
                    "type": "dict",
                    "default": {},
                    "help": "Additional training parameters"
                }
            },
            "data": {
                "dataset_name": {
                    "type": "str",
                    "default": "wikitext",
                    "help": "Dataset name"
                },
                "tokenizer_name": {
                    "type": "str",
                    "default": "gpt2",
                    "help": "Tokenizer name or path"
                },
                "train_file": {
                    "type": "str",
                    "optional": True,
                    "help": "Path to training data file"
                },
                "validation_file": {
                    "type": "str",
                    "optional": True,
                    "help": "Path to validation data file"
                },
                "test_file": {
                    "type": "str",
                    "optional": True,
                    "help": "Path to test data file"
                },
                "data_path": {
                    "type": "str",
                    "optional": True,
                    "help": "Path to data directory"
                },
                "max_length": {
                    "type": "int",
                    "range": [16, 8192],
                    "default": 512,
                    "help": "Maximum sequence length"
                },
                "stride": {
                    "type": "int",
                    "range": [16, 2048],
                    "default": 256,
                    "help": "Stride for overlapping chunks"
                },
                "preprocessing_num_workers": {
                    "type": "int",
                    "range": [1, 64],
                    "default": 4,
                    "help": "Number of preprocessing workers"
                },
                "cache_dir": {
                    "type": "str",
                    "default": ".cache",
                    "help": "Cache directory"
                },
                "return_tensors": {
                    "type": "str",
                    "choices": ["pt", "tf", "np"],
                    "default": "pt",
                    "help": "Return tensor type"
                },
                "subset_size": {
                    "type": "int",
                    "optional": True,
                    "help": "Subset size for debugging"
                },
                "system_prompt": {
                    "type": "str",
                    "optional": True,
                    "help": "System prompt for dialogue datasets"
                },
                "function_defs_path": {
                    "type": "str",
                    "optional": True,
                    "help": "Path to function definitions file"
                },
                "extra_data_params": {
                    "type": "dict",
                    "default": {},
                    "help": "Additional data parameters"
                }
            }
        }
    
    def validate(self, config: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Validate a configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of error messages, empty if validation passes
        """
        errors = []
        
        # Validate each section
        for section, section_schema in self.schema.items():
            # Check if section exists
            if section not in config:
                errors.append(f"Missing section: {section}")
                continue
            
            # Get section configuration
            section_config = config[section]
            
            # Validate each field
            for field, field_schema in section_schema.items():
                # Check if field exists
                if field not in section_config:
                    # If the field is optional, it's allowed to be missing
                    if field_schema.get("optional", False):
                        continue
                    
                    # If the field has a default value, add it to the configuration
                    if "default" in field_schema:
                        section_config[field] = field_schema["default"]
                        continue
                    
                    # Required field is missing
                    errors.append(f"Missing field: {section}.{field}")
                    continue
                
                # Get field value
                value = section_config[field]
                
                # Skip validation for None values in optional fields
                if value is None and field_schema.get("optional", False):
                    continue
                
                # Validate field value
                field_type = field_schema.get("type", "str")
                
                if field_type == "int":
                    if not isinstance(value, int):
                        errors.append(f"Expected int for {section}.{field}, got {type(value).__name__}")
                    elif "range" in field_schema:
                        min_val, max_val = field_schema["range"]
                        if value < min_val or value > max_val:
                            errors.append(f"Value for {section}.{field} must be between {min_val} and {max_val}")
                
                elif field_type == "float":
                    if not isinstance(value, (int, float)):
                        errors.append(f"Expected float for {section}.{field}, got {type(value).__name__}")
                    elif "range" in field_schema:
                        min_val, max_val = field_schema["range"]
                        if value < min_val or value > max_val:
                            errors.append(f"Value for {section}.{field} must be between {min_val} and {max_val}")
                
                elif field_type == "bool":
                    if not isinstance(value, bool):
                        errors.append(f"Expected bool for {section}.{field}, got {type(value).__name__}")
                
                elif field_type == "str":
                    if not isinstance(value, str):
                        errors.append(f"Expected str for {section}.{field}, got {type(value).__name__}")
                    elif "choices" in field_schema and value not in field_schema["choices"]:
                        choices_str = ", ".join(field_schema["choices"])
                        errors.append(f"Value for {section}.{field} must be one of: {choices_str}")
                
                elif field_type == "list":
                    if not isinstance(value, list):
                        errors.append(f"Expected list for {section}.{field}, got {type(value).__name__}")
                
                elif field_type == "dict":
                    if not isinstance(value, dict):
                        errors.append(f"Expected dict for {section}.{field}, got {type(value).__name__}")
        
        return errors


def get_schema() -> ConfigSchema:
    """
    Get the configuration schema.
    
    Returns:
        Configuration schema
    """
    return ConfigSchema()