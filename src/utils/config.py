"""
Configuration utilities for the Quantum Resonance Language Model.
Provides structured configuration management and serialization.
"""

import os
import json
import yaml
import dataclasses
from typing import Dict, Any, Optional, List, Union, Type, TypeVar, get_type_hints
from dataclasses import dataclass, field


T = TypeVar('T')


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Core architecture
    vocab_size: int = 50257
    hidden_dim: int = 768
    num_layers: int = 4
    num_heads: int = 12
    base_dim: int = 64    # Base dimension for Prime Hilbert Encoder
    max_seq_length: int = 1024  # Maximum sequence length
    
    # FFN properties
    ff_dim: int = 3072
    activation: str = "gelu"
    
    # Resonance properties
    max_iterations: int = 10
    resonance_epsilon: float = 0.1
    resonance_momentum: float = 0.2
    entropy_penalty: float = 0.05
    
    # Phase modulation
    use_phase_modulation: bool = True
    phase_factor: float = 0.1
    freq_factor: float = 2.0
    phase_offset: float = 0.0
    
    # Temperature scheduling
    use_temperature_scheduling: bool = True
    beta_0: float = 0.8
    beta_delta: float = 0.2
    
    # Convergence
    use_cosine_convergence: bool = True
    convergence_threshold: float = 0.95
    
    # Quantum subspace settings
    primes: List[int] = field(default_factory=lambda: [7, 11, 13, 17, 19])
    use_prime_embeddings: bool = True
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    ff_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Other settings
    tie_weights: bool = True
    max_position_embeddings: int = 1024
    initializer_range: float = 0.02
    pad_token_id: int = 0


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Basic training settings
    output_dir: str = "runs/quantum_resonance"
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True
    
    # Optimization settings
    optimizer: str = "adamw"
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Training schedule
    batch_size: int = 32
    max_epochs: int = 10
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    
    # Learning rate schedule
    lr_scheduler: str = "linear"
    num_cycles: int = 1  # For cosine scheduler
    
    # Evaluation settings
    eval_batch_size: int = 32
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 10
    logging_dir: str = "logs"
    log_level: str = "info"
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Dataset settings
    dataset_name: str = "wikitext"
    dataset_config_name: str = "wikitext-103-raw-v1"
    cache_dir: str = ".cache"
    
    # Tokenization
    tokenizer_name: str = "gpt2"
    max_length: int = 512
    stride: int = 128
    
    # Data loading
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Preprocessing
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    
    # Data splitting
    validation_split_percentage: int = 5
    test_split_percentage: int = 5


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    max_length: int = 200
    min_length: int = 10
    do_sample: bool = True
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    bad_words_ids: Optional[List[List[int]]] = None
    num_return_sequences: int = 1
    use_cache: bool = True


def get_default_configs() -> Dict[str, Any]:
    """
    Get default configurations.
    
    Returns:
        dict: Dictionary with default configurations
    """
    return {
        "model": ModelConfig(),
        "training": TrainingConfig(),
        "data": DataConfig(),
        "generation": GenerationConfig()
    }


def save_dataclass_to_json(obj: Any, filepath: str, indent: int = 2) -> None:
    """
    Save a dataclass instance to a JSON file.
    
    Args:
        obj: Dataclass instance to save
        filepath: Path to save the JSON file
        indent: JSON indentation level
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert to dict, handling non-serializable objects
    if dataclasses.is_dataclass(obj):
        obj_dict = dataclasses.asdict(obj)
    else:
        obj_dict = obj.__dict__
    
    # Handle non-serializable objects
    for key, value in obj_dict.items():
        if isinstance(value, (set, frozenset)):
            obj_dict[key] = list(value)
        elif hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool, list, dict)):
            obj_dict[key] = dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value.__dict__
    
    # Write to file
    with open(filepath, 'w') as f:
        json.dump(obj_dict, f, indent=indent)


def load_dataclass_from_json(cls: Type[T], filepath: str) -> T:
    """
    Load a dataclass instance from a JSON file.
    
    Args:
        cls: Dataclass type to create
        filepath: Path to the JSON file
        
    Returns:
        T: Instance of the dataclass
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    # Load JSON
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Get type hints to handle nested dataclasses
    type_hints = get_type_hints(cls)
    
    # Process nested dataclasses
    for key, hint in type_hints.items():
        if key in data and hasattr(hint, "__origin__") and hint.__origin__ is Union:
            # Handle Optional types
            inner_types = [t for t in hint.__args__ if t is not type(None)]
            if len(inner_types) == 1 and dataclasses.is_dataclass(inner_types[0]):
                if data[key] is not None:
                    data[key] = _dict_to_dataclass(inner_types[0], data[key])
        elif key in data and dataclasses.is_dataclass(hint):
            data[key] = _dict_to_dataclass(hint, data[key])
    
    # Create instance
    return cls(**{k: v for k, v in data.items() if k in type_hints})


def _dict_to_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    """
    Convert a dictionary to a dataclass instance.
    
    Args:
        cls: Dataclass type to create
        data: Dictionary with field values
        
    Returns:
        T: Instance of the dataclass
    """
    field_names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in field_names})


def update_config_from_args(config: Any, args: Any, prefix: str = "") -> None:
    """
    Update configuration from command line arguments.
    
    Args:
        config: Configuration object to update
        args: Parsed arguments
        prefix: Prefix for nested configs
    """
    if not hasattr(args, "__dict__"):
        return
    
    # Get all fields in the dataclass
    if dataclasses.is_dataclass(config):
        fields = {f.name: f for f in dataclasses.fields(config)}
    else:
        fields = {key: None for key in dir(config) if not key.startswith("_") and not callable(getattr(config, key))}
    
    # Update fields from args
    for field_name in fields:
        arg_name = f"{prefix}{field_name}" if prefix else field_name
        
        # Check if argument exists and is not None
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            arg_value = getattr(args, arg_name)
            
            # Handle nested dataclasses
            if hasattr(config, field_name):
                current_value = getattr(config, field_name)
                if dataclasses.is_dataclass(current_value):
                    update_config_from_args(current_value, args, f"{field_name}_")
                else:
                    setattr(config, field_name, arg_value)


def merge_configs(base: Any, override: Any) -> None:
    """
    Merge override config into base config.
    
    Args:
        base: Base configuration to update
        override: Override configuration with new values
    """
    if not override:
        return
    
    # Get all fields in the override dataclass
    if dataclasses.is_dataclass(override):
        for field in dataclasses.fields(override):
            field_name = field.name
            override_value = getattr(override, field_name)
            
            # Skip None values
            if override_value is None:
                continue
            
            # Handle nested dataclasses
            if hasattr(base, field_name):
                base_value = getattr(base, field_name)
                if dataclasses.is_dataclass(base_value) and dataclasses.is_dataclass(override_value):
                    merge_configs(base_value, override_value)
                else:
                    setattr(base, field_name, override_value)
    else:
        # Handle non-dataclass objects
        for key in dir(override):
            if key.startswith("_") or callable(getattr(override, key)):
                continue
            
            override_value = getattr(override, key)
            
            # Skip None values
            if override_value is None:
                continue
            
            if hasattr(base, key):
                base_value = getattr(base, key)
                if hasattr(base_value, "__dict__") and hasattr(override_value, "__dict__"):
                    merge_configs(base_value, override_value)
                else:
                    setattr(base, key, override_value)


def load_config_from_file(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from a file (JSON or YAML).
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    # Determine file type from extension
    _, ext = os.path.splitext(filepath)
    
    if ext.lower() in [".json"]:
        with open(filepath, 'r') as f:
            return json.load(f)
    elif ext.lower() in [".yaml", ".yml"]:
        try:
            import yaml
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML config files")
    else:
        raise ValueError(f"Unsupported config file format: {ext}")


def load_configs(config_dir: str) -> Dict[str, Any]:
    """
    Load all configuration files from a directory.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        dict: Dictionary with all configs
    """
    # Get default configurations
    configs = get_default_configs()
    
    # Load model configuration
    model_config_path = os.path.join(config_dir, "model_config.json")
    if os.path.exists(model_config_path):
        model_config_dict = load_config_from_file(model_config_path)
        configs["model"] = load_dataclass_from_json(ModelConfig, model_config_path)
    
    # Load training configuration
    training_config_path = os.path.join(config_dir, "training_config.json")
    if os.path.exists(training_config_path):
        configs["training"] = load_dataclass_from_json(TrainingConfig, training_config_path)
    
    # Load data configuration
    data_config_path = os.path.join(config_dir, "data_config.json")
    if os.path.exists(data_config_path):
        configs["data"] = load_dataclass_from_json(DataConfig, data_config_path)
    
    # Load generation configuration
    generation_config_path = os.path.join(config_dir, "generation_config.json")
    if os.path.exists(generation_config_path):
        configs["generation"] = load_dataclass_from_json(GenerationConfig, generation_config_path)
    
    return configs


def save_configs(configs: Dict[str, Any], output_dir: str) -> None:
    """
    Save all configurations to the output directory.
    
    Args:
        configs: Dictionary with configurations
        output_dir: Directory to save configurations to
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model configuration
    if "model" in configs:
        save_dataclass_to_json(configs["model"], os.path.join(output_dir, "model_config.json"))
    
    # Save training configuration
    if "training" in configs:
        save_dataclass_to_json(configs["training"], os.path.join(output_dir, "training_config.json"))
    
    # Save data configuration
    if "data" in configs:
        save_dataclass_to_json(configs["data"], os.path.join(output_dir, "data_config.json"))
    
    # Save generation configuration
    if "generation" in configs:
        save_dataclass_to_json(configs["generation"], os.path.join(output_dir, "generation_config.json"))