"""
Data configuration for the Quantum Resonance Language Model.

This module defines the data configuration parameters for datasets, 
preprocessing, and data loading for model training and evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DataConfig:
    """Data configuration parameters."""
    
    # Dataset parameters
    dataset_name: str = "wikitext"  # Options: wikitext, daily_dialog, custom
    dataset_variant: str = "wikitext-103-v1"  # For wikitext: wikitext-103-v1, wikitext-2-v1
    tokenizer_name: str = "gpt2"
    
    # File paths
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    data_path: Optional[str] = None
    
    # Sequence parameters
    max_length: int = 512
    stride: int = 256
    
    # Preprocessing parameters
    preprocessing_num_workers: int = 4
    cache_dir: str = ".cache"
    return_tensors: str = "pt"  # pt (PyTorch), tf (TensorFlow), np (NumPy)
    
    # Subset for debugging
    subset_size: Optional[int] = None
    
    # Dialogue-specific parameters
    system_prompt: Optional[str] = None
    
    # Function calling parameters
    function_defs_path: Optional[str] = None
    
    # Extra parameters for flexibility
    extra_data_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        """Create from dictionary representation."""
        filtered_data = {
            k: v for k, v in data.items()
            if hasattr(cls, k) and not k.startswith("_")
        }
        return cls(**filtered_data)