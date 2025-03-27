"""
Training configuration for the Quantum Resonance Language Model.

This module defines the training configuration parameters for model
training, including learning rates, optimizers, etc.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    
    # Basic training parameters
    batch_size: int = 16
    eval_batch_size: int = 16  # Added this to fix the error
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_epochs: int = 3
    
    # Training modes and types
    training_type: str = "standard"  # standard, dialogue, verbose
    learning_mode: str = "adaptive"  # adaptive, scheduled, feedback_driven
    
    # Device and precision
    device: Optional[str] = None  # cuda, cpu, mps, auto
    use_mixed_precision: bool = True
    
    # Optimizer settings
    optimizer: str = "adamw"
    max_grad_norm: float = 1.0
    accumulation_steps: int = 1
    
    # Scheduler settings
    lr_scheduler: str = "cosine"
    warmup_steps: int = 0
    
    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 0
    eval_steps: int = 0
    save_every_epoch: bool = True
    disable_optimizer_saving: bool = False
    
    # Output directory
    output_dir: str = "runs/quantum_resonance"
    checkpoint_dir: Optional[str] = None  # For backward compatibility with quick_start.py
    
    # Reproducibility
    seed: int = 42
    
    # Extra parameters for flexibility
    extra_training_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary representation."""
        filtered_data = {
            k: v for k, v in data.items()
            if hasattr(cls, k) and not k.startswith("_")
        }
        return cls(**filtered_data)