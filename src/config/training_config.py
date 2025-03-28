"""
Training configuration for QLLM.

This module provides a simplified training configuration class that extends
the ConfigurationBase class from the core module, reducing code duplication
and ensuring consistent behavior.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union

from src.core.configuration import ConfigurationBase


@dataclass
class TrainingConfig(ConfigurationBase):
    """
    Training configuration for QLLM.
    
    This class defines parameters for model training and optimization,
    extending the ConfigurationBase with training-specific parameters.
    
    Note: This is a simplified version that relies on the shared ConfigurationBase
    class to reduce code duplication.
    """
    
    # Basic training parameters
    batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    max_epochs: int = 3
    
    # Optimizer and scheduler parameters
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    optimizer_type: str = "adamw"
    scheduler_type: str = "linear"
    
    # Device and precision settings
    device: str = "cuda"
    fp16: bool = False
    bf16: bool = False
    use_mixed_precision: bool = False
    
    # Checkpointing parameters
    save_steps: int = 100
    save_total_limit: int = 5
    save_strategy: str = "steps"  # "steps", "epochs", or "no"
    save_only_best: bool = False
    
    # Evaluation parameters
    eval_steps: int = 50
    eval_strategy: str = "steps"  # "steps", "epochs", or "no"
    eval_accumulation_steps: Optional[int] = None
    eval_delay: int = 0
    
    # Output and logging
    output_dir: str = "outputs"
    logging_dir: Optional[str] = None
    logging_steps: int = 10
    logging_first_step: bool = False
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    # Distributed training
    local_rank: int = -1
    world_size: int = 1
    ddp_find_unused_parameters: bool = False
    
    # Resume training
    auto_resume: bool = False
    resume_from_checkpoint: Optional[str] = None
    
    # Training type and model type
    training_strategy: str = "standard"  # "standard", "pretrain", "finetune"
    model_type: str = "standard"  # "standard", "dialogue", "multimodal"
    
    # Extensions
    enabled_extensions: List[str] = field(default_factory=list)
    extension_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and initialize derived parameters after instance creation."""
        # Initialize directories if not provided
        if self.logging_dir is None and self.output_dir:
            self.logging_dir = f"{self.output_dir}/logs"
        
        # Convert warmup_ratio to warmup_steps if needed
        if self.warmup_ratio > 0 and self.warmup_steps == 0:
            # This is a placeholder - in practice this would require total steps
            # which depends on dataset size. We'll handle this during training setup.
            pass
        
        # Set proper mixed precision flags
        if self.use_mixed_precision:
            if not self.fp16 and not self.bf16:
                # Default to fp16 if not specified
                self.fp16 = True
    
    def validate(self) -> List[str]:
        """
        Validate training configuration values.
        
        Returns:
            List of validation error messages, empty if valid
        """
        errors = super().validate()
        
        # Validate specific training parameters
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.weight_decay < 0:
            errors.append(f"weight_decay must be non-negative, got {self.weight_decay}")
        
        if self.max_epochs <= 0:
            errors.append(f"max_epochs must be positive, got {self.max_epochs}")
        
        if self.warmup_steps < 0:
            errors.append(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        
        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            errors.append(f"warmup_ratio must be between 0 and 1, got {self.warmup_ratio}")
        
        if self.accumulation_steps <= 0:
            errors.append(f"accumulation_steps must be positive, got {self.accumulation_steps}")
        
        if self.max_grad_norm <= 0:
            errors.append(f"max_grad_norm must be positive, got {self.max_grad_norm}")
        
        # Validate optimizer type
        valid_optimizers = ["adamw", "adam", "sgd", "adafactor", "adagrad"]
        if self.optimizer_type not in valid_optimizers:
            errors.append(
                f"optimizer_type must be one of {valid_optimizers}, "
                f"got {self.optimizer_type}"
            )
        
        # Validate scheduler type
        valid_schedulers = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "none"]
        if self.scheduler_type not in valid_schedulers:
            errors.append(
                f"scheduler_type must be one of {valid_schedulers}, "
                f"got {self.scheduler_type}"
            )
        
        # Validate save strategy
        valid_save_strategies = ["steps", "epochs", "no"]
        if self.save_strategy not in valid_save_strategies:
            errors.append(
                f"save_strategy must be one of {valid_save_strategies}, "
                f"got {self.save_strategy}"
            )
        
        # Validate evaluation strategy
        valid_eval_strategies = ["steps", "epochs", "no"]
        if self.eval_strategy not in valid_eval_strategies:
            errors.append(
                f"eval_strategy must be one of {valid_eval_strategies}, "
                f"got {self.eval_strategy}"
            )
        
        # Validate device
        valid_devices = ["cpu", "cuda", "mps", "auto"]
        if not any(self.device.startswith(d) for d in valid_devices):
            errors.append(
                f"device must start with one of {valid_devices}, "
                f"got {self.device}"
            )
        
        # Validate training strategy
        valid_strategies = ["standard", "pretrain", "finetune"]
        if self.training_strategy not in valid_strategies:
            errors.append(
                f"training_strategy must be one of {valid_strategies}, "
                f"got {self.training_strategy}"
            )
        
        # Validate model type
        valid_model_types = ["standard", "dialogue", "multimodal"]
        if self.model_type not in valid_model_types:
            errors.append(
                f"model_type must be one of {valid_model_types}, "
                f"got {self.model_type}"
            )
        
        return errors
    
    def should_save_checkpoint(self, epoch: int, step: int) -> bool:
        """
        Determine if a checkpoint should be saved at the current epoch/step.
        
        Args:
            epoch: Current epoch
            step: Current global step
            
        Returns:
            True if a checkpoint should be saved
        """
        if self.save_strategy == "no":
            return False
        elif self.save_strategy == "epochs" and epoch % 1 == 0:
            return True
        elif self.save_strategy == "steps" and step % self.save_steps == 0:
            return True
        return False
    
    def should_evaluate(self, epoch: int, step: int) -> bool:
        """
        Determine if evaluation should be performed at the current epoch/step.
        
        Args:
            epoch: Current epoch
            step: Current global step
            
        Returns:
            True if evaluation should be performed
        """
        if self.eval_strategy == "no":
            return False
        elif self.eval_strategy == "epochs" and epoch % 1 == 0:
            return True
        elif self.eval_strategy == "steps" and step % self.eval_steps == 0:
            return True
        return False
    
    def get_extension_config(self, extension_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific extension.
        
        Args:
            extension_name: Name of the extension
            
        Returns:
            Configuration dictionary for the extension
        """
        if self.extension_config is None:
            return {}
        
        return self.extension_config.get(extension_name, {})
    
    def is_extension_enabled(self, extension_name: str) -> bool:
        """
        Check if an extension is enabled in the training configuration.
        
        Args:
            extension_name: Name of the extension
            
        Returns:
            True if the extension is enabled
        """
        return extension_name in self.enabled_extensions