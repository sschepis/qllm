"""
Gradient scaling utilities for mixed precision training.

This module provides utilities for gradient scaling and Automatic Mixed Precision (AMP)
training to optimize training performance while maintaining numerical stability.
"""

import logging
from typing import Dict, Optional, Union, Any, Callable, List, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from src.config.training_config import TrainingConfig


# Get logger
logger = logging.getLogger("quantum_resonance")


class GradScalerManager:
    """
    Manager for gradient scaling during mixed precision training.
    
    This class handles the creation and management of gradient scalers for
    automatic mixed precision training, with device-specific optimizations
    and fallback mechanisms.
    """
    
    def __init__(
        self,
        config: Union[dict, TrainingConfig],
        device: Optional[torch.device] = None,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: Optional[bool] = None
    ):
        """
        Initialize the gradient scaler manager.
        
        Args:
            config: Training configuration
            device: Device to use
            init_scale: Initial scale factor
            growth_factor: Factor by which the scale is multiplied when no infs/NaNs occur
            backoff_factor: Factor by which the scale is multiplied when infs/NaNs occur
            growth_interval: Number of consecutive unskipped steps before growing the scale
            enabled: Whether to enable gradient scaling (default: auto-detect)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract AMP settings from config
        if isinstance(config, dict):
            self.use_amp = enabled if enabled is not None else config.get("use_amp", True)
            self.fp16_opt_level = config.get("fp16_opt_level", "O1")
            self.loss_scale = config.get("loss_scale", init_scale)
        else:
            self.use_amp = enabled if enabled is not None else getattr(config, "use_amp", True)
            self.fp16_opt_level = getattr(config, "fp16_opt_level", "O1")
            self.loss_scale = getattr(config, "loss_scale", init_scale)
        
        # Initialize gradient scaler for PyTorch AMP
        self.grad_scaler = None
        
        # Initialize apex if requested and available
        self.using_apex = False
        
        # Only enable AMP on CUDA devices
        self.use_amp = self.use_amp and torch.cuda.is_available()
        
        # Initialize gradient scaler if AMP is enabled
        if self.use_amp:
            self._initialize_amp(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval
            )
    
    def _initialize_amp(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000
    ) -> None:
        """
        Initialize Automatic Mixed Precision utilities.
        
        Args:
            init_scale: Initial scale factor
            growth_factor: Factor by which the scale is multiplied when no infs/NaNs occur
            backoff_factor: Factor by which the scale is multiplied when infs/NaNs occur
            growth_interval: Number of consecutive unskipped steps before growing the scale
        """
        # Try to use apex if O1/O2 opt level is requested
        if self.fp16_opt_level in ["O1", "O2"] and "apex" in self.fp16_opt_level.lower():
            try:
                from apex import amp
                self.using_apex = True
                logger.info(f"Using Apex AMP with opt_level {self.fp16_opt_level}")
            except ImportError:
                logger.warning("Apex requested but not installed, falling back to PyTorch AMP")
                self.using_apex = False
        
        # Otherwise use PyTorch AMP
        if not self.using_apex:
            self.grad_scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
                enabled=self.use_amp
            )
            logger.info("Using PyTorch native AMP for mixed precision training")
    
    def scale_loss(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> torch.Tensor:
        """
        Scale the loss for mixed precision training.
        
        Args:
            loss: Loss tensor
            optimizer: Optimizer
            
        Returns:
            Scaled loss
        """
        if not self.use_amp:
            return loss
        
        if self.using_apex:
            # When using apex, the scaling is done in the backward pass
            return loss
        
        return self.grad_scaler.scale(loss)
    
    def step(
        self,
        optimizer: torch.optim.Optimizer,
        closure: Optional[Callable] = None
    ) -> Optional[torch.Tensor]:
        """
        Perform optimizer step with gradient scaling.
        
        Args:
            optimizer: Optimizer
            closure: Closure for computing gradients (if needed)
            
        Returns:
            Optional loss tensor (from closure)
        """
        if not self.use_amp:
            return optimizer.step(closure)
        
        if self.using_apex:
            return optimizer.step(closure)
        
        return self.grad_scaler.step(optimizer, closure)
    
    def update(self) -> None:
        """
        Update the gradient scaler.
        
        Should be called after optimizer.step().
        """
        if self.use_amp and not self.using_apex and self.grad_scaler is not None:
            self.grad_scaler.update()
    
    def unscale_(
        self,
        optimizer: torch.optim.Optimizer
    ) -> None:
        """
        Unscale gradients for gradient clipping.
        
        Call before gradient clipping and after backward, but before optimizer.step().
        
        Args:
            optimizer: Optimizer
        """
        if self.use_amp and not self.using_apex and self.grad_scaler is not None:
            self.grad_scaler.unscale_(optimizer)
    
    def get_scale(self) -> float:
        """
        Get current scale value.
        
        Returns:
            Current scale value
        """
        if self.use_amp and not self.using_apex and self.grad_scaler is not None:
            return self.grad_scaler.get_scale()
        return 1.0
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dictionary for checkpointing.
        
        Returns:
            State dictionary
        """
        state = {"use_amp": self.use_amp}
        
        if self.use_amp and not self.using_apex and self.grad_scaler is not None:
            state["grad_scaler"] = self.grad_scaler.state_dict()
        
        return state
    
    def load_state_dict(
        self,
        state_dict: Dict[str, Any]
    ) -> None:
        """
        Load state dictionary from checkpoint.
        
        Args:
            state_dict: State dictionary
        """
        if "use_amp" in state_dict:
            self.use_amp = state_dict["use_amp"]
        
        if self.use_amp and "grad_scaler" in state_dict and self.grad_scaler is not None:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
    
    @property
    def is_enabled(self) -> bool:
        """
        Check if mixed precision is enabled.
        
        Returns:
            True if mixed precision is enabled
        """
        return self.use_amp
    
    def __call__(
        self,
        enabled: Optional[bool] = None
    ) -> 'autocast':
        """
        Create an autocast context manager.
        
        Args:
            enabled: Override whether autocast is enabled
            
        Returns:
            autocast context manager
        """
        _enabled = enabled if enabled is not None else self.use_amp
        return autocast(enabled=_enabled)


def apply_gradients(
    optimizer: torch.optim.Optimizer,
    grad_scaler: Optional[GradScalerManager] = None,
    max_grad_norm: Optional[float] = None,
    closure: Optional[Callable] = None
) -> Optional[torch.Tensor]:
    """
    Apply gradients to parameters with optional gradient scaling and clipping.
    
    This function handles the complete gradient application process with proper
    handling of gradient scaling, clipping, and optimizer stepping.
    
    Args:
        optimizer: Optimizer
        grad_scaler: Gradient scaler manager
        max_grad_norm: Maximum gradient norm for clipping
        closure: Closure for computing gradients (if needed)
        
    Returns:
        Optional loss tensor (from closure)
    """
    # Unscale gradients if using a scaler
    if grad_scaler is not None and grad_scaler.is_enabled:
        grad_scaler.unscale_(optimizer)
    
    # Clip gradients if requested
    if max_grad_norm is not None and max_grad_norm > 0:
        # Get all parameters that require grad
        parameters = [p for param_group in optimizer.param_groups 
                     for p in param_group['params'] if p.requires_grad]
        
        if parameters:
            # Only clip if we have parameters that require grad
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    
    # Apply gradients using the optimizer
    if grad_scaler is not None and grad_scaler.is_enabled:
        loss = grad_scaler.step(optimizer, closure)
        grad_scaler.update()
        return loss
    else:
        return optimizer.step(closure)


def optimize_memory_for_training(
    model: nn.Module,
    use_amp: bool = True,
    use_channels_last: bool = True,
    use_tf32: bool = True
) -> None:
    """
    Apply memory optimizations for model training.
    
    Args:
        model: Model to optimize
        use_amp: Whether to optimize for AMP (affects settings)
        use_channels_last: Whether to use channels last memory format
        use_tf32: Whether to use TF32 precision (A100+ GPUs)
    """
    # Set TF32 precision (improves performance on A100)
    if torch.cuda.is_available() and use_tf32:
        # Only affects Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled TF32 precision for CUDA operations")
    
    # Use channels last memory format for better performance with AMP
    if torch.cuda.is_available() and use_channels_last:
        # Only use for convolutional models
        if any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) 
              for m in model.modules()):
            try:
                model = model.to(memory_format=torch.channels_last)
                logger.info("Using channels last memory format")
            except Exception as e:
                logger.warning(f"Failed to convert model to channels last format: {e}")
    
    # Additional memory optimizations
    if torch.cuda.is_available():
        # Set higher CUDA allocation size (reduces fragmentation)
        torch.cuda.empty_cache()
        
        # Enable CUDA caching allocator
        if hasattr(torch.cuda, "memory_stats"):
            torch.cuda.memory_stats()