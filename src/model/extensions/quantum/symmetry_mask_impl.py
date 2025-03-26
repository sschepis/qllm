"""
Symmetry Mask Extension Implementation Module.

This module defines the implementation for applying structured masks based on 
quantum group symmetries to neural networks.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_quantum_core import BaseQuantumExtension
from .quantum_config import SymmetryMaskConfig
from .mask_generators import MaskGenerators
from .quantum_patterns import QuantumPatternGenerator


class SymmetryMaskExtension(BaseQuantumExtension):
    """
    Extension for applying structured masks based on quantum group symmetries.
    
    This extension creates and applies masks to neural network weights
    following principles from group theory and quantum-inspired structured
    sparsity, significantly reducing parameter count while preserving model
    capacity for resonant patterns.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the symmetry mask extension.
        
        Args:
            name: Unique name for this extension instance
            config: Configuration dictionary for the extension
        """
        super().__init__(name, config)
        
        # Create symmetry mask specific configuration
        self.mask_config = SymmetryMaskConfig.from_dict(config)
        
        # Extract key configuration parameters
        self.mask_update_interval = self.mask_config.mask_update_interval
        self.use_gradual_pruning = self.mask_config.use_gradual_pruning
        self.final_sparsity = self.mask_config.final_sparsity
        self.initial_sparsity = self.mask_config.initial_sparsity
        self.pruning_steps = self.mask_config.pruning_steps
        self.prime_moduli = self.mask_config.prime_moduli
        self.importance_threshold = self.mask_config.importance_threshold
        self.importance_measure = self.mask_config.importance_measure
        self.use_quantum_patterns = self.mask_config.use_quantum_patterns
        self.harmonic_levels = self.mask_config.harmonic_levels
        self.use_hilbert_projections = self.mask_config.use_hilbert_projections
        self.hilbert_dim = self.mask_config.hilbert_dim
        self.use_dynamic_masks = self.mask_config.use_dynamic_masks
        self.evolution_rate = self.mask_config.evolution_rate
        self.mask_resonance_factor = self.mask_config.mask_resonance_factor
        self.apply_mask_evolution = self.mask_config.apply_mask_evolution
        self.evolution_interval = self.mask_config.evolution_interval
        self.evolution_method = self.mask_config.evolution_method
        
        # Initialize mask generators
        self.mask_generator = MaskGenerators(device=self.get_device())
        
        # Initialize quantum pattern generator if needed
        if self.use_quantum_patterns:
            self.quantum_pattern_generator = QuantumPatternGenerator(
                group_order=self.group_order,
                harmonic_levels=self.harmonic_levels,
                hilbert_dim=self.hilbert_dim,
                use_hilbert_projections=self.use_hilbert_projections,
                device=self.get_device()
            )
        
        # Tracking for mask evolution
        self.mask_evolution_history = {}  # Track mask changes over time
        self.parameter_importance = {}    # Track parameter importance for evolution
        
        # Initialize mask tracking
        self.mask_update_counter = 0
        self.masked_parameter_count = 0
        self.total_parameter_count = 0
        self.layer_sparsities = {}
        
        # Cache for evolved masks
        self.evolved_masks = {}
    
    def create_mask(self, shape: Tuple[int, ...], sparsity: Optional[float] = None) -> torch.Tensor:
        """
        Create a mask for a parameter tensor.
        
        Args:
            shape: Shape of the tensor to mask
            sparsity: Target sparsity
            
        Returns:
            Binary mask of the same shape
        """
        # Use default sparsity if not specified
        if sparsity is None:
            sparsity = self.mask_sparsity
        
        # Check if we have this mask cached
        cache_key = f"{shape}_{sparsity}_{self.mask_type}"
        if cache_key in self.cached_masks:
            return self.cached_masks[cache_key]
        
        # Create the appropriate mask based on type
        if self.mask_type == "mod":
            mask = self.mask_generator.create_mod_mask(shape, self.prime_moduli, sparsity)
            
        elif self.mask_type == "prime":
            mask = self.mask_generator.create_prime_mask(shape, sparsity)
            
        elif self.mask_type == "adaptive":
            # For adaptive masks, we need a weight tensor
            # Since we don't have it here, we'll create a dummy one
            # In practice, this would be called with the actual weight tensor
            weight_tensor = torch.randn(shape, device=self.get_device())
            mask = self.mask_generator.create_adaptive_mask(
                shape, 
                weight_tensor, 
                sparsity, 
                self.importance_measure
            )
            
        elif self.mask_type == "quantum" and self.use_quantum_patterns:
            # Use our quantum pattern generator
            pattern_type = self.mask_config.extra_config.get("quantum_pattern_type", "harmonic")
            mask = self.quantum_pattern_generator.create_pattern_mask(shape, pattern_type, sparsity)
        
        else:
            # Default to modular mask
            mask = self.mask_generator.create_mod_mask(shape, [2, 3, 5], sparsity)
        
        # Cache the mask
        self.cached_masks[cache_key] = mask
        
        # Update statistics
        self.stats["avg_sparsity"] = sparsity
        
        return mask
    
    def apply_mask_to_parameter(self, 
                              parameter: nn.Parameter, 
                              weight_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply a mask to a parameter tensor.
        
        Args:
            parameter: Parameter to mask
            weight_mask: Mask to apply
            
        Returns:
            Masked parameter tensor
        """
        if weight_mask is None:
            # Create mask if not provided
            weight_mask = self.create_mask(parameter.shape, self.mask_sparsity)
        
        # Apply mask in-place
        masked_parameter = parameter.data * weight_mask
        
        # Update parameter
        with torch.no_grad():
            parameter.copy_(masked_parameter)
        
        # Update statistics
        total_params = parameter.numel()
        masked_params = total_params - weight_mask.sum().item()
        
        self.masked_parameter_count += masked_params
        self.total_parameter_count += total_params
        
        return masked_parameter
    
    def evolve_masks(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Evolve masks dynamically based on gradient information and quantum resonance.
        
        Args:
            model: Model to evolve masks for
            
        Returns:
            Dictionary mapping parameter names to evolved masks
        """
        if not self.use_dynamic_masks:
            return self.apply_masks_to_model(model)
        
        device = self.get_device()
        evolved_masks = {}
        
        # Track parameters that need gradient information
        params_requiring_grad = {}
        
        # First pass: collect gradient and importance information
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    param = module.weight
                    param_name = f"{name}.weight"
                    
                    # Skip if no gradient available
                    if not param.requires_grad or param.grad is None:
                        continue
                        
                    # Get current mask if available
                    current_mask = self.cached_masks.get(param_name)
                    if current_mask is None:
                        # Create initial mask if not available
                        current_mask = self.create_mask(param.shape, self.mask_sparsity)
                        self.cached_masks[param_name] = current_mask
                    
                    # Calculate parameter importance based on method
                    if self.evolution_method == "gradient_sensitive":
                        # Use gradient magnitude as importance measure
                        importance = torch.abs(param.grad)
                        # Apply current mask to importance (only consider non-masked params)
                        importance = importance * current_mask
                        
                    elif self.evolution_method == "momentum":
                        # Use exponential moving average of gradient magnitudes
                        if param_name not in self.parameter_importance:
                            self.parameter_importance[param_name] = torch.abs(param.grad)
                        else:
                            # Update with momentum
                            self.parameter_importance[param_name] = (
                                0.9 * self.parameter_importance[param_name] + 
                                0.1 * torch.abs(param.grad)
                            )
                        importance = self.parameter_importance[param_name] * current_mask
                        
                    elif self.evolution_method == "resonance":
                        # Quantum-inspired resonance method
                        if param_name not in self.parameter_importance:
                            # Initialize with gradient magnitude
                            self.parameter_importance[param_name] = torch.abs(param.grad)
                        else:
                            # Apply resonance factor to parameters that consistently 
                            # have gradients in the same direction
                            grad_sign = torch.sign(param.grad)
                            prev_grad_sign = torch.sign(self.parameter_importance[param_name])
                            
                            # Resonance occurs when signs align
                            resonance = (grad_sign == prev_grad_sign).float()
                            
                            # Update importance with resonance
                            self.parameter_importance[param_name] = (
                                self.parameter_importance[param_name] + 
                                self.mask_resonance_factor * torch.abs(param.grad) * resonance
                            )
                        
                        # Apply current mask
                        importance = self.parameter_importance[param_name] * current_mask
                    
                    else:  # Default to gradient magnitude
                        importance = torch.abs(param.grad) * current_mask
                    
                    # Store for mask evolution
                    params_requiring_grad[param_name] = {
                        'param': param,
                        'importance': importance,
                        'current_mask': current_mask
                    }
        
        # Second pass: evolve masks based on importance and resonance patterns
        for param_name, param_data in params_requiring_grad.items():
            param = param_data['param']
            importance = param_data['importance']
            current_mask = param_data['current_mask']
            
            # Get target sparsity level
            if self.use_gradual_pruning and self.mask_update_counter < self.pruning_steps:
                # Calculate current sparsity from gradual pruning schedule
                target_sparsity = self.initial_sparsity + (
                    (self.final_sparsity - self.initial_sparsity) *
                    (self.mask_update_counter / self.pruning_steps)
                )
            else:
                target_sparsity = self.mask_sparsity
            
            # Create evolved mask:
            # 1. Keep a portion of the current mask (stability)
            # 2. Allow some parameters to change based on importance (adaptivity)
            
            # Calculate how many parameters can change
            num_active = current_mask.sum().item()
            num_total = current_mask.numel()
            num_to_change = int(self.evolution_rate * num_active)
            
            # Identify lowest importance active parameters
            active_importances = importance[current_mask > 0]
            if len(active_importances) > 0:
                threshold = torch.kthvalue(active_importances.reshape(-1), 
                                          min(num_to_change, len(active_importances))).values
                
                # Parameters to deactivate (low importance active params)
                to_deactivate = (importance < threshold) & (current_mask > 0)
                
                # Identify highest importance inactive parameters
                inactive_importances = importance[current_mask == 0]
                if len(inactive_importances) > 0:
                    inactive_threshold = torch.kthvalue(
                        inactive_importances.reshape(-1), 
                        max(1, len(inactive_importances) - num_to_change)
                    ).values
                    
                    # Parameters to activate (high importance inactive params)
                    to_activate = (importance > inactive_threshold) & (current_mask == 0)
                    
                    # Create new mask by updating current mask
                    new_mask = current_mask.clone()
                    new_mask[to_deactivate] = 0.0
                    new_mask[to_activate] = 1.0
                    
                    # Ensure target sparsity is maintained
                    new_active = new_mask.sum().item()
                    current_sparsity = 1.0 - (new_active / num_total)
                    
                    if abs(current_sparsity - target_sparsity) > 0.01:
                        # Adjust to match target sparsity using mask_generator
                        new_mask = self.mask_generator.adjust_mask_sparsity(new_mask, target_sparsity)
                    
                    # Apply quantum pattern influence to maintain structured sparsity
                    if self.use_quantum_patterns:
                        # Generate quantum pattern at target sparsity
                        quantum_mask = self.quantum_pattern_generator.create_pattern_mask(
                            param.shape, 
                            "harmonic", 
                            target_sparsity
                        )
                        
                        # Blend with evolved mask - allow quantum pattern to influence
                        # but not completely override the importance-based evolution
                        blend_factor = 0.2  # 20% influence from quantum pattern
                        blended_mask = (new_mask * (1 - blend_factor) + 
                                      quantum_mask * blend_factor)
                        
                        # Re-binarize
                        final_mask = (blended_mask > 0.5).float()
                    else:
                        final_mask = new_mask
                    
                    # Store evolved mask
                    evolved_masks[param_name] = final_mask
                    self.cached_masks[param_name] = final_mask
                    
                    # Track evolution history if needed
                    if param_name not in self.mask_evolution_history:
                        self.mask_evolution_history[param_name] = []
                    
                    # Only store if there's a significant change
                    mask_change = (final_mask != current_mask).float().mean().item()
                    if mask_change > 0.01:  # More than 1% change
                        self.mask_evolution_history[param_name].append({
                            'step': self.mask_update_counter,
                            'change_percent': mask_change * 100,
                            'sparsity': 1.0 - (final_mask.sum().item() / final_mask.numel())
                        })
                    
                    # Apply the evolved mask
                    self.apply_mask_to_parameter(param, final_mask)
                    
                    # Track sparsity
                    actual_sparsity = 1.0 - (final_mask.sum().item() / final_mask.numel())
                    self.layer_sparsities[param_name] = actual_sparsity
        
        # Update overall sparsity statistic
        self.stats["total_mask_evolutions"] = self.stats.get("total_mask_evolutions", 0) + 1
        
        return evolved_masks
    
    def initialize(self, model: nn.Module) -> None:
        """
        Initialize the extension with the main model.
        
        Args:
            model: The main model instance
        """
        self.model = model
        
        # Apply initial masks to model parameters
        if self.mask_config.extra_config.get("apply_masks_on_init", True):
            self.apply_masks_to_model(model)
        
        # Register hooks for mask reapplication during training
        def mask_hook(module, grad_input, grad_output):
            # Skip if not in training mode
            if not model.training:
                return
            
            # Track update counter
            self.mask_update_counter += 1
            
            # Reapply masks if needed
            if self.mask_update_counter % self.mask_update_interval == 0:
                # Only apply to this specific module
                for name, param in module.named_parameters(recurse=False):
                    if param.requires_grad and len(param.shape) > 1:  # Only apply to weight matrices
                        # Get cached mask if available
                        full_name = f"{module.__class__.__name__}.{name}"
                        mask = self.cached_masks.get(full_name)
                        
                        # If using gradual pruning, adjust sparsity
                        if self.use_gradual_pruning and self.mask_update_counter < self.pruning_steps:
                            current_sparsity = self.initial_sparsity + (
                                (self.final_sparsity - self.initial_sparsity) *
                                (self.mask_update_counter / self.pruning_steps)
                            )
                            mask = self.create_mask(param.shape, current_sparsity)
                            self.cached_masks[full_name] = mask
                        
                        # Apply mask
                        if mask is not None:
                            self.apply_mask_to_parameter(param, mask)
            
            # Apply dynamic mask evolution if enabled
            if self.use_dynamic_masks and self.mask_update_counter % self.evolution_interval == 0:
                self.evolve_masks(model)
        
        # Register hook for each applicable module
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.register_forward_hook(mask_hook)
        
        self.initialized = True
    
    def apply_masks_to_model(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Apply masks to all applicable parameters in the model.
        
        Args:
            model: Model to apply masks to
            
        Returns:
            Dictionary mapping parameter names to masks
        """
        # Reset counters
        self.masked_parameter_count = 0
        self.total_parameter_count = 0
        self.layer_sparsities = {}
        
        # Dictionary to store masks
        masks = {}
        
        # Apply masks to all parameters
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Apply to weight
                if hasattr(module, 'weight') and module.weight is not None:
                    # Create mask
                    weight_mask = self.create_mask(module.weight.shape, self.mask_sparsity)
                    
                    # Apply mask
                    self.apply_mask_to_parameter(module.weight, weight_mask)
                    
                    # Store mask
                    mask_name = f"{name}.weight"
                    masks[mask_name] = weight_mask
                    self.cached_masks[mask_name] = weight_mask
                    
                    # Track sparsity
                    sparsity = 1.0 - (weight_mask.sum().item() / weight_mask.numel())
                    self.layer_sparsities[mask_name] = sparsity
        
        # Update overall sparsity statistic
        if self.total_parameter_count > 0:
            overall_sparsity = self.masked_parameter_count / self.total_parameter_count
            self.stats["avg_sparsity"] = overall_sparsity
        
        return masks
    
    def forward(self, 
               x: torch.Tensor,
               model_outputs: Optional[Dict[str, Any]] = None, 
               extension_outputs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass of the symmetry mask extension.
        
        Args:
            x: Input tensor
            model_outputs: Outputs from the main model
            extension_outputs: Outputs from other extensions
            
        Returns:
            Tuple of (modified tensor, extension metadata)
        """
        # This extension primarily operates on model parameters,
        # not on the forward pass data. The forward method is mainly
        # for consistency with the extension API.
        
        # Prepare metadata
        metadata = {
            "extension_type": "symmetry_mask",
            "mask_type": self.mask_type,
            "mask_sparsity": self.mask_sparsity,
            "masked_parameters": self.masked_parameter_count,
            "total_parameters": self.total_parameter_count,
            "layer_sparsities": self.layer_sparsities,
            "mask_update_counter": self.mask_update_counter,
            "stats": self.stats
        }
        
        # Add dynamic mask evolution info if enabled
        if self.use_dynamic_masks:
            metadata["dynamic_masks_enabled"] = True
            metadata["evolution_method"] = self.evolution_method
            metadata["evolution_rate"] = self.evolution_rate
            metadata["evolution_interval"] = self.evolution_interval
            
            # Include recent evolution history
            if len(self.mask_evolution_history) > 0:
                # Add summary of recent changes
                recent_changes = {}
                for param_name, history in self.mask_evolution_history.items():
                    if len(history) > 0:
                        recent_changes[param_name] = history[-1]  # Most recent change
                
                metadata["recent_mask_changes"] = recent_changes
        
        # If in training mode and we have a model, ensure masks are applied
        if hasattr(self, 'model') and self.model is not None and self.model.training:
            if model_outputs and model_outputs.get("is_training", False):
                # Check if we should evolve masks
                if self.use_dynamic_masks and self.apply_mask_evolution:
                    # Only evolve masks periodically
                    if self.mask_update_counter % self.evolution_interval == 0:
                        evolved_masks = self.evolve_masks(self.model)
                        metadata["evolved_masks"] = {k: v.sum().item() / v.numel() for k, v in evolved_masks.items()}
        
        return x, metadata