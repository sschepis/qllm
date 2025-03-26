"""
Extension for applying structured masks based on quantum group symmetries.

This module provides an extension that applies structured sparsity
patterns inspired by quantum group theory to neural networks.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_quantum_extension import BaseQuantumExtension


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
            name (str): Unique name for this extension instance
            config (Dict[str, Any]): Configuration dictionary for the extension
        """
        super().__init__(name, config)
        
        # Additional mask configuration
        self.mask_update_interval = config.get("mask_update_interval", 1000)
        self.use_gradual_pruning = config.get("use_gradual_pruning", False)
        self.final_sparsity = config.get("final_sparsity", 0.9)
        self.initial_sparsity = config.get("initial_sparsity", 0.5)
        self.pruning_steps = config.get("pruning_steps", 10000)
        
        # Prime pattern configuration
        self.prime_moduli = config.get("prime_moduli", [2, 3, 5, 7, 11])
        
        # Adaptive mask settings
        self.importance_threshold = config.get("importance_threshold", 0.01)
        self.importance_measure = config.get("importance_measure", "magnitude")  # or "gradient", "sensitivity"
        
        # Advanced quantum-inspired configurations
        self.group_order = config.get("group_order", 8)
        self.use_quantum_patterns = config.get("use_quantum_patterns", False)
        self.harmonic_levels = config.get("harmonic_levels", 5)
        self.use_hilbert_projections = config.get("use_hilbert_projections", False)
        self.hilbert_dim = config.get("hilbert_dim", 16)
        
        # Dynamic mask evolution settings
        self.use_dynamic_masks = config.get("use_dynamic_masks", False)
        self.evolution_rate = config.get("evolution_rate", 0.1)
        self.mask_resonance_factor = config.get("mask_resonance_factor", 0.2)
        self.apply_mask_evolution = config.get("apply_mask_evolution", True)
        self.evolution_interval = config.get("evolution_interval", 100)
        self.evolution_method = config.get("evolution_method", "gradient_sensitive")
        
        # Tracking for mask evolution
        self.mask_evolution_history = {}  # Track mask changes over time
        self.parameter_importance = {}    # Track parameter importance for evolution
        
        # Initialize advanced masking components
        if self.use_quantum_patterns:
            self._initialize_quantum_patterns()
        
        # Initialize mask tracking
        self.mask_update_counter = 0
        self.masked_parameter_count = 0
        self.total_parameter_count = 0
        self.layer_sparsities = {}
        
        # Cache for evolved masks
        self.evolved_masks = {}
    
    def _initialize_quantum_patterns(self):
        """Initialize quantum-inspired masking components."""
        device = self.get_device()
        
        # Initialize harmonic patterns
        self.harmonic_patterns = []
        for i in range(1, self.harmonic_levels + 1):
            # Create harmonic wave with increasing frequency
            x = torch.linspace(0, 1, 100, device=device)
            pattern = torch.sin(2 * math.pi * i * x) 
            self.harmonic_patterns.append(pattern)
        
        # Initialize Hilbert space projections
        if self.use_hilbert_projections:
            self.hilbert_projection = nn.Linear(
                self.group_order, self.hilbert_dim, device=device
            )
            self.hilbert_norm = nn.LayerNorm(self.hilbert_dim, device=device)
        
        # Create mask generators for different patterns
        self.mask_generators = nn.ModuleDict({
            "prime": nn.Sequential(
                nn.Linear(self.group_order, 64, device=device),
                nn.ReLU(),
                nn.Linear(64, self.group_order, device=device),
                nn.Sigmoid()
            ),
            "cyclic": nn.Sequential(
                nn.Linear(self.group_order, 32, device=device),
                nn.Tanh(),
                nn.Linear(32, self.group_order, device=device),
                nn.Sigmoid()
            ),
            "orthogonal": nn.Sequential(
                nn.Linear(self.group_order, 16, device=device),
                nn.GELU(),
                nn.Linear(16, self.group_order, device=device),
                nn.Sigmoid()
            )
        })
    
    def get_device(self) -> torch.device:
        """
        Get the device to use for extension operations.
        
        Returns:
            torch.device: Device to use
        """
        return next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device("cpu")
    
    def create_mod_mask(self, shape: Tuple[int, ...], moduli: List[int], sparsity: float) -> torch.Tensor:
        """
        Create a mask using the modular arithmetic pattern.
        
        Args:
            shape (Tuple[int, ...]): Shape of the tensor to mask
            moduli (List[int]): List of moduli to use
            sparsity (float): Target sparsity
            
        Returns:
            torch.Tensor: Binary mask of the same shape
        """
        device = self.get_device()
        
        # Create mask (all zeros initially)
        mask = torch.zeros(shape, device=device)
        
        # Create flat index tensor
        if len(shape) == 1:
            indices = torch.arange(shape[0], device=device)
            
            # Apply modular patterns
            for modulus in moduli:
                mask[(indices % modulus) == 0] = 1.0
                
        elif len(shape) == 2:
            rows, cols = shape
            row_indices = torch.arange(rows, device=device).unsqueeze(1)
            col_indices = torch.arange(cols, device=device).unsqueeze(0)
            
            # Apply modular patterns for each modulus
            for modulus in moduli:
                mask[(row_indices % modulus) == 0] = 1.0
                mask[(col_indices % modulus) == 0] = 1.0
                
        else:
            # For tensors with more than 2 dimensions, apply to the last 2 dims
            flat_shape = [torch.prod(torch.tensor(shape[:-2])).item(), shape[-2], shape[-1]]
            reshaped_mask = self.create_mod_mask(
                (flat_shape[-2], flat_shape[-1]), 
                moduli, 
                sparsity
            )
            mask = reshaped_mask.unsqueeze(0).expand(flat_shape).reshape(shape)
        
        # Adjust to target sparsity if needed
        current_sparsity = 1.0 - (mask.sum().item() / mask.numel())
        
        if abs(current_sparsity - sparsity) > 0.1:
            # If too dense, randomly zero out some elements
            if current_sparsity < sparsity:
                # Need more zeros (more sparsity)
                num_excess = int((sparsity - current_sparsity) * mask.numel())
                indices = torch.where(mask > 0.5)
                if len(indices[0]) > 0:
                    # Randomly select elements to zero out
                    to_zero = torch.randperm(len(indices[0]), device=device)[:num_excess]
                    for i in range(num_excess):
                        if i < len(to_zero):
                            idx = tuple(dim[to_zero[i]] for dim in indices)
                            mask[idx] = 0.0
            else:
                # Too sparse, randomly add some elements
                num_needed = int((current_sparsity - sparsity) * mask.numel())
                indices = torch.where(mask < 0.5)
                if len(indices[0]) > 0:
                    # Randomly select elements to set to 1
                    to_one = torch.randperm(len(indices[0]), device=device)[:num_needed]
                    for i in range(num_needed):
                        if i < len(to_one):
                            idx = tuple(dim[to_one[i]] for dim in indices)
                            mask[idx] = 1.0
        
        return mask
    
    def create_prime_mask(self, shape: Tuple[int, ...], sparsity: float) -> torch.Tensor:
        """
        Create a mask using prime number patterns.
        
        Args:
            shape (Tuple[int, ...]): Shape of the tensor to mask
            sparsity (float): Target sparsity
            
        Returns:
            torch.Tensor: Binary mask of the same shape
        """
        device = self.get_device()
        
        # Create mask (all zeros initially)
        mask = torch.zeros(shape, device=device)
        
        # Use small prime numbers for pattern
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        if len(shape) == 1:
            n = shape[0]
            for i in range(n):
                # Keep if index+1 is prime or multiple of prime with high enough weight
                for p in primes:
                    if (i + 1) % p == 0:
                        mask[i] = 1.0
                        break
                        
        elif len(shape) == 2:
            rows, cols = shape
            for i in range(rows):
                for j in range(cols):
                    # Use prime-based pattern
                    if (i + 1) % 2 == 0 and (j + 1) % 3 == 0:
                        mask[i, j] = 1.0
                    elif (i + 1) % 3 == 0 and (j + 1) % 2 == 0:
                        mask[i, j] = 1.0
                    elif (i + 1) % 5 == 0 or (j + 1) % 5 == 0:
                        mask[i, j] = 1.0
                    elif (i + 1) % 7 == 0 or (j + 1) % 7 == 0:
                        mask[i, j] = 1.0
        else:
            # For higher dimensions, apply to last 2 dims
            flat_shape = [torch.prod(torch.tensor(shape[:-2])).item(), shape[-2], shape[-1]]
            reshaped_mask = self.create_prime_mask(
                (flat_shape[-2], flat_shape[-1]), 
                sparsity
            )
            mask = reshaped_mask.unsqueeze(0).expand(flat_shape).reshape(shape)
        
        # Adjust to target sparsity
        current_sparsity = 1.0 - (mask.sum().item() / mask.numel())
        
        if abs(current_sparsity - sparsity) > 0.1:
            # Similar adjustment as in create_mod_mask
            if current_sparsity < sparsity:
                # Need more zeros (more sparsity)
                indices = torch.where(mask > 0.5)
                num_excess = int((sparsity - current_sparsity) * mask.numel())
                if len(indices[0]) > 0:
                    to_zero = torch.randperm(len(indices[0]), device=device)[:num_excess]
                    for i in range(len(to_zero)):
                        idx = tuple(dim[to_zero[i]] for dim in indices)
                        mask[idx] = 0.0
            else:
                # Too sparse, randomly add some elements
                indices = torch.where(mask < 0.5)
                num_needed = int((current_sparsity - sparsity) * mask.numel())
                if len(indices[0]) > 0:
                    to_one = torch.randperm(len(indices[0]), device=device)[:num_needed]
                    for i in range(len(to_one)):
                        idx = tuple(dim[to_one[i]] for dim in indices)
                        mask[idx] = 1.0
        
        return mask
    
    def create_adaptive_mask(self, 
                           shape: Tuple[int, ...], 
                           weight_tensor: torch.Tensor, 
                           sparsity: float) -> torch.Tensor:
        """
        Create an adaptive mask based on weight magnitudes.
        
        Args:
            shape (Tuple[int, ...]): Shape of the tensor to mask
            weight_tensor (torch.Tensor): Weight tensor to adapt mask to
            sparsity (float): Target sparsity
            
        Returns:
            torch.Tensor: Binary mask of the same shape
        """
        device = self.get_device()
        
        # Use the importance measure to determine mask
        if self.importance_measure == "magnitude":
            importance = torch.abs(weight_tensor)
        elif self.importance_measure == "gradient" and weight_tensor.grad is not None:
            importance = torch.abs(weight_tensor.grad)
        elif self.importance_measure == "sensitivity" and weight_tensor.grad is not None:
            importance = torch.abs(weight_tensor * weight_tensor.grad)
        else:
            # Default to magnitude
            importance = torch.abs(weight_tensor)
        
        # Calculate threshold for desired sparsity
        threshold = torch.quantile(
            importance.view(-1), 
            sparsity
        )
        
        # Create mask by thresholding
        mask = (importance > threshold).float()
        
        return mask
    
    def create_quantum_mask(self, 
                          shape: Tuple[int, ...], 
                          sparsity: float,
                          pattern_type: str = "harmonic") -> torch.Tensor:
        """
        Create a mask using quantum-inspired patterns.
        
        Args:
            shape (Tuple[int, ...]): Shape of the tensor to mask
            sparsity (float): Target sparsity
            pattern_type (str): Type of quantum pattern to use
                Options: "harmonic", "hilbert", "cyclic", "prime", "orthogonal"
            
        Returns:
            torch.Tensor: Binary mask of the same shape
        """
        device = self.get_device()
        
        # Initialize mask with zeros
        mask = torch.zeros(shape, device=device)
        
        if pattern_type == "harmonic" and self.use_quantum_patterns:
            # Apply harmonic patterns
            if len(shape) == 2:
                rows, cols = shape
                
                # Process the matrix using harmonic patterns
                for i in range(rows):
                    for j in range(cols):
                        # Scale indices to [0, 1] range
                        x = i / rows
                        y = j / cols
                        
                        # Combine multiple harmonic patterns
                        val = 0.0
                        for n, pattern in enumerate(self.harmonic_patterns):
                            # Get amplitude at normalized position
                            pos_x = int(x * 99)  # Scale to pattern length
                            pos_y = int(y * 99)
                            
                            # Using both x and y positions with different patterns
                            weight = 1.0 / (n + 1)  # Weight decreases with harmonic number
                            val += weight * (pattern[pos_x] + pattern[pos_y]) / 2
                        
                        # Apply quantum interference effects
                        quantum_val = torch.sin(torch.tensor(val * math.pi))
                        
                        # Threshold based on sparsity
                        if quantum_val > (1.0 - sparsity * 2):
                            mask[i, j] = 1.0
            
            elif len(shape) > 2:
                # For higher dimensions, apply to the last 2 dimensions
                flat_shape = [torch.prod(torch.tensor(shape[:-2])).item(), shape[-2], shape[-1]]
                reshaped_mask = self.create_quantum_mask(
                    (flat_shape[-2], flat_shape[-1]),
                    sparsity,
                    pattern_type
                )
                
                # Expand to all higher dimensions
                mask = reshaped_mask.unsqueeze(0).expand(flat_shape).reshape(shape)
                
            else:
                # For 1D tensors, use direct harmonic patterns
                n_elements = shape[0]
                scale = 100 / n_elements
                
                # Combine multiple harmonics
                for i in range(n_elements):
                    val = 0.0
                    for n, pattern in enumerate(self.harmonic_patterns):
                        idx = min(int(i * scale), 99)
                        weight = 1.0 / (n + 1)
                        val += weight * pattern[idx]
                    
                    # Apply quantum interference effect
                    qval = torch.sin(torch.tensor(val * math.pi))
                    mask[i] = (qval > 0).float()
        
        elif pattern_type == "hilbert" and self.use_hilbert_projections:
            # Use Hilbert space projections for more sophisticated patterns
            if len(shape) == 2:
                rows, cols = shape
                
                # Create basis vectors in Hilbert space
                basis = torch.randn(self.group_order, self.hilbert_dim, device=device)
                basis = F.normalize(basis, dim=1)
                
                # Project each position into Hilbert space
                for i in range(rows):
                    for j in range(cols):
                        # Create position encoding
                        pos_enc = torch.zeros(self.group_order, device=device)
                        
                        # Encode position using different frequencies
                        for k in range(self.group_order):
                            pos_enc[k] = torch.sin(torch.tensor(
                                (i / rows) * (k + 1) * math.pi +
                                (j / cols) * (k + 1) * math.pi/2
                            ))
                        
                        # Project to Hilbert space
                        proj = self.hilbert_projection(pos_enc)
                        proj = self.hilbert_norm(proj)
                        
                        # Calculate inner products with basis vectors
                        similarities = F.cosine_similarity(
                            proj.unsqueeze(0),
                            basis,
                            dim=1
                        )
                        
                        # Mask based on similarity threshold
                        max_sim = similarities.max().item()
                        if max_sim > (1.0 - sparsity):
                            mask[i, j] = 1.0
            
            else:
                # For non-2D tensors, use similar logic as in harmonic case
                if len(shape) > 2:
                    flat_shape = [torch.prod(torch.tensor(shape[:-2])).item(), shape[-2], shape[-1]]
                    reshaped_mask = self.create_quantum_mask(
                        (flat_shape[-2], flat_shape[-1]),
                        sparsity,
                        pattern_type
                    )
                    mask = reshaped_mask.unsqueeze(0).expand(flat_shape).reshape(shape)
                else:
                    # Fallback to prime mask for 1D
                    mask = self.create_prime_mask(shape, sparsity)
                    
        elif pattern_type in ["cyclic", "prime", "orthogonal"]:
            # Use the mask generators from our ModuleDict
            if len(shape) == 2:
                rows, cols = shape
                
                # Generate basis pattern using the appropriate generator
                generator = self.mask_generators[pattern_type]
                input_tensor = torch.randn(self.group_order, device=device)
                basis_pattern = generator(input_tensor)
                
                # Apply the pattern differently based on the type
                if pattern_type == "cyclic":
                    # Create cyclic pattern
                    for i in range(rows):
                        for j in range(cols):
                            # Cyclic index calculation
                            idx = (i + j) % self.group_order
                            if basis_pattern[idx] > (1.0 - sparsity):
                                mask[i, j] = 1.0
                                
                elif pattern_type == "prime":
                    # Prime-based pattern using the generator
                    for i in range(rows):
                        for j in range(cols):
                            # Use position-dependent index
                            idx = (i * 3 + j * 5) % self.group_order
                            if basis_pattern[idx] > (1.0 - sparsity):
                                mask[i, j] = 1.0
                                
                elif pattern_type == "orthogonal":
                    # Create orthogonal patterns
                    for i in range(rows):
                        for j in range(cols):
                            # Calculate orthogonal basis index
                            idx = (i ^ j) % self.group_order  # XOR operation
                            if basis_pattern[idx] > (1.0 - sparsity):
                                mask[i, j] = 1.0
            else:
                # For higher dimensions, use mod mask as fallback
                mask = self.create_mod_mask(shape, self.prime_moduli, sparsity)
        
        else:
            # Default to mod mask if pattern not recognized
            mask = self.create_mod_mask(shape, self.prime_moduli, sparsity)
        
        # Ensure target sparsity is maintained
        current_sparsity = 1.0 - (mask.sum().item() / mask.numel())
        
        # Only adjust if sparsity is significantly off
        if abs(current_sparsity - sparsity) > 0.1:
            if current_sparsity < sparsity:
                # Too dense, need more zeros
                active_indices = torch.where(mask > 0.5)
                if len(active_indices[0]) > 0:
                    num_excess = int((sparsity - current_sparsity) * mask.numel())
                    to_zero = torch.randperm(len(active_indices[0]), device=device)[:num_excess]
                    for i in range(len(to_zero)):
                        idx = tuple(dim[to_zero[i]] for dim in active_indices)
                        mask[idx] = 0.0
            else:
                # Too sparse, need more ones
                inactive_indices = torch.where(mask < 0.5)
                if len(inactive_indices[0]) > 0:
                    num_needed = int((current_sparsity - sparsity) * mask.numel())
                    to_one = torch.randperm(len(inactive_indices[0]), device=device)[:num_needed]
                    for i in range(len(to_one)):
                        idx = tuple(dim[to_one[i]] for dim in inactive_indices)
                        mask[idx] = 1.0
        
        return mask
    
    def create_mask(self, shape: Tuple[int, ...], sparsity: Optional[float] = None) -> torch.Tensor:
        """
        Create a mask for a parameter tensor.
        
        Args:
            shape (Tuple[int, ...]): Shape of the tensor to mask
            sparsity (float, optional): Target sparsity
            
        Returns:
            torch.Tensor: Binary mask of the same shape
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
            mask = self.create_mod_mask(shape, self.prime_moduli, sparsity)
            
        elif self.mask_type == "prime":
            mask = self.create_prime_mask(shape, sparsity)
            
        elif self.mask_type == "adaptive":
            # For adaptive masks, we need a weight tensor
            # Since we don't have it here, we'll create a dummy one
            # In practice, this would be called with the actual weight tensor
            weight_tensor = torch.randn(shape, device=self.get_device())
            mask = self.create_adaptive_mask(shape, weight_tensor, sparsity)
            
        elif self.mask_type == "quantum" and self.use_quantum_patterns:
            # Use our new quantum mask generator
            pattern_type = self.config.get("quantum_pattern_type", "harmonic")
            mask = self.create_quantum_mask(shape, sparsity, pattern_type)
        
        else:
            # Default to modular mask
            mask = self.create_mod_mask(shape, [2, 3, 5], sparsity)
        
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
            parameter (nn.Parameter): Parameter to mask
            weight_mask (torch.Tensor, optional): Mask to apply
            
        Returns:
            torch.Tensor: Masked parameter tensor
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
            model (nn.Module): Model to evolve masks for
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping parameter names to evolved masks
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
                        # Adjust to match target sparsity
                        if current_sparsity < target_sparsity:
                            # Too dense, need to deactivate more
                            active_indices = torch.where(new_mask > 0)
                            active_importances = importance[active_indices]
                            _, sorted_indices = torch.sort(active_importances)
                            
                            # Number to deactivate to reach target sparsity
                            num_excess = int((target_sparsity - current_sparsity) * num_total)
                            
                            # Deactivate lowest importance active params
                            for i in range(min(num_excess, len(sorted_indices))):
                                idx = tuple(dim[sorted_indices[i]] for dim in active_indices)
                                new_mask[idx] = 0.0
                        else:
                            # Too sparse, need to activate more
                            inactive_indices = torch.where(new_mask == 0)
                            inactive_importances = importance[inactive_indices]
                            _, sorted_indices = torch.sort(inactive_importances, descending=True)
                            
                            # Number to activate to reach target sparsity
                            num_needed = int((current_sparsity - target_sparsity) * num_total)
                            
                            # Activate highest importance inactive params
                            for i in range(min(num_needed, len(sorted_indices))):
                                idx = tuple(dim[sorted_indices[i]] for dim in inactive_indices)
                                new_mask[idx] = 1.0
                    
                    # Apply quantum pattern influence to maintain structured sparsity
                    if self.use_quantum_patterns:
                        # Generate quantum pattern at target sparsity
                        quantum_mask = self.create_quantum_mask(param.shape, target_sparsity, "harmonic")
                        
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
            model (nn.Module): The main model instance
        """
        self.model = model
        
        # Apply initial masks to model parameters
        if self.config.get("apply_masks_on_init", True):
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
            model (nn.Module): Model to apply masks to
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping parameter names to masks
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
            x (torch.Tensor): Input tensor
            model_outputs (Dict[str, Any], optional): Outputs from the main model
            extension_outputs (Dict[str, Any], optional): Outputs from other extensions
            
        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Modified tensor and extension metadata
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