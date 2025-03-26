"""
Mask Generators Module.

This module provides functions and classes for generating structured masks
for neural network parameters.
"""

import math
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantum_types import MaskType, MaskGeneratorFn


class MaskGenerators:
    """
    Provides methods for generating structured masks for neural networks.
    
    This class implements various masking strategies such as modular arithmetic-based,
    prime-based, and adaptive masks.
    """
    
    def __init__(self, device: torch.device = None):
        """
        Initialize mask generators.
        
        Args:
            device: Device to create masks on
        """
        self.device = device if device is not None else torch.device("cpu")
        
        # Cache for generated masks
        self.cached_masks = {}
    
    def get_device(self) -> torch.device:
        """Get the device for mask operations."""
        return self.device
    
    def create_mod_mask(
        self, 
        shape: Tuple[int, ...], 
        moduli: List[int], 
        sparsity: float
    ) -> torch.Tensor:
        """
        Create a mask using the modular arithmetic pattern.
        
        Args:
            shape: Shape of the tensor to mask
            moduli: List of moduli to use
            sparsity: Target sparsity
            
        Returns:
            Binary mask of the same shape
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
    
    def create_prime_mask(
        self, 
        shape: Tuple[int, ...], 
        sparsity: float
    ) -> torch.Tensor:
        """
        Create a mask using prime number patterns.
        
        Args:
            shape: Shape of the tensor to mask
            sparsity: Target sparsity
            
        Returns:
            Binary mask of the same shape
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
    
    def create_adaptive_mask(
        self, 
        shape: Tuple[int, ...], 
        weight_tensor: torch.Tensor, 
        sparsity: float,
        importance_measure: str = "magnitude"
    ) -> torch.Tensor:
        """
        Create an adaptive mask based on weight magnitudes.
        
        Args:
            shape: Shape of the tensor to mask
            weight_tensor: Weight tensor to adapt mask to
            sparsity: Target sparsity
            importance_measure: Measure of parameter importance
            
        Returns:
            Binary mask of the same shape
        """
        device = self.get_device()
        
        # Use the importance measure to determine mask
        if importance_measure == "magnitude":
            importance = torch.abs(weight_tensor)
        elif importance_measure == "gradient" and weight_tensor.grad is not None:
            importance = torch.abs(weight_tensor.grad)
        elif importance_measure == "sensitivity" and weight_tensor.grad is not None:
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
    
    def adjust_mask_sparsity(
        self, 
        mask: torch.Tensor, 
        target_sparsity: float
    ) -> torch.Tensor:
        """
        Adjust a mask to achieve a target sparsity level.
        
        Args:
            mask: Input binary mask
            target_sparsity: Target sparsity level (0.0-1.0)
            
        Returns:
            Adjusted binary mask
        """
        device = self.get_device()
        
        # Calculate current sparsity
        current_sparsity = 1.0 - (mask.sum().item() / mask.numel())
        
        # Only adjust if significantly off
        if abs(current_sparsity - target_sparsity) > 0.01:
            if current_sparsity < target_sparsity:
                # Too dense, need more zeros
                active_indices = torch.where(mask > 0.5)
                if len(active_indices[0]) > 0:
                    num_excess = int((target_sparsity - current_sparsity) * mask.numel())
                    to_zero = torch.randperm(len(active_indices[0]), device=device)[:num_excess]
                    for i in range(len(to_zero)):
                        idx = tuple(dim[to_zero[i]] for dim in active_indices)
                        mask[idx] = 0.0
            else:
                # Too sparse, need more ones
                inactive_indices = torch.where(mask < 0.5)
                if len(inactive_indices[0]) > 0:
                    num_needed = int((current_sparsity - target_sparsity) * mask.numel())
                    to_one = torch.randperm(len(inactive_indices[0]), device=device)[:num_needed]
                    for i in range(len(to_one)):
                        idx = tuple(dim[to_one[i]] for dim in inactive_indices)
                        mask[idx] = 1.0
        
        return mask
    
    def create_mask(
        self, 
        shape: Tuple[int, ...], 
        mask_type: MaskType, 
        sparsity: float,
        weight_tensor: Optional[torch.Tensor] = None,
        moduli: Optional[List[int]] = None,
        importance_measure: str = "magnitude"
    ) -> torch.Tensor:
        """
        Create a mask for a parameter tensor.
        
        Args:
            shape: Shape of the tensor to mask
            mask_type: Type of mask to create
            sparsity: Target sparsity
            weight_tensor: Optional weight tensor for adaptive masks
            moduli: Optional list of moduli for mod masks
            importance_measure: Measure for adaptive masks
            
        Returns:
            Binary mask of the same shape
        """
        # Check if we have this mask cached
        cache_key = f"{shape}_{sparsity}_{mask_type}"
        if cache_key in self.cached_masks:
            return self.cached_masks[cache_key]
        
        # Default moduli if not provided
        if moduli is None:
            moduli = [2, 3, 5, 7, 11]
        
        # Create the appropriate mask based on type
        if mask_type == "mod":
            mask = self.create_mod_mask(shape, moduli, sparsity)
            
        elif mask_type == "prime":
            mask = self.create_prime_mask(shape, sparsity)
            
        elif mask_type == "adaptive" and weight_tensor is not None:
            mask = self.create_adaptive_mask(
                shape, 
                weight_tensor, 
                sparsity, 
                importance_measure
            )
        else:
            # Default to modular mask
            mask = self.create_mod_mask(shape, moduli, sparsity)
        
        # Cache the mask
        self.cached_masks[cache_key] = mask
        
        return mask
    
    def clear_cache(self):
        """Clear the mask cache."""
        self.cached_masks = {}