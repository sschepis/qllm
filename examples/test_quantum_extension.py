#!/usr/bin/env python
"""
Test script for the Quantum Group Symmetry Extension.

This example demonstrates how to use the enhanced SymmetryMaskExtension
with quantum-inspired masking patterns.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.extensions.quantum.symmetry_mask_extension import SymmetryMaskExtension


def visualize_mask(mask: torch.Tensor, title: str):
    """Visualize a 2D mask."""
    plt.figure(figsize=(8, 8))
    plt.imshow(mask.cpu().numpy(), cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.show()


def main():
    """Test the quantum mask extension."""
    # Create extension with quantum patterns enabled
    config = {
        "mask_type": "quantum",
        "mask_sparsity": 0.7,
        "mask_update_interval": 100,
        "use_quantum_patterns": True,
        "group_order": 8,
        "harmonic_levels": 5,
        "use_hilbert_projections": True,
        "hilbert_dim": 16,
        "quantum_pattern_type": "harmonic",  # Options: harmonic, hilbert, cyclic, prime, orthogonal
    }
    
    extension = SymmetryMaskExtension("quantum_symmetry", config)
    
    # Test creating different types of quantum masks
    shape = (64, 64)
    
    # Compare different quantum pattern types
    pattern_types = ["harmonic", "hilbert", "cyclic", "prime", "orthogonal"]
    
    # Check if we're running in a notebook or with display
    has_display = "DISPLAY" in os.environ or "JUPYTER_RUNTIME_DIR" in os.environ
    
    for pattern_type in pattern_types:
        print(f"Creating quantum mask with pattern type: {pattern_type}")
        # Set the pattern type in the config
        config["quantum_pattern_type"] = pattern_type
        extension.config = config
        
        # Create the mask
        mask = extension.create_quantum_mask(shape, 0.7, pattern_type)
        
        # Print statistics
        ones = mask.sum().item()
        total = mask.numel()
        sparsity = 1.0 - (ones / total)
        print(f"  - Mask shape: {mask.shape}")
        print(f"  - Non-zero elements: {ones}/{total}")
        print(f"  - Sparsity: {sparsity:.4f}")
        
        # Visualize the mask if display is available
        if has_display:
            visualize_mask(mask, f"Quantum Mask - {pattern_type}")
    
    # Test application to neural network parameters
    print("\nTesting application to neural network parameters...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100)
    )
    
    # Initialize the extension with the model
    extension.initialize(model)
    
    # Apply masks to model parameters
    masks = extension.apply_masks_to_model(model)
    
    # Print statistics
    total_params = sum(p.numel() for p in model.parameters())
    masked_params = extension.masked_parameter_count
    print(f"Total parameters: {total_params}")
    print(f"Masked parameters: {masked_params}")
    print(f"Overall sparsity: {masked_params / total_params:.4f}")
    
    # Print layer-wise sparsity
    print("\nLayer-wise sparsity:")
    for name, sparsity in extension.layer_sparsities.items():
        print(f"  - {name}: {sparsity:.4f}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()