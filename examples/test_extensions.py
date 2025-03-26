#!/usr/bin/env python3
"""
Test script for the Enhanced Semantic Resonance Model with extensions.

This script demonstrates how to use the enhanced model with various extensions
for multimodal processing, extended memory, and quantum group symmetries.
"""

import os
import sys
import argparse
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.semantic_resonance_model_with_extensions import EnhancedSemanticResonanceModel
from src.config import ModelConfig
from src.model.extensions.extension_config import ExtensionConfig, MultimodalConfig, MemoryConfig, QuantumConfig
from src.model.extensions.multimodal.vision_extension import VisionExtension
from src.model.extensions.memory.knowledge_graph_extension import KnowledgeGraphExtension
from src.model.extensions.quantum.symmetry_mask_extension import SymmetryMaskExtension


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Semantic Resonance Model Extensions")
    
    parser.add_argument("--extension", type=str, default="all",
                        choices=["multimodal", "memory", "quantum", "all"],
                        help="Extension type to test")
    
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to image for multimodal testing")
    
    parser.add_argument("--input_text", type=str, 
                        default="The quantum resonance model can process",
                        help="Input text for generation")
    
    parser.add_argument("--generate_length", type=int, default=30,
                        help="Length of text to generate")
    
    parser.add_argument("--memory_path", type=str, default=None,
                        help="Path to load/save memory")
    
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (default: cuda if available)")
    
    return parser.parse_args()


def create_model_config():
    """Create a default model configuration."""
    # Using primes that sum to a value divisible by num_heads
    # 12 heads needs a sum divisible by 12
    config = ModelConfig(
        vocab_size=50257,  # Default GPT-2 vocabulary size
        hidden_dim=768,
        num_layers=6,
        num_heads=12,
        max_seq_length=512,
        dropout=0.1,
        primes=[12, 12, 12, 12, 12],  # Sum = 60, divisible by 12
        base_dim=768,
        max_iterations=10,
        entropy_threshold=0.1,
        use_prime_mask=True,
        enable_hcw=True,
        memory_size=1000,
        memory_key_dim=128
    )
    return config


def create_extension_config(extension_type="all"):
    """Create extension configuration."""
    # Create default config with all extensions disabled
    ext_config = ExtensionConfig(extensions_enabled=True)
    
    # Enable selected extensions
    if extension_type in ["multimodal", "all"]:
        # The prime sum (60) needs to be divisible by fusion_heads (default is 8)
        # So let's set the fusion_heads to 6 (60 ÷ 6 = 10)
        ext_config.multimodal = MultimodalConfig(
            enabled=True,
            vision_encoder_type="resnet",
            vision_encoder_model="resnet50",
            vision_embedding_dim=768,
            vision_primes=[12, 12, 12, 12, 12],  # Sum = 60
            fusion_heads=6,  # 60 is divisible by 6
            fusion_type="attention"
        )
    
    if extension_type in ["memory", "all"]:
        # We need to match the memory_value_dim with our embedding_dim (sum of primes)
        embedding_dim = 60  # Using 5 primes of 12 each
        
        ext_config.memory = MemoryConfig(
            enabled=True,
            memory_size=10000,
            memory_key_dim=128,
            memory_value_dim=embedding_dim,  # Match with the embedding dimension
            use_graph_structure=True,
            max_relations=5,
            num_neighbors=10
        )
    
    if extension_type in ["quantum", "all"]:
        ext_config.quantum = QuantumConfig(
            enabled=True,
            group_type="cyclic",
            group_order=5,
            mask_type="prime",
            mask_sparsity=0.8
        )
    
    return ext_config


def prepare_image_input(image_path):
    """Prepare image for model input."""
    if image_path is None:
        return None
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def prepare_text_input(text, device):
    """Prepare text for model input."""
    # Simple encoding for demonstration
    # In a real implementation, you would use a proper tokenizer
    char_to_id = {chr(i+96): i for i in range(1, 27)}
    char_to_id.update({' ': 27, '.': 28, ',': 29})
    
    # Encode text (simplified)
    input_ids = torch.tensor([[char_to_id.get(c.lower(), 0) for c in text]], device=device)
    
    return input_ids


def test_multimodal_extension(model, image_tensor, input_ids, device, generate_length=30):
    """Test the multimodal extension."""
    print("\n=== Testing Multimodal Extension ===")
    
    # Ensure device compatibility
    if image_tensor is not None:
        image_tensor = image_tensor.to(device)
    
    # Create extension inputs
    extension_inputs = {}
    if image_tensor is not None:
        extension_inputs["image_input"] = image_tensor
        print(f"Providing image input with shape: {image_tensor.shape}")
    
    # Generate text with image context
    generated_ids = model.generate(
        input_ids=input_ids,
        max_length=generate_length,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        **extension_inputs
    )
    
    # Simple decoding for demonstration
    id_to_char = {i: chr(i+96) for i in range(1, 27)}
    id_to_char.update({27: ' ', 28: '.', 29: ',', 0: '#'})
    
    generated_text = ''.join([id_to_char.get(i.item(), '#') for i in generated_ids[0]])
    
    print("\nGenerated text with image context:")
    print(generated_text)
    
    # Get vision extension stats
    vision_ext = model.get_extension("multimodal", "vision")
    if vision_ext:
        print("\nVision Extension Statistics:")
        metadata = vision_ext.forward(input_ids.expand(-1, model.config.hidden_dim), 
                                     {"image_input": image_tensor})[1]
        for key, value in metadata.items():
            if not isinstance(value, torch.Tensor):
                print(f"  {key}: {value}")
    
    return generated_text


def test_memory_extension(model, input_ids, device, memory_path=None, generate_length=30):
    """Test the memory extension."""
    print("\n=== Testing Memory Extension ===")
    
    # Get memory extension
    memory_ext = model.get_extension("memory", "knowledge_graph")
    
    # Load memory if path is provided
    if memory_path and memory_ext:
        memory_ext.persistence_path = memory_path
        memory_ext.persistence_enabled = True
        if os.path.exists(memory_path):
            success = memory_ext.load_memory()
            print(f"Loaded memory from {memory_path}: {success}")
    
    # Add some sample knowledge to the memory
    if memory_ext:
        print("Adding sample knowledge to memory...")
        
        # Create sample embeddings
        sample_keys = torch.randn(5, memory_ext.key_dim, device=device)
        sample_values = torch.randn(5, memory_ext.value_dim, device=device)
        
        # Sample relation data
        sample_metadata = {
            "entity_types": torch.tensor([1, 2, 1, 3, 2], device=device),
            "relations": [
                (0, 1, 1),  # Entity 0 relates to entity 1 with relation type 1
                (1, 2, 2),  # Entity 1 relates to entity 2 with relation type 2
                (2, 1, 0),  # Entity 2 relates to entity 0 with relation type 1
                (3, 3, 4),  # Entity 3 relates to entity 4 with relation type 3
            ]
        }
        
        # Add to memory
        indices = memory_ext.add_to_memory(sample_keys, sample_values, sample_metadata)
        print(f"Added entities with indices: {indices.cpu().numpy()}")
        
        # Get memory stats
        print("\nMemory Statistics:")
        stats = memory_ext.get_memory_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\nGraph Statistics:")
        # Check if the get_graph_stats method exists
        if hasattr(memory_ext, 'get_graph_stats'):
            graph_stats = memory_ext.get_graph_stats()
            for key, value in graph_stats.items():
                print(f"  {key}: {value}")
        else:
            print("  No graph statistics available")
        
        # Save memory if path is provided
        if memory_path:
            memory_ext.persistence_path = memory_path
            memory_ext.persistence_enabled = True
            memory_ext.persist_memory()
            print(f"Saved memory to {memory_path}")
    
    # Generate text with memory context
    print("\nGenerating text with memory context...")
    extension_inputs = {"is_training": False}  # Don't update memory during generation
    
    generated_ids = model.generate(
        input_ids=input_ids,
        max_length=generate_length,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        **extension_inputs
    )
    
    # Simple decoding for demonstration
    id_to_char = {i: chr(i+96) for i in range(1, 27)}
    id_to_char.update({27: ' ', 28: '.', 29: ',', 0: '#'})
    
    generated_text = ''.join([id_to_char.get(i.item(), '#') for i in generated_ids[0]])
    
    print("\nGenerated text with memory context:")
    print(generated_text)
    
    return generated_text


def test_quantum_extension(model, input_ids, device, generate_length=30):
    """Test the quantum group symmetry extension."""
    print("\n=== Testing Quantum Group Symmetry Extension ===")
    
    # Get quantum extension
    quantum_ext = model.get_extension("quantum", "symmetry_mask")
    
    if quantum_ext:
        # Apply masks to model parameters
        print("Applying quantum masks to model parameters...")
        quantum_ext.apply_masks_to_model(model)
        
        # Get mask statistics
        print("\nMask Statistics:")
        masks = quantum_ext.cached_masks
        print(f"  Total masks created: {len(masks)}")
        
        total_params = quantum_ext.total_parameter_count
        masked_params = quantum_ext.masked_parameter_count
        if total_params > 0:
            sparsity = masked_params / total_params
            print(f"  Total parameters: {total_params:,}")
            print(f"  Masked parameters: {masked_params:,}")
            print(f"  Overall sparsity: {sparsity:.2%}")
        
        # Print some layer sparsities
        print("\nLayer sparsities:")
        for layer_name, sparsity in list(quantum_ext.layer_sparsities.items())[:5]:
            print(f"  {layer_name}: {sparsity:.2%}")
        
        if len(quantum_ext.layer_sparsities) > 5:
            print(f"  ... and {len(quantum_ext.layer_sparsities) - 5} more layers")
    
    # Generate text with quantum masks applied
    print("\nGenerating text with quantum masks applied...")
    
    generated_ids = model.generate(
        input_ids=input_ids,
        max_length=generate_length,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    
    # Simple decoding for demonstration
    id_to_char = {i: chr(i+96) for i in range(1, 27)}
    id_to_char.update({27: ' ', 28: '.', 29: ',', 0: '#'})
    
    generated_text = ''.join([id_to_char.get(i.item(), '#') for i in generated_ids[0]])
    
    print("\nGenerated text with quantum masks:")
    print(generated_text)
    
    return generated_text


def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model and extension configurations
    model_config = create_model_config()
    ext_config = create_extension_config(args.extension)
    
    # Create enhanced model
    model = EnhancedSemanticResonanceModel(model_config, ext_config)
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    print(f"Extensions enabled: {ext_config.extensions_enabled}")
    print(f"Multimodal extension: {ext_config.multimodal.enabled}")
    print(f"Memory extension: {ext_config.memory.enabled}")
    print(f"Quantum extension: {ext_config.quantum.enabled}")
    
    # Prepare inputs
    input_text = args.input_text
    input_ids = prepare_text_input(input_text, device)
    print(f"\nInput text: {input_text}")
    
    image_tensor = None
    if args.image_path:
        image_tensor = prepare_image_input(args.image_path)
        print(f"Image loaded from: {args.image_path}")
    
    # Test extensions based on user selection
    if args.extension in ["multimodal", "all"] and ext_config.multimodal.enabled:
        if image_tensor is not None:
            test_multimodal_extension(model, image_tensor, input_ids, device, args.generate_length)
        else:
            print("\nSkipping multimodal test: no image provided")
    
    if args.extension in ["memory", "all"] and ext_config.memory.enabled:
        test_memory_extension(model, input_ids, device, args.memory_path, args.generate_length)
    
    if args.extension in ["quantum", "all"] and ext_config.quantum.enabled:
        test_quantum_extension(model, input_ids, device, args.generate_length)
    
    print("\nExtension testing complete!")


if __name__ == "__main__":
    main()