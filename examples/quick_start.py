#!/usr/bin/env python3
"""
Quick start example for Semantic Resonance Language Model.

This script demonstrates how to train a small model on a subset of
WikiText-103 for a few steps, to verify the implementation works.
"""

import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import sys

# Add the repository root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ModelConfig, TrainingConfig, DataConfig
from src.model.semantic_resonance_model import SemanticResonanceModel
from src.data.wikitext_dataset import WikiTextDataset
from torch.utils.data import DataLoader


def quick_start_demo():
    """Run a quick demonstration of the model."""
    print("Semantic Resonance Language Model - Quick Start Demo")
    print("===================================================")
    
    # Create a minimal model configuration
    model_config = ModelConfig(
        vocab_size=30000,  # Will be updated based on tokenizer
        hidden_dim=256,    # Smaller than normal for quick testing
        num_layers=2,      # Reduced layers for faster training
        num_heads=8,       # Adjusted to be divisible into sum of primes (7+11+13=31*8=248)
        max_seq_length=128,  # Shorter sequences
        dropout=0.1,
        primes=[8, 8, 8, 8, 8, 8, 8, 8],  # Make sure sum is divisible by num_heads
        base_dim=256,
        max_iterations=5,  # Fewer iterations
        entropy_threshold=0.2,
        use_prime_mask=True,
        enable_hcw=True,
        memory_size=100,   # Smaller memory
        memory_key_dim=64  # Smaller key dimension
    )
    
    # Create training configuration
    training_config = TrainingConfig(
        batch_size=4,      # Small batch size
        learning_rate=5e-4,  # Slightly higher for quick convergence
        weight_decay=0.01,
        max_epochs=1,      # Just one epoch for demo
        warmup_steps=10,
        accumulation_steps=1,
        save_steps=50,
        eval_steps=25
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Update vocabulary size
    model_config.vocab_size = len(tokenizer)
    
    # Load a small subset of WikiText
    print("Loading data...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:100]")
    
    # Prepare dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=model_config.max_seq_length)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.map(
        lambda examples: {"labels": examples["input_ids"]},
        batched=True
    )
    
    # Convert dataset to PyTorch format
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    
    # Create dataloader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=training_config.batch_size,
        shuffle=True
    )
    
    # Initialize model
    print("Initializing model...")
    model = SemanticResonanceModel(model_config)
    model.to(device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    
    # Train for a few steps
    print("Training for a few steps...")
    model.train()
    
    for i, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
        
        # Forward pass
        outputs = model(**batch, return_dict=True)
        loss = outputs["loss"]
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Print progress
        print(f"Step {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        # Get metadata from the last layer
        if "metadata" in outputs:
            last_layer_data = next((m for m in outputs["metadata"] if m.get("layer") == "final"), None)
            if last_layer_data and "metadata" in last_layer_data:
                entropy = last_layer_data["metadata"].get("entropy")
                iterations = last_layer_data["metadata"].get("iterations")
                
                if isinstance(entropy, torch.Tensor):
                    print(f"  Average entropy: {entropy.mean().item():.4f}")
                
                if isinstance(iterations, torch.Tensor):
                    avg_iters = iterations.float().mean().item()
                    print(f"  Average iterations: {avg_iters:.2f}")
        
        # Stop after 5 steps for quick demo
        if i >= 4:
            break
    
    # Save tiny model
    os.makedirs("examples/output", exist_ok=True)
    model.save_pretrained("examples/output/tiny_model")
    tokenizer.save_pretrained("examples/output/tiny_model")
    
    # Generate a short text sample
    print("\nGenerating text sample...")
    model.eval()
    
    # Prepare input
    prompt = "In the future, artificial intelligence will"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=30,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
    
    # Decode and print output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\nPrompt + Generated text:")
    print("-" * 40)
    print(generated_text)
    print("-" * 40)
    
    print("\nDemo completed successfully!")
    print(f"Saved model to examples/output/tiny_model")


if __name__ == "__main__":
    # Make sure the examples directory exists
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    quick_start_demo()