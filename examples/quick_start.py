#!/usr/bin/env python3
"""
Quick start example for Semantic Resonance Language Model.

This script demonstrates how to train a small model on a subset of
WikiText-103 for a few steps, to verify the implementation works.
It also shows how to save and load checkpoints.
"""

import os
import torch
import argparse
import json
from datetime import datetime
from transformers import AutoTokenizer
from datasets import load_dataset
import sys

# Add the repository root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ModelConfig, TrainingConfig, DataConfig
from src.model.semantic_resonance_model import SemanticResonanceModel
from src.training.checkpoint import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader


def save_training_checkpoint(
    model, 
    optimizer, 
    epoch, 
    step, 
    model_config, 
    training_config, 
    loss, 
    output_dir, 
    tokenizer=None
):
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        epoch: Current epoch
        step: Current step
        model_config: Model configuration
        training_config: Training configuration
        loss: Current loss value
        output_dir: Directory to save checkpoint in
        tokenizer: Optional tokenizer to save
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a checkpoint filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}_step{step}_{timestamp}.pt")
    
    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "model_config": model_config.to_dict() if hasattr(model_config, "to_dict") else vars(model_config),
        "training_config": training_config.to_dict() if hasattr(training_config, "to_dict") else vars(training_config),
        "loss": loss
    }
    
    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer_path = os.path.join(output_dir, "tokenizer")
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_path)
        
    # Save configs as JSON for easy inspection
    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump(checkpoint["model_config"], f, indent=4)
        
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(checkpoint["training_config"], f, indent=4)
    
    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def load_training_checkpoint(checkpoint_path, model, optimizer, device):
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load into
        optimizer: Optimizer to load into
        device: Device to load model to
        
    Returns:
        epoch, step, model_config, training_config, loss
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Restore model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Restore optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Return training state
    return (
        checkpoint["epoch"], 
        checkpoint["step"], 
        checkpoint["model_config"], 
        checkpoint["training_config"], 
        checkpoint["loss"]
    )


def quick_start_demo(resume_checkpoint=None):
    """Run a quick demonstration of the model."""
    print("Semantic Resonance Language Model - Quick Start Demo")
    print("===================================================")
    
    # Determine output directory for checkpoints
    output_dir = os.path.join("examples", "output", "quick_start_checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a minimal model configuration
    model_config = ModelConfig(
        vocab_size=30000,  # Will be updated based on tokenizer
        hidden_dim=768,    # Smaller than normal for quick testing
        num_layers=8,      # Reduced layers for faster training
        num_heads=8,       # Adjusted to be divisible into sum of primes (7+11+13=31*8=248)
        max_seq_length=512,  # Shorter sequences
        dropout=0.1,
        primes=[8, 8, 8, 8, 8, 8, 8, 8],  # Make sure sum is divisible by num_heads
        base_dim=256,
        max_iterations=10,  # Fewer iterations
        entropy_threshold=0.2,
        use_prime_mask=True,
        enable_hcw=True,
        memory_size=250,   # Smaller memory
        memory_key_dim=128  # Smaller key dimension
    )
    
    # Create training configuration
    training_config = TrainingConfig(
        batch_size=8,      # Small batch size
        learning_rate=5e-5,  # Slightly higher for quick convergence
        weight_decay=0.01,
        max_epochs=3,      # Multiple epochs for better training
        warmup_steps=10,
        accumulation_steps=1,
        save_steps=50,      # Save every 50 steps
        eval_steps=25,
        checkpoint_dir=output_dir
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
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:100000]")
    
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
    
    # Add generate method to model if it doesn't exist already
    if not hasattr(model, 'generate'):
        print("Adding generate method to model...")
        
        def generate(
            self,
            input_ids,
            max_length=50,
            temperature=1.0,
            do_sample=True,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            num_return_sequences=1,
            pad_token_id=None,
            **kwargs  # Accept additional parameters for compatibility
        ):
            """Generate text based on input_ids."""
            device = input_ids.device
            batch_size = input_ids.shape[0]
            
            # Initialize generated sequences with input_ids
            generated = input_ids.clone()
            
            # Create attention mask
            attention_mask = torch.ones_like(generated, device=device)
            
            # Continue generating until we reach max_length
            with torch.no_grad():
                for _ in range(max_length - generated.shape[1]):
                    # Get model outputs
                    outputs = self.forward(
                        input_ids=generated,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    
                    # Get logits for next token (last token in sequence)
                    next_token_logits = outputs["logits"][:, -1, :]
                    
                    # Apply temperature
                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                        filter_mask = torch.zeros_like(next_token_logits, device=device)
                        filter_mask.scatter_(1, top_k_indices, 1.0)
                        next_token_logits = torch.where(
                            filter_mask > 0,
                            next_token_logits,
                            torch.full_like(next_token_logits, float('-inf'))
                        )
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        sorted_probs = torch.softmax(sorted_logits, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits = torch.where(
                            indices_to_remove,
                            torch.full_like(next_token_logits, float('-inf')),
                            next_token_logits
                        )
                    
                    # Sample from the filtered distribution
                    if do_sample:
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_tokens = torch.multinomial(probs, num_samples=1)
                    else:
                        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Append the new token to the sequence
                    generated = torch.cat([generated, next_tokens], dim=1)
                    
                    # Update attention mask
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, 1), device=device)
                    ], dim=1)
                    
            return generated
        
        # Add the method to the model instance
        import types
        model.generate = types.MethodType(generate, model)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    
    # Initialize variables for training state
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')
    
    # Load checkpoint if specified or find the latest one
    checkpoint_to_load = resume_checkpoint
    
    # If no checkpoint specified, try to find the latest one
    if not checkpoint_to_load:
        # Look for checkpoints in the output directory
        checkpoint_files = []
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                if filename.startswith('checkpoint_') and filename.endswith('.pt'):
                    checkpoint_path = os.path.join(output_dir, filename)
                    checkpoint_files.append(checkpoint_path)
        
        if checkpoint_files:
            # Sort by modification time (newest first)
            checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            checkpoint_to_load = checkpoint_files[0]
            print(f"Found latest checkpoint: {os.path.basename(checkpoint_to_load)}")
    
    # Load the checkpoint if available
    if checkpoint_to_load:
        start_epoch, global_step, loaded_model_config, loaded_training_config, last_loss = \
            load_training_checkpoint(checkpoint_to_load, model, optimizer, device)
        
        print(f"Resuming from epoch {start_epoch+1}, step {global_step}")

        # Update configs with loaded values
        model_config = ModelConfig(**loaded_model_config)
        training_config = TrainingConfig(**loaded_training_config)
        
        # Start from the next epoch
        start_epoch += 1
    else:
        print("No checkpoints found. Starting training from scratch.")
        
    # Train for multiple epochs
    print(f"Training for {training_config.max_epochs} epochs starting from epoch {start_epoch+1}...")
    model.train()
    
    for epoch in range(start_epoch, training_config.max_epochs):
        epoch_loss = 0.0
        steps_in_epoch = 0
        print(f"\nEpoch {epoch+1}/{training_config.max_epochs}")
        print("-" * 30)
        
        for i, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
            
            # Forward pass
            outputs = model(**batch, return_dict=True)
            loss = outputs["loss"]
            
            # Track loss
            epoch_loss += loss.item()
            steps_in_epoch += 1
            global_step += 1
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Print progress
            print(f"Epoch {epoch+1}/{training_config.max_epochs}, Step {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}, Global Step: {global_step}")
            
            # Save checkpoint at regular intervals
            if global_step % training_config.save_steps == 0:
                checkpoint_path = save_training_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                    model_config=model_config,
                    training_config=training_config,
                    loss=loss.item(),
                    output_dir=output_dir,
                    tokenizer=tokenizer if global_step % (training_config.save_steps * 5) == 0 else None  # Save tokenizer less frequently
                )
            
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
        
        # Print epoch summary
        avg_epoch_loss = epoch_loss / steps_in_epoch
        print(f"\nEpoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint after each epoch
        epoch_checkpoint_path = save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
            model_config=model_config,
            training_config=training_config,
            loss=avg_epoch_loss,
            output_dir=output_dir,
            tokenizer=tokenizer  # Save tokenizer with epoch checkpoints
        )
        
        # Track best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = os.path.join(output_dir, "best_model.pt")
            print(f"New best model! Saving to {best_model_path}")
            torch.save({
                "model_state_dict": model.state_dict(),
                "loss": best_loss,
                "epoch": epoch,
                "model_config": model_config.to_dict() if hasattr(model_config, "to_dict") else vars(model_config),
            }, best_model_path)
    
    # Save final model
    final_output_dir = os.path.join("examples", "output", "tiny_model")
    os.makedirs(final_output_dir, exist_ok=True)
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
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
    print(f"Final model saved to {final_output_dir}")
    print(f"Checkpoints saved to {output_dir}")
    print(f"Best model saved to {os.path.join(output_dir, 'best_model.pt')}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Semantic Resonance Language Model Quick Start Demo")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from", default=None)
    args = parser.parse_args()
    
    # Make sure the examples directory exists
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    # Run demo with optional checkpoint resumption
    quick_start_demo(resume_checkpoint=args.resume)