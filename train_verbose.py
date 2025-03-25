#!/usr/bin/env python3
"""
Verbose training script for the Quantum Resonance Language Model.

This script provides detailed logging and visualization of the
training process, especially for the resonance attention mechanism.
It's primarily used for debugging and understanding the model.
"""

import os
import sys
import time
import argparse
import logging
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# Import from project modules
from src.utils.logging import setup_logger
from src.utils.device import get_device, print_device_info
from src.utils.config import ModelConfig, TrainingConfig, DataConfig
from src.utils.config import get_default_configs, save_configs
from src.model.semantic_resonance_model import SemanticResonanceModel 
from src.data.dataloaders import get_wikitext_dataloaders, compute_perplexity
from src.training.checkpoint import load_checkpoint, save_checkpoint, find_latest_checkpoint


def parse_verbose_args():
    """Parse command line arguments for verbose training."""
    parser = argparse.ArgumentParser(description="Verbose Quantum Resonance Model Training")
    
    # Basic configuration
    parser.add_argument("--output_dir", type=str, default="runs/verbose_training",
                        help="Output directory for model and logs")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--ignore_checkpoints", action="store_true",
                        help="Ignore existing checkpoints and start fresh")
    
    # Model configuration
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Size of hidden dimensions")
    parser.add_argument("--embedding_dim", type=int, default=256,
                       help="Size of embedding dimensions")
    parser.add_argument("--num_layers", type=int, default=4,
                       help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4,
                       help="Number of attention heads")
    parser.add_argument("--base_dim", type=int, default=64,
                       help="Base dimension for Prime Hilbert Encoder")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                       help="Maximum sequence length")
    
    # HCW configuration
    parser.add_argument("--enable_hcw", action="store_true",
                       help="Enable Homomorphic Computational Wrapper")
    parser.add_argument("--memory_size", type=int, default=64,
                       help="Memory size for HCW")
    parser.add_argument("--memory_key_dim", type=int, default=128,
                       help="Memory key dimension for HCW")
    
    # Resonance settings
    parser.add_argument("--max_iterations", type=int, default=10,
                       help="Maximum number of resonance iterations")
    parser.add_argument("--resonance_epsilon", type=float, default=0.1,
                       help="Convergence threshold for resonance")
    parser.add_argument("--resonance_momentum", type=float, default=0.2,
                       help="Momentum for resonance updates")
    parser.add_argument("--phase_factor", type=float, default=0.1,
                       help="Phase modulation factor")
    
    # Pre-Manifest Layer settings
    parser.add_argument("--pre_manifest_iterations", type=int, default=5,
                       help="Maximum iterations for pre-manifest layer")
    parser.add_argument("--pre_manifest_entropy_threshold", type=float, default=0.1,
                       help="Entropy threshold for pre-manifest layer")
    
    # Training settings
    parser.add_argument("--max_epochs", type=int, default=100,
                       help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size (reduced for full dataset)")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                       help="Learning rate")
    parser.add_argument("--subset_size", type=int, default=None,
                       help="Number of examples to use (defaults to full dataset)")
    
    # Logging settings
    parser.add_argument("--log_level", type=str, default="info",
                       choices=["debug", "info", "warning", "error"],
                       help="Logging level")
    parser.add_argument("--log_metrics_every", type=int, default=10,
                       help="Log metrics every N batches")
    parser.add_argument("--log_entropy_every", type=int, default=300,
                       help="Log detailed entropy stats every N batches")
    parser.add_argument("--log_samples", action="store_true",
                       help="Generate text samples during training")
    parser.add_argument("--log_layer_outputs", action="store_true",
                       help="Log detailed layer outputs (very verbose)")
    
    return parser.parse_args()


def setup_training_environment(args):
    """Set up the training environment."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, "verbose_training.log")
    logger = setup_logger(
        name="quantum_resonance",
        log_file=log_file,
        log_level=getattr(logging, args.log_level.upper())
    )
    
    # Log start
    logger.info("Starting training with verbose output")
    logger.info(f"Saving checkpoints and model to {args.output_dir}")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    return logger, device


def log_layer_entropy_stats(model, prefix=""):
    """Log detailed entropy statistics for each ResonanceAttention layer."""
    logger = logging.getLogger("quantum_resonance")
    
    # Find all ResonanceAttention layers
    for i, layer in enumerate(model.transformer.h):
        attention = layer.attn
        if not hasattr(attention, 'entropy_history'):
            continue
            
        # Log entropy history for this layer
        logger.info(f"=== Layer {i} Entropy History ===")
        for j, entropy in enumerate(attention.entropy_history):
            logger.info(f"  Iteration {j+1}: Mean Entropy = {entropy:.4f}")
        
        # Log summary statistics
        if len(attention.entropy_history) > 1:
            initial = attention.entropy_history[0]
            final = attention.entropy_history[-1]
            reduction = initial - final
            reduction_pct = (reduction / initial) * 100
            
            logger.info(f"  Total Entropy Reduction: {reduction:.4f} ({reduction_pct:.2f}%)")
            
            # Convergence gap
            if hasattr(attention, 'convergence_gap'):
                logger.info(f"  Convergence Gap: {attention.convergence_gap:.4f} (Threshold: {attention.epsilon:.4f})")


def train_model_verbose():
    """Train model with verbose logging of the quantum resonance process."""
    # Parse arguments
    args = parse_verbose_args()
    
    # Set up training environment
    logger, device = setup_training_environment(args)
    
    # Initialize configurations
    configs = get_default_configs()
    model_config = configs["model"]
    training_config = configs["training"]
    data_config = configs["data"]
    
    # Update model config from args
    model_config.hidden_dim = args.hidden_dim
    model_config.embedding_dim = args.embedding_dim  # Add this for resonance blocks
    model_config.num_layers = args.num_layers
    model_config.num_heads = args.num_heads
    model_config.base_dim = args.base_dim
    model_config.max_seq_length = args.max_seq_length
    model_config.max_iterations = args.max_iterations
    model_config.resonance_epsilon = args.resonance_epsilon
    model_config.entropy_threshold = args.resonance_epsilon  # Map epsilon to entropy_threshold
    model_config.resonance_momentum = args.resonance_momentum
    model_config.phase_factor = args.phase_factor
    
    # HCW settings
    model_config.enable_hcw = args.enable_hcw
    model_config.memory_size = args.memory_size
    model_config.memory_key_dim = args.memory_key_dim
    
    # Pre-Manifest Layer settings
    model_config.pre_manifest_iterations = args.pre_manifest_iterations
    model_config.pre_manifest_entropy_threshold = args.pre_manifest_entropy_threshold
    
    # Update training config from args
    training_config.max_epochs = args.max_epochs
    training_config.batch_size = args.batch_size
    training_config.learning_rate = args.learning_rate
    training_config.output_dir = args.output_dir
    
    # Save configurations
    save_configs(configs, args.output_dir)
    
    # Initialize tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_name)
    model_config.vocab_size = len(tokenizer)
    
    # Create dataloaders with the full dataset
    logger.info("Loading full dataset for comprehensive training...")
    start_time = time.time()
    dataloaders = get_wikitext_dataloaders(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=data_config.max_length,
        stride=data_config.stride,
        num_workers=data_config.preprocessing_num_workers,
        cache_dir=data_config.cache_dir,
        subset_size=None  # Use the entire dataset
    )
    
    # Log dataset loading time and size statistics
    loading_time = time.time() - start_time
    logger.info(f"Dataset loading completed in {loading_time:.2f} seconds")
    
    # Log dataset sizes
    total_train_examples = len(dataloaders["train"].dataset)
    total_val_examples = len(dataloaders["validation"].dataset) if "validation" in dataloaders else 0
    total_test_examples = len(dataloaders["test"].dataset) if "test" in dataloaders else 0
    
    logger.info(f"Created dataloaders with {total_train_examples} training samples and {total_val_examples} validation samples")
    
    # Estimate memory usage
    if torch.cuda.is_available():
        # Report current GPU memory usage
        allocated_memory = torch.cuda.memory_allocated() / 1024**2
        max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        logger.info(f"Current GPU memory usage: {allocated_memory:.2f}MB / {max_memory:.2f}MB")
    
    # Check for existing checkpoint
    if not args.ignore_checkpoints and not args.checkpoint_path:
        args.checkpoint_path = find_latest_checkpoint(args.output_dir)
        if args.checkpoint_path:
            logger.info(f"Found checkpoint: {args.checkpoint_path}")
    
    # Initialize model
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        logger.info("Loading existing model from checkpoint...")
        model = SemanticResonanceModel(model_config)
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        # Get epoch number from the checkpoint filename
        import re
        epoch_match = re.search(r'epoch_(\d+)\.pt', args.checkpoint_path)
        if epoch_match:
            start_epoch = int(epoch_match.group(1))
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            start_epoch = 0
        
        # Load model state dict
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        # Log missing and unexpected keys
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        
        model.to(device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        
        # Load optimizer state
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Loaded optimizer state from checkpoint")
            
            # Print last loss
            logger.info(f"Previous loss: {checkpoint['loss']:.4f}")
        except (ValueError, KeyError) as e:
            logger.warning(f"Error loading optimizer state: {e}")
            logger.warning("Using fresh optimizer state instead")
            
        # Create scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config.max_epochs * len(dataloaders["train"]),
            eta_min=1e-6
        )
        
        # Adjust for resumed training
        if start_epoch > 0:
            logger.info(f"Adjusting scheduler for resumed training from epoch {start_epoch}")
            for _ in range(start_epoch * len(dataloaders["train"])):
                lr_scheduler.step()
    else:
        # Initialize new model
        logger.info("Initializing new model...")
        model = SemanticResonanceModel(model_config)
        model.to(device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        
        # Create scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config.max_epochs * len(dataloaders["train"]),
            eta_min=1e-6
        )
        
        start_epoch = 0
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model size: {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Training steps per epoch
    steps_per_epoch = len(dataloaders["train"])
    logger.info(f"Training steps per epoch: {steps_per_epoch}")
    
    # Training loop
    logger.info("Starting training...")
    model.train()
    
    # Use AMP for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Train for specified number of epochs
    for epoch in range(start_epoch, args.max_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_step = 0
        
        # Create progress bar
        train_iterator = tqdm(
            dataloaders["train"],
            desc=f"Epoch {epoch+1}/{args.max_epochs}",
            leave=True
        )
        
        for batch_idx, batch in enumerate(train_iterator):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
            
            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(**batch, return_dict=True)
                loss = outputs["loss"]
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # LR scheduler step
            lr_scheduler.step()
            
            # Update statistics
            epoch_loss += loss.item()
            epoch_step += 1
            
            # Update progress bar
            lr = lr_scheduler.get_last_lr()[0]
            train_iterator.set_postfix(loss=loss.item(), avg_loss=epoch_loss/epoch_step, lr=lr)
            
            # Log training metrics
            if batch_idx % args.log_metrics_every == 0:
                # Get entropy details
                avg_iterations = 0
                avg_entropy = 0.0
                avg_gap = 0.0
                avg_threshold = 0.0
                
                count = 0
                for layer in model.transformer.h:
                    if hasattr(layer.attn, 'last_iterations'):
                        avg_iterations += layer.attn.last_iterations
                        count += 1
                    if hasattr(layer.attn, 'last_entropy'):
                        avg_entropy += layer.attn.last_entropy
                    if hasattr(layer.attn, 'convergence_gap'):
                        avg_gap += layer.attn.convergence_gap
                    if hasattr(layer.attn, 'epsilon'):
                        avg_threshold += layer.attn.epsilon
                
                if count > 0:
                    avg_iterations /= count
                    avg_entropy /= count
                    avg_gap /= count
                    avg_threshold /= count
                
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{steps_per_epoch}, "
                           f"Loss: {loss.item():.4f}, LR: {lr:.6f}, "
                           f"Entropy: {avg_entropy:.4f}, Avg Iter: {avg_iterations:.2f}, "
                           f"Gap: {avg_gap:.4f}, Threshold: {avg_threshold:.4f}")
                
                # Log detailed entropy statistics periodically
                if batch_idx % args.log_entropy_every == 0:
                    log_layer_entropy_stats(model)
            
            # Generate text samples if requested
            if args.log_samples and batch_idx % 500 == 0 and batch_idx > 0:
                model.eval()
                
                # Generate from a simple prompt
                prompt = "The quantum theory of resonance suggests that"
                
                with torch.no_grad():
                    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        output_ids = model.generate(
                            input_ids=input_ids,
                            max_length=50,
                            temperature=0.7,
                            do_sample=True,
                            top_k=50,
                            top_p=0.95
                        )
                    
                    sample_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    logger.info(f"Sample text: {sample_text}")
                
                # Back to training
                model.train()
        
        # End of epoch
        avg_epoch_loss = epoch_loss / epoch_step
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        
        # Check available disk space
        try:
            import shutil
            disk_stats = shutil.disk_usage(args.output_dir)
            free_space_mb = disk_stats.free / (1024 * 1024)
            logger.info(f"Available disk space: {free_space_mb:.2f} MB")
            
            if free_space_mb < 500:  # Less than 500MB available
                logger.warning(f"Low disk space warning: {free_space_mb:.2f} MB available")
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
        
        # Try to save the full checkpoint
        try:
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_epoch_loss
            }, checkpoint_path)
            logger.info(f"Checkpoint saved successfully to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            
            # Try to save just the model weights as a fallback
            try:
                model_only_path = os.path.join(args.output_dir, f"model_only_epoch_{epoch+1}.pt")
                logger.info(f"Attempting to save model-only checkpoint to {model_only_path}")
                torch.save(model.state_dict(), model_only_path)
                logger.info(f"Model-only checkpoint saved successfully")
            except Exception as e2:
                logger.error(f"Could not save model-only checkpoint either: {e2}")
                
                # Final fallback: try to save at a different location
                try:
                    alt_path = os.path.join(os.path.dirname(args.output_dir), f"emergency_save_epoch_{epoch+1}.pt")
                    logger.info(f"Final attempt: Saving model to alternative location: {alt_path}")
                    torch.save(model.state_dict(), alt_path)
                    logger.info(f"Emergency save successful at {alt_path}")
                except Exception as e3:
                    logger.error(f"All checkpoint saving methods failed: {e3}")
    
    # Final evaluation
    logger.info("Performing final evaluation...")
    model.eval()
    
    val_loss = 0.0
    val_steps = 0
    
    with torch.no_grad():
        for val_batch in tqdm(dataloaders["validation"], desc="Final Validation"):
            val_batch = {k: v.to(device) for k, v in val_batch.items() if k in ["input_ids", "attention_mask", "labels"]}
            val_outputs = model(**val_batch, return_dict=True)
            val_loss += val_outputs["loss"].item()
            val_steps += 1
    
    avg_val_loss = val_loss / val_steps
    val_ppl = compute_perplexity(avg_val_loss)
    logger.info(f"Final validation loss: {avg_val_loss:.4f}, perplexity: {val_ppl:.2f}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Print completion message
    logger.info("Verbose training completed!")


if __name__ == "__main__":
    try:
        train_model_verbose()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)