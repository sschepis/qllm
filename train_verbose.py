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
from src.data.tensor_collate import debug_batch_structure
from src.data.batch_utils import batch_to_device, debug_batch_contents
from src.model.fixed_autocast import device_aware_autocast
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
    parser.add_argument("--batch_size", type=int, default=16,
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
    
    # Check for MPS (Apple Silicon) device and warn about float64 limitations
    if device.type == 'mps':
        logger.info("Detected Apple Silicon MPS device - enabling float64 workarounds")
        logger.info("Note: Some operations may use lower precision due to MPS limitations")
        
        # Verify PyTorch version is compatible with MPS
        pt_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if pt_version < (1, 12):
            logger.warning(f"PyTorch {torch.__version__} may have limited MPS support. Version 1.12+ recommended.")
    
    # Set random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    return logger, device


def log_layer_entropy_stats(model, prefix=""):
    """Log detailed entropy statistics for each ResonanceBlock layer by running a forward pass."""
    logger = logging.getLogger("quantum_resonance")
    
    # Create a small sample input to get metadata
    device = next(model.parameters()).device
    sample_input = torch.randint(0, model.config.vocab_size, (1, 24), device=device)
    
    # Track entropy history across layers
    with torch.no_grad():
        # Get initial embeddings
        encoder_output = model.encoder(sample_input)
        hidden_states = model.encoder_projection(encoder_output)
        hidden_states = model.encoder_norm(hidden_states)
        
        # Pass through each layer and collect metadata
        for i, layer in enumerate(model.layers):
            # Run forward pass on this layer to get metadata
            with device_aware_autocast(device=device, enabled=True):
                _, metadata = layer(hidden_states.clone(), None, return_attention=True)
            
            # Log entropy history if available in metadata
            if 'entropy_history' in metadata and metadata['entropy_history']:
                logger.info(f"=== Layer {i} Entropy History ===")
                
                # Extract mean entropy for each iteration
                for j, entry in enumerate(metadata['entropy_history']):
                    if isinstance(entry, dict) and 'mean_entropy' in entry:
                        # Extract from dictionary structure
                        mean_entropy = entry['mean_entropy'].mean().item()
                        logger.info(f"  Iteration {j+1}: Mean Entropy = {mean_entropy:.4f}")
                    elif isinstance(entry, torch.Tensor):
                        # Handle case where it might be a direct tensor
                        mean_entropy = entry.mean().item()
                        logger.info(f"  Iteration {j+1}: Mean Entropy = {mean_entropy:.4f}")
                
                # Log summary statistics if we have multiple iterations
                if len(metadata['entropy_history']) > 1:
                    # Get initial and final entropy values
                    if isinstance(metadata['entropy_history'][0], dict):
                        # Safely extract and convert mean entropy
                        if isinstance(metadata['entropy_history'][0]['mean_entropy'], torch.Tensor):
                            # Ensure it's a floating point tensor before computing mean
                            initial_tensor = metadata['entropy_history'][0]['mean_entropy'].float()
                            final_tensor = metadata['entropy_history'][-1]['mean_entropy'].float()
                            initial = initial_tensor.mean().item()
                            final = final_tensor.mean().item()
                        else:
                            # Handle case where it might be a scalar value
                            initial = metadata['entropy_history'][0]['mean_entropy']
                            final = metadata['entropy_history'][-1]['mean_entropy']
                    else:
                        # Direct tensor case - ensure floating point
                        initial_tensor = metadata['entropy_history'][0].float() if metadata['entropy_history'][0].dtype != torch.float else metadata['entropy_history'][0]
                        final_tensor = metadata['entropy_history'][-1].float() if metadata['entropy_history'][-1].dtype != torch.float else metadata['entropy_history'][-1]
                        initial = initial_tensor.mean().item()
                        final = final_tensor.mean().item()
                    
                    # Calculate reduction
                    reduction = initial - final
                    reduction_pct = (reduction / initial) * 100 if initial != 0 else 0
                    
                    logger.info(f"  Total Entropy Reduction: {reduction:.4f} ({reduction_pct:.2f}%)")
                    
                    # Convergence gap
                    if 'convergence_gap' in metadata:
                        # Ensure gap is a floating point tensor
                        if isinstance(metadata['convergence_gap'], torch.Tensor):
                            gap_tensor = metadata['convergence_gap'].float() if metadata['convergence_gap'].dtype != torch.float else metadata['convergence_gap']
                            gap = gap_tensor.mean().item()
                        else:
                            gap = metadata['convergence_gap']
                        threshold = metadata.get('entropy_threshold', 0.1)
                        logger.info(f"  Convergence Gap: {gap:.4f} (Threshold: {threshold:.4f})")


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
    
    # Log model configuration details
    logger.info("Model Architecture Configuration:")
    logger.info(f"- Hidden dimension: {model_config.hidden_dim}")
    logger.info(f"- Embedding dimension: {model_config.embedding_dim}")
    logger.info(f"- Base dimension: {model_config.base_dim}")
    logger.info(f"- Prime numbers: {model_config.primes} (sum: {sum(model_config.primes)})")
    logger.info(f"- Encoder output dimension: {sum(model_config.primes)}")
    logger.info(f"- Projection: {sum(model_config.primes)} → {model_config.embedding_dim}")
    
    # Use AMP for mixed precision training - with device-aware config
    if torch.cuda.is_available():
        logger.info("Using CUDA mixed-precision training")
        scaler = torch.amp.GradScaler()
    elif device.type == 'mps':
        # For MPS (Apple Silicon), we need to handle the fact that it doesn't support float64
        logger.info("Using MPS-compatible mixed-precision training")
        # Use float32 scale to avoid float64 conversion in unscale_
        torch._C._jit_set_profiling_executor(False)  # Disable JIT to avoid other MPS issues
        scaler = torch.amp.GradScaler(enabled=True)
    else:
        logger.info("Mixed-precision not available - using CPU")
        scaler = torch.amp.GradScaler(enabled=False)
    
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
            # Debug first batch details for understanding structure
            if batch_idx == 0:
                logger.info(f"First batch structure debugging (device={device}):")
                debug_batch_structure(batch, name="Initial batch")
                debug_batch_contents(batch)
            
            try:
                # Move batch to device with robust error handling
                batch = batch_to_device(batch, device)
                
                # Log the processed batch for the first iteration
                if batch_idx == 0:
                    logger.info("After batch_to_device processing:")
                    debug_batch_structure(batch, name="Processed batch")
                    if 'input_ids' in batch:
                        logger.info(f"Input shape: {batch['input_ids'].shape}")
                    else:
                        logger.warning("No input_ids found in batch!")
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                if batch_idx == 0:
                    # Much more detailed error reporting for the first batch
                    logger.error("Batch content details:")
                    for k, v in batch.items():
                        try:
                            logger.error(f"  {k}: {type(v).__name__}, {str(v)[:100]}")
                        except:
                            logger.error(f"  {k}: {type(v).__name__}, <cannot display>")
                raise
            
            # Validate batch has required keys before forward pass
            for key in ["input_ids", "attention_mask", "labels"]:
                if key not in batch:
                    raise ValueError(f"Required key '{key}' missing from batch")
                if not isinstance(batch[key], torch.Tensor):
                    raise ValueError(f"Batch[{key}] is not a tensor but {type(batch[key])}")
            
            # Forward pass with AMP using device-aware autocast
            with device_aware_autocast(device=device, enabled=True):
                outputs = model(**batch, return_dict=True)
                loss = outputs["loss"]
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Handle gradient unscaling with MPS compatibility
            if device.type == 'mps':
                try:
                    # Try standard unscaling first
                    scaler.unscale_(optimizer)
                except TypeError as e:
                    if "Cannot convert a MPS Tensor to float64" in str(e):
                        # MPS-specific workaround: manually unscale without using double precision
                        if scaler.is_enabled() and scaler._scale is not None:
                            for param_group in optimizer.param_groups:
                                for param in param_group['params']:
                                    if param.grad is not None:
                                        # Use direct division by scale as float32
                                        param.grad.div_(scaler._scale.to(dtype=torch.float32))
                        logger.debug("Applied MPS-compatible gradient unscaling")
                    else:
                        # Re-raise if it's not the expected MPS float64 error
                        raise
            else:
                # Standard unscaling for other devices
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
                
                # Run a forward pass to get metadata
                with torch.no_grad():
                    sample_batch = {
                        'input_ids': batch['input_ids'][:1],  # Take just first sample to save computation
                        'attention_mask': batch['attention_mask'][:1] if 'attention_mask' in batch else None
                    }
                    
                    # Get normalized entropy information from layers
                    count = 0
                    for i, layer in enumerate(model.layers):
                        # Get metadata from a forward pass through just this layer
                        with device_aware_autocast(device=device, enabled=True):
                            # Get the input to this layer (output of previous layer or the embedding)
                            if i == 0:
                                # For first layer, use embeddings
                                encoder_output = model.encoder(sample_batch['input_ids'])
                                layer_input = model.encoder_projection(encoder_output)
                                layer_input = model.encoder_norm(layer_input)
                            else:
                                # For other layers, we would need previous layer output
                                # This is simplified - in practice you might need to track the full forward pass
                                continue
                                
                            # Pass through just this layer
                            _, layer_metadata = layer(layer_input, None)
                            
                            # Extract metrics from metadata
                            if 'iterations' in layer_metadata:
                                # Convert Long tensor to float before computing mean
                                iterations = layer_metadata['iterations'].float()
                                avg_iterations += iterations.mean().item()
                                count += 1
                            if 'entropy' in layer_metadata:
                                avg_entropy += layer_metadata['entropy'].mean().item()
                            if 'convergence_gap' in layer_metadata:
                                # Ensure tensor is float type before computing mean
                                gap = layer_metadata['convergence_gap'].float() if layer_metadata['convergence_gap'].dtype != torch.float else layer_metadata['convergence_gap']
                                avg_gap += gap.mean().item()
                            if 'entropy_threshold' in layer_metadata:
                                # Handle threshold which might be a scalar or tensor
                                if isinstance(layer_metadata['entropy_threshold'], torch.Tensor):
                                    avg_threshold += layer_metadata['entropy_threshold'].item()
                                else:
                                    avg_threshold += layer_metadata['entropy_threshold']
                
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
                
                try:
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
                except Exception as e:
                    logger.warning(f"Error during text generation: {e}")
                    logger.info("Continuing with training without generation sample")
                finally:
                    # Always ensure we return to training mode
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