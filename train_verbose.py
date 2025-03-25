#!/usr/bin/env python3
"""
Training script with verbose output for Semantic Resonance Language Model.

This script provides more detailed updates during training.
"""

import os
import torch
from transformers import AutoTokenizer
import logging
from tqdm import tqdm
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.config import ModelConfig, TrainingConfig, DataConfig
from src.model.semantic_resonance_model import SemanticResonanceModel
from src.data.wikitext_dataset import get_wikitext_dataloaders
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_perplexity, compute_entropy_stats


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    return logging.getLogger(__name__)


def train_model_verbose():
    """Train the model with verbose output."""
    logger = setup_logging()
    logger.info("Starting training with verbose output")
    
    # Model configuration
    model_config = ModelConfig(
        vocab_size=30000,  # Will be updated based on tokenizer
        hidden_dim=192,
        num_layers=4,
        num_heads=8,
        max_seq_length=512,
        dropout=0.1,
        primes=[24, 24, 24, 24, 24, 24, 24, 24],
        base_dim=384,  # Larger base dimension for better embedding
        max_iterations=10,
        entropy_threshold=0.1,
        use_prime_mask=True,
        enable_hcw=True,
        memory_size=1000,
        memory_key_dim=128
    )
    
    # Training configuration with more frequent logging and memory optimizations
    training_config = TrainingConfig(
        batch_size=8,    # Reduced batch size to prevent OOM errors
        learning_rate=5e-4,
        weight_decay=0.01,
        max_epochs=25,   # Reduced to 25 epochs for faster training
        warmup_steps=100,
        accumulation_steps=4,  # Accumulate gradients over 4 batches (effective batch size = 32)
        save_steps=100,   # Save more frequently
        eval_steps=50,    # Evaluate more frequently
        entropy_regularization_weight=0.01,
        adapter_l2_penalty=0.001,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_mixed_precision=torch.cuda.is_available()  # Use mixed precision on CUDA devices
    )
    
    # Data configuration
    data_config = DataConfig(
        dataset_name="wikitext-103-raw-v1",
        tokenizer_name="gpt2",
        max_length=512,
        stride=256,
        cache_dir=".cache"
    )
    
    # Output directory with explicit directory creation
    output_dir = "runs/verbose_training"
    # Make sure all parent directories exist
    if not os.path.exists("runs"):
        os.makedirs("runs", exist_ok=True)
    # Create the specific output directory
    os.makedirs(output_dir, exist_ok=True)
    training_config.output_dir = output_dir
    logger.info(f"Saving checkpoints and model to {output_dir}")
    
    # Set device
    device = torch.device(training_config.device)
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Update vocabulary size
    model_config.vocab_size = len(tokenizer)
    
    # Custom collate function to handle variable-length sequences
    def collate_fn(batch):
        """Collate function for padding sequences in a batch."""
        # Get batch elements
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad sequences to the same length
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 is ignored in loss
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded
        }
    
    # Load a smaller subset of the dataset for faster training
    logger.info("Loading dataset subset for faster training...")
    from datasets import load_dataset
    
    # Load a larger portion of the train dataset for better learning
    train_dataset = load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        split="train[:10000]",  # Use 10000 examples (10x more data)
        cache_dir=data_config.cache_dir
    )
    
    # Load the validation dataset
    val_dataset = load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        split="validation[:500]",  # Use 500 examples for validation
        cache_dir=data_config.cache_dir
    )
    
    # Create custom WikiTextDataset instances
    from src.data.wikitext_dataset import WikiTextDataset
    
    # Use the existing collate function
    from src.data.wikitext_dataset import collate_fn
    
    # Create datasets
    train_ds = WikiTextDataset(
        tokenizer=tokenizer,
        split="train",
        max_length=data_config.max_length,
        stride=data_config.stride,
        return_tensors=True,
        dataset=train_dataset  # Pass the dataset directly
    )
    
    val_ds = WikiTextDataset(
        tokenizer=tokenizer,
        split="validation",
        max_length=data_config.max_length,
        stride=data_config.stride,
        return_tensors=True,
        dataset=val_dataset  # Pass the dataset directly
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        collate_fn=collate_fn
    )
    
    # Create a dictionary of dataloaders
    dataloaders = {
        "train": train_loader,
        "validation": val_loader
    }
    
    logger.info(f"Created dataloaders with {len(train_ds)} training samples and {len(val_ds)} validation samples")
    
    # Check for existing checkpoints
    import glob
    checkpoint_files = sorted(glob.glob(os.path.join(output_dir, "checkpoint_epoch_*.pt")))
    
    start_epoch = 0
    if checkpoint_files:
        # Find the latest checkpoint
        latest_checkpoint = checkpoint_files[-1]
        logger.info(f"Found checkpoint: {latest_checkpoint}")
        
        # Load the checkpoint
        checkpoint = torch.load(latest_checkpoint)
        
        # Initialize model
        logger.info("Loading existing model from checkpoint...")
        model = SemanticResonanceModel(model_config)
        
        # Use strict=False to allow loading even with architecture changes
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        # Log missing and unexpected keys for debugging
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
            
        model.to(device)
        
        # Get the epoch number from the checkpoint filename
        import re
        epoch_match = re.search(r'epoch_(\d+)\.pt', latest_checkpoint)
        if epoch_match:
            start_epoch = int(epoch_match.group(1))
            logger.info(f"Resuming from epoch {start_epoch}")
        
        # Create optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        
        # Check if there were missing or unexpected keys in the model state dict
        architecture_changed = hasattr(model, "_load_missing_keys") and bool(model._load_missing_keys)
        
        try:
            # Only load optimizer state if model architecture is compatible
            if not architecture_changed:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Loaded optimizer state from checkpoint")
            else:
                logger.warning("Model architecture changed - using fresh optimizer state")
                
            # Print last loss
            logger.info(f"Previous loss: {checkpoint['loss']:.4f}")
        except (ValueError, KeyError) as e:
            logger.warning(f"Error loading optimizer state: {e}")
            logger.warning("Using fresh optimizer state instead")
    else:
        # Initialize new model
        logger.info("Initializing new model...")
        model = SemanticResonanceModel(model_config)
        model.to(device)
        
        # Create optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model size: {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Training loop
    logger.info("Starting training...")
    model.train()
    
    # Calculate the number of training steps per epoch
    train_steps_per_epoch = len(dataloaders["train"])
    logger.info(f"Training steps per epoch: {train_steps_per_epoch}")
    
    # Use a cosine annealing scheduler with warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    # Create a warmup phase that increases the learning rate linearly
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,  # Start at 10% of the base learning rate
        end_factor=1.0,    # Reach the full learning rate
        total_iters=train_steps_per_epoch * 2  # Warmup for 2 epochs
    )
    
    # Then use cosine annealing to gradually reduce learning rate
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_steps_per_epoch * (training_config.max_epochs - 2),  # Remaining epochs
        eta_min=1e-6  # Minimum learning rate at the end
    )
    
    # Combine both schedulers in sequence
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[train_steps_per_epoch * 2]  # Switch after 2 epochs
    )
    
    # Update scheduler setup to handle resuming from checkpoint
    if start_epoch > 0:
        logger.info(f"Adjusting scheduler for resumed training from epoch {start_epoch}")
        # Skip steps that would have happened in previous epochs
        for _ in range(start_epoch * train_steps_per_epoch):
            scheduler.step()
    
    for epoch in range(start_epoch, training_config.max_epochs):
        epoch_loss = 0.0
        epoch_step = 0
        
        progress_bar = tqdm(
            dataloaders["train"],
            desc=f"Epoch {epoch + 1}/{training_config.max_epochs}",
            leave=True
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
            # Standard single precision training
            # Forward pass
            outputs = model(**batch, return_dict=True)
            loss = outputs["loss"]
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % training_config.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            else:
                # Standard single precision training
                # Forward pass
                outputs = model(**batch, return_dict=True)
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                
                # Optimizer step
                if (batch_idx + 1) % training_config.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_step += 1
            
            # Log detailed statistics every 10 batches
            if batch_idx % 10 == 0:
                # Extract entropy and iteration information
                entropy_stats = compute_entropy_stats(outputs)
                
                # Log statistics
                current_lr = scheduler.get_last_lr()[0]
                
                log_msg = f"Epoch {epoch+1}, Batch {batch_idx}/{len(progress_bar)}, "
                log_msg += f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
                
                if "mean_entropy" in entropy_stats:
                    log_msg += f", Entropy: {entropy_stats['mean_entropy']:.4f}"
                
                if "mean_iterations" in entropy_stats:
                    log_msg += f", Avg Iter: {entropy_stats['mean_iterations']:.2f}"
                
                if "mean_convergence_gap" in entropy_stats:
                    log_msg += f", Gap: {entropy_stats['mean_convergence_gap']:.4f}"
                
                if "entropy_threshold" in entropy_stats:
                    log_msg += f", Threshold: {entropy_stats['entropy_threshold']:.4f}"
                
                logger.info(log_msg)
                
                # Log detailed entropy history for the first batch of each 5 epochs
                if batch_idx == 0 and (epoch % 5 == 0 or epoch < 5):
                    if "metadata" in outputs and outputs["metadata"]:
                        layer_data = []
                        # Collect data from each layer
                        for meta in outputs["metadata"]:
                            if "metadata" in meta and "entropy_history" in meta["metadata"]:
                                history = meta["metadata"]["entropy_history"]
                                layer_name = meta.get("layer", "unknown")
                                
                                # Log the per-iteration entropy values
                                logger.info(f"=== Layer {layer_name} Entropy History ===")
                                for iter_data in history:
                                    iteration = iter_data["iteration"]
                                    mean_ent = iter_data["mean_entropy"].mean().item()
                                    logger.info(f"  Iteration {iteration}: Mean Entropy = {mean_ent:.4f}")
                                
                                # Calculate and log entropy reduction rate
                                if len(history) > 1:
                                    first_ent = history[0]["mean_entropy"].mean().item()
                                    last_ent = history[-1]["mean_entropy"].mean().item()
                                    reduction = first_ent - last_ent
                                    pct_reduction = (reduction / first_ent) * 100 if first_ent != 0 else 0
                                    logger.info(f"  Total Entropy Reduction: {reduction:.4f} ({pct_reduction:.2f}%)")
                                    
                                    # Check if close to convergence
                                    threshold = meta["metadata"].get("entropy_threshold", 0.1)
                                    gap = last_ent - threshold
                                    logger.info(f"  Convergence Gap: {gap:.4f} (Threshold: {threshold:.4f})")
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{epoch_loss/epoch_step:.4f}",
                    "lr": f"{current_lr:.6f}"
                })
            
            # Evaluate model
            if batch_idx > 0 and batch_idx % 50 == 0:
                logger.info("Evaluating on validation set...")
                model.eval()
                
                val_loss = 0.0
                val_steps = 0
                
                with torch.no_grad():
                    val_progress = tqdm(dataloaders["validation"], desc="Validation", leave=False)
                    for val_batch in val_progress:
                        val_batch = {k: v.to(device) for k, v in val_batch.items() if k in ["input_ids", "attention_mask", "labels"]}
                        val_outputs = model(**val_batch, return_dict=True)
                        val_loss += val_outputs["loss"].item()
                        val_steps += 1
                
                avg_val_loss = val_loss / val_steps
                val_ppl = compute_perplexity(avg_val_loss)
                
                logger.info(f"Validation Loss: {avg_val_loss:.4f}, Perplexity: {val_ppl:.2f}")
                
                # Generate a sample
                if batch_idx % 200 == 0:
                    logger.info("Generating sample text...")
                    model.eval()
                    
                    prompt = "In the future, artificial intelligence will"
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
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_epoch_loss
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
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
    
    logger.info(f"Final Validation Loss: {avg_val_loss:.4f}, Perplexity: {val_ppl:.2f}")
    
    # Generate final sample
    logger.info("Generating final sample text...")
    model.eval()
    
    prompt = "In a world where quantum computing has become mainstream,"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=100,
            temperature=0.8,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
    
    sample_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    logger.info(f"Final sample:\n{sample_text}")
    
    # Save the final model
    final_model_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    logger.info(f"Training completed. Final model saved to {final_model_dir}")
    return model, tokenizer


if __name__ == "__main__":
    train_model_verbose()