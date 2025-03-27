"""
Dummy data loaders for testing.

This module provides dummy data loaders that can be used
when real datasets cannot be loaded, for testing purposes.
"""

import torch
import logging
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from src.data.tensor_collate import default_collate_fn, dialogue_collate_fn

# Set up logging
logger = logging.getLogger("qllm_dataloaders")


class DummyLMDataset(Dataset):
    """Dummy language modeling dataset for testing."""
    
    def __init__(self, size: int = 100, seq_length: int = 128, vocab_size: int = 10000):
        """
        Initialize the dummy dataset.
        
        Args:
            size: Number of examples in the dataset
            seq_length: Sequence length
            vocab_size: Vocabulary size
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    
    def __len__(self):
        """Get the number of examples."""
        return self.size
    
    def __getitem__(self, idx):
        """Get a dummy example."""
        # Create random input_ids and attention_mask
        input_ids = torch.randint(
            0, self.vocab_size, (self.seq_length,), dtype=torch.long
        )
        attention_mask = torch.ones(self.seq_length, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def create_dummy_dataloaders(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create dummy data loaders for testing.
    
    Args:
        tokenizer: Tokenizer to use for tokenization
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary of data loaders
    """
    logger.warning("Creating dummy language modeling datasets for testing")
    
    # Set evaluation batch size
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    # Create dummy datasets
    train_dataset = DummyLMDataset(size=100, vocab_size=len(tokenizer))
    val_dataset = DummyLMDataset(size=20, vocab_size=len(tokenizer))
    test_dataset = DummyLMDataset(size=20, vocab_size=len(tokenizer))
    
    # Create data loaders
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(num_workers, 2),  # Reduce workers for dummy data
        ),
        "validation": DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=min(num_workers, 2),
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=min(num_workers, 2),
        ),
    }
    
    return dataloaders


def create_dummy_dialogue_dataloaders(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 4,
    system_prompt: Optional[str] = None
) -> Dict[str, DataLoader]:
    """
    Create dummy dialogue data loaders for testing.
    
    Args:
        tokenizer: Tokenizer to use for tokenization
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading
        system_prompt: System prompt for dialogue
        
    Returns:
        Dictionary of data loaders
    """
    logger.warning("Creating dummy dialogue datasets for testing")
    
    # Set evaluation batch size
    if eval_batch_size is None:
        eval_batch_size = batch_size
        
    # Import dialogue dataset
    from src.data.dialogue_dataset import DialogueDataset
    
    # Create dummy dialogues
    dummy_dialogues = [
        [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
        ],
        [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about quantum computing."},
            {"role": "assistant", "content": "Quantum computing uses quantum mechanics to process information."}
        ]
    ] * 5  # Repeat to create more examples
    
    # Create dialogue datasets
    train_size = int(0.7 * len(dummy_dialogues))
    val_size = int(0.15 * len(dummy_dialogues))
    test_size = len(dummy_dialogues) - train_size - val_size
    
    all_dialogues = dummy_dialogues
    train_dialogues = all_dialogues[:train_size]
    val_dialogues = all_dialogues[train_size:train_size+val_size]
    test_dialogues = all_dialogues[train_size+val_size:]
    
    train_dataset = DialogueDataset(
        tokenizer=tokenizer,
        dialogues=train_dialogues,
        max_length=128  # Shorter for dummy data
    )
    
    val_dataset = DialogueDataset(
        tokenizer=tokenizer,
        dialogues=val_dialogues,
        max_length=128
    )
    
    test_dataset = DialogueDataset(
        tokenizer=tokenizer,
        dialogues=test_dialogues,
        max_length=128
    )
    
    # Create dataloaders
    dataloaders = {}
    
    dataloaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(num_workers, 2),  # Reduce workers for dummy data
        collate_fn=dialogue_collate_fn
    )
    
    dataloaders["validation"] = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=min(num_workers, 2),
        collate_fn=dialogue_collate_fn
    )
    
    dataloaders["test"] = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=min(num_workers, 2),
        collate_fn=dialogue_collate_fn
    )
    
    return dataloaders