"""
WikiText dataset loader.

This module provides functions for loading and processing the WikiText dataset
for language modeling tasks.
"""

import os
import torch
import logging
import time
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader
from datasets import load_dataset, disable_caching
from transformers import PreTrainedTokenizer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from src.data.tensor_collate import default_collate_fn
from src.data.dataloader_utils import setup_cache_dir, load_from_cache, save_to_cache

# Set up logging
logger = logging.getLogger("qllm_dataloaders")

# Configure dataset library
disable_caching()  # Disable caching to prevent hanging on slow downloads


def get_wikitext_dataloaders(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    max_length: int = 512,
    stride: int = 256,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    variant: str = "wikitext-103-v1",
    timeout: int = 300
) -> Dict[str, DataLoader]:
    """
    Create data loaders for the WikiText dataset.
    
    Args:
        tokenizer: Tokenizer to use for tokenizing the dataset
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation (if None, use batch_size)
        max_length: Maximum sequence length
        stride: Stride for overlapping chunks
        num_workers: Number of workers for data loading
        cache_dir: Cache directory for datasets
        variant: Which WikiText variant to use
        timeout: Timeout in seconds for dataset download
        
    Returns:
        Dictionary of data loaders for train, validation, and test splits
    """
    # Set evaluation batch size
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    # Make sure cache directory exists
    cache_dir = setup_cache_dir(cache_dir)
    
    # Check for cached dataset to avoid redundant downloads
    cached_path = os.path.join(cache_dir or ".cache", f"wikitext-{variant}.cached")
    wikitext = load_from_cache(cached_path)
    
    if wikitext is None:
        # Load WikiText dataset with timeout protection
        logger.info(f"Downloading WikiText dataset (variant: {variant})...")
        
        try:
            # Use executor with timeout to prevent hanging
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(load_dataset, "wikitext", variant, cache_dir=cache_dir)
                try:
                    wikitext = future.result(timeout=timeout)
                    logger.info("WikiText dataset downloaded successfully")
                    # Cache the dataset for future use
                    if cache_dir:
                        save_to_cache(wikitext, cached_path)
                except TimeoutError:
                    logger.error(f"Timeout ({timeout}s) while downloading WikiText dataset")
                    raise ValueError(f"Dataset download timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Error loading {variant}: {e}")
            if variant == "wikitext-103-v1":
                logger.info("Trying wikitext-2-v1 instead...")
                try:
                    # Use smaller dataset with timeout
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(load_dataset, "wikitext", "wikitext-2-v1", cache_dir=cache_dir)
                        wikitext = future.result(timeout=timeout)
                        logger.info("WikiText-2 dataset downloaded successfully")
                except Exception as e2:
                    logger.error(f"Error loading wikitext-2: {e2}")
                    logger.warning("Using dummy datasets for testing")
                    from src.data.dummy_loaders import create_dummy_dataloaders
                    return create_dummy_dataloaders(
                        tokenizer, batch_size, eval_batch_size, num_workers
                    )
            else:
                logger.warning("Using dummy datasets for testing")
                from src.data.dummy_loaders import create_dummy_dataloaders
                return create_dummy_dataloaders(
                    tokenizer, batch_size, eval_batch_size, num_workers
                )
    
    # Display dataset sizes
    logger.info(f"WikiText dataset loaded: {len(wikitext['train'])} training examples")
    
    # Tokenization function
    def tokenize_function(examples):
        # Tokenize all texts
        tokenized = tokenizer(
            examples["text"],
            truncation=False,
            return_token_type_ids=False,
        )
        return tokenized
    
    # Apply tokenization
    logger.info("Tokenizing datasets...")
    tokenized_datasets = wikitext.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=["text"],
        desc="Tokenizing WikiText dataset",
    )
    
    # Create language modeling dataset
    def group_texts(examples):
        # Concatenate all texts
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        
        # Drop small remainder
        total_length = (total_length // max_length) * max_length
        
        # Split by chunks of max_length
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, stride)]
            for k, t in concatenated.items()
        }
        
        # Create labels
        result["labels"] = result["input_ids"].copy()
        
        return result
    
    # Apply grouping
    logger.info("Grouping texts...")
    grouped_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_workers,
        desc="Grouping texts",
    )
    
    # Create data loaders
    dataloaders = {}
    
    logger.info("Creating dataloaders...")
    train_dataset = grouped_datasets["train"]
    dataloaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=default_collate_fn,
    )
    
    validation_dataset = grouped_datasets["validation"]
    dataloaders["validation"] = DataLoader(
        validation_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=default_collate_fn,
    )
    
    test_dataset = grouped_datasets["test"]
    dataloaders["test"] = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=default_collate_fn,
    )
    
    logger.info(f"Created dataloaders: {len(dataloaders['train'])} training batches")
    
    return dataloaders