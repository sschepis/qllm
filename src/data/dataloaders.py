"""
Data loaders for the Quantum Resonance Language Model.

This module provides functions for creating data loaders for different
datasets used in training and evaluation.
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, Tuple, List
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

# Import specific loaders
from src.data.wikitext_loader import get_wikitext_dataloaders
from src.data.daily_dialog_loader import get_daily_dialog_dataloaders
from src.data.custom_loader import get_custom_dataloaders
from src.data.dummy_loaders import create_dummy_dataloaders, create_dummy_dialogue_dataloaders
from src.data.dataloader_utils import setup_cache_dir

# Set up logging
logger = logging.getLogger("qllm_dataloaders")


def get_appropriate_dataloaders(
    data_config,
    tokenizer,
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    timeout: int = 300
) -> Dict[str, DataLoader]:
    """
    Get appropriate data loaders based on the configuration.
    
    Args:
        data_config: Data configuration
        tokenizer: Tokenizer to use
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading
        timeout: Timeout in seconds for dataset operations
        
    Returns:
        Dictionary of data loaders
    """
    # Extract configuration values with defaults
    if num_workers is None:
        num_workers = getattr(data_config, "preprocessing_num_workers", 4)
    
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    dataset_name = getattr(data_config, "dataset_name", "").lower()
    
    logger.info(f"Creating dataloaders for dataset: {dataset_name}")
    
    try:
        if dataset_name == "wikitext":
            return get_wikitext_dataloaders(
                tokenizer=tokenizer,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                max_length=getattr(data_config, "max_length", 512),
                stride=getattr(data_config, "stride", 256),
                num_workers=num_workers,
                cache_dir=getattr(data_config, "cache_dir", None),
                variant=getattr(data_config, "dataset_variant", "wikitext-103-v1"),
                timeout=timeout
            )
        elif dataset_name == "daily_dialog":
            return get_daily_dialog_dataloaders(
                tokenizer=tokenizer,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                max_length=getattr(data_config, "max_length", 512),
                num_workers=num_workers,
                cache_dir=getattr(data_config, "cache_dir", None),
                system_prompt=getattr(data_config, "system_prompt", "You are a helpful assistant."),
                timeout=timeout,
                use_local_fallback=True
            )
        elif dataset_name == "custom":
            return get_custom_dataloaders(
                tokenizer=tokenizer,
                train_file=getattr(data_config, "train_file", ""),
                validation_file=getattr(data_config, "validation_file", None),
                test_file=getattr(data_config, "test_file", None),
                is_dialogue=hasattr(data_config, "system_prompt") and data_config.system_prompt is not None,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                max_length=getattr(data_config, "max_length", 512),
                num_workers=num_workers,
                system_prompt=getattr(data_config, "system_prompt", None),
                timeout=timeout
            )
        else:
            logger.warning(f"Unknown dataset name: {dataset_name}, using WikiText")
            return get_wikitext_dataloaders(
                tokenizer=tokenizer,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                max_length=getattr(data_config, "max_length", 512),
                stride=getattr(data_config, "stride", 256),
                num_workers=num_workers,
                cache_dir=getattr(data_config, "cache_dir", None),
                timeout=timeout
            )
    except Exception as e:
        logger.error(f"Error creating dataloaders for {dataset_name}: {e}")
        logger.warning("Falling back to dummy dataloaders")
        
        # Create appropriate dummy dataloaders based on dataset type
        if dataset_name == "daily_dialog":
            return create_dummy_dialogue_dataloaders(
                tokenizer=tokenizer,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                num_workers=num_workers,
                system_prompt=getattr(data_config, "system_prompt", "You are a helpful assistant.")
            )
        else:
            return create_dummy_dataloaders(
                tokenizer=tokenizer,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                num_workers=num_workers
            )