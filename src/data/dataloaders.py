"""
DataLoader factory functions for QLLM.

This module provides factory functions for creating DataLoader instances
for various types of datasets, handling common configuration options
and providing a simplified interface.
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Union, Optional, Callable

import torch
from torch.utils.data import DataLoader

from src.data.base import BaseDataset, BaseLoader
from src.data.datasets import (
    TextDataset,
    DialogueDataset,
    WikitextDataset,
    FunctionCallingDataset,
    MultimodalDataset
)
from src.data.loaders import (
    WikitextLoader,
    DailyDialogLoader
    # CustomLoader  # Not implemented yet
)
from src.data.utils import tensor_collate


logger = logging.getLogger("qllm.data")


def create_text_dataloader(
    data_path: str,
    tokenizer: Any,
    batch_size: int = 8,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    preprocessing_fn: Optional[Callable] = None,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for general text data.
    
    Args:
        data_path: Path to the data file or directory
        tokenizer: Tokenizer for processing text
        batch_size: Batch size for the dataloader
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        use_cache: Whether to use caching for preprocessed data
        cache_dir: Directory to use for caching
        preprocessing_fn: Optional function for preprocessing raw data
        collate_fn: Optional function for collating batches
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        **dataset_kwargs: Additional arguments for the dataset
        
    Returns:
        DataLoader for text data
    """
    # Create the dataset
    dataset = TextDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        use_cache=use_cache,
        cache_dir=cache_dir,
        preprocessing_fn=preprocessing_fn,
        **dataset_kwargs
    )
    
    # Use tensor collate function if not provided
    if collate_fn is None:
        collate_fn = tensor_collate["default_collate_fn"]
    
    # Create and return the dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def create_dialogue_dataloader(
    data_path: str,
    tokenizer: Any,
    batch_size: int = 8,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    preprocessing_fn: Optional[Callable] = None,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    max_history_turns: int = 3,
    separate_input_response: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for dialogue data.
    
    Args:
        data_path: Path to the dialogue data file or directory
        tokenizer: Tokenizer for processing text
        batch_size: Batch size for the dataloader
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        use_cache: Whether to use caching for preprocessed data
        cache_dir: Directory to use for caching
        preprocessing_fn: Optional function for preprocessing raw data
        collate_fn: Optional function for collating batches
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        max_history_turns: Maximum number of conversation history turns to include
        separate_input_response: Whether to keep input and response separate
        **dataset_kwargs: Additional arguments for the dataset
        
    Returns:
        DataLoader for dialogue data
    """
    # Create dialogue-specific dataset
    dataset = DialogueDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        use_cache=use_cache,
        cache_dir=cache_dir,
        preprocessing_fn=preprocessing_fn,
        max_history_turns=max_history_turns,
        separate_input_response=separate_input_response,
        **dataset_kwargs
    )
    
    # Use dialogue-specific collate function if not provided
    if collate_fn is None:
        collate_fn = tensor_collate["dialogue_collate_fn"]
    
    # Create and return the dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def create_wikitext_dataloader(
    data_path: str,
    tokenizer: Any,
    batch_size: int = 8,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    preprocessing_fn: Optional[Callable] = None,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for Wikitext data.
    
    Args:
        data_path: Path to the Wikitext data file or directory
        tokenizer: Tokenizer for processing text
        batch_size: Batch size for the dataloader
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        use_cache: Whether to use caching for preprocessed data
        cache_dir: Directory to use for caching
        preprocessing_fn: Optional function for preprocessing raw data
        collate_fn: Optional function for collating batches
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        **dataset_kwargs: Additional arguments for the dataset
        
    Returns:
        DataLoader for Wikitext data
    """
    # Create Wikitext-specific dataset
    dataset = WikitextDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        use_cache=use_cache,
        cache_dir=cache_dir,
        preprocessing_fn=preprocessing_fn,
        **dataset_kwargs
    )
    
    # Use tensor collate function if not provided
    if collate_fn is None:
        collate_fn = tensor_collate["default_collate_fn"]
    
    # Create and return the dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def create_function_calling_dataloader(
    data_path: str,
    tokenizer: Any,
    batch_size: int = 8,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    preprocessing_fn: Optional[Callable] = None,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    function_schema: Optional[Dict[str, Any]] = None,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for function calling data.
    
    Args:
        data_path: Path to the function calling data file or directory
        tokenizer: Tokenizer for processing text
        batch_size: Batch size for the dataloader
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        use_cache: Whether to use caching for preprocessed data
        cache_dir: Directory to use for caching
        preprocessing_fn: Optional function for preprocessing raw data
        collate_fn: Optional function for collating batches
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        function_schema: Schema for the functions (names, parameters, etc.)
        **dataset_kwargs: Additional arguments for the dataset
        
    Returns:
        DataLoader for function calling data
    """
    # Create function calling-specific dataset
    dataset = FunctionCallingDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        use_cache=use_cache,
        cache_dir=cache_dir,
        preprocessing_fn=preprocessing_fn,
        function_schema=function_schema,
        **dataset_kwargs
    )
    
    # Use function calling-specific collate function if not provided
    if collate_fn is None:
        collate_fn = tensor_collate["function_calling_collate_fn"]
    
    # Create and return the dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def create_multimodal_dataloader(
    data_path: str,
    tokenizer: Any,
    image_processor: Optional[Any] = None,
    batch_size: int = 8,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    preprocessing_fn: Optional[Callable] = None,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    modalities: List[str] = ["text", "image"],
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for multimodal data.
    
    Args:
        data_path: Path to the multimodal data file or directory
        tokenizer: Tokenizer for processing text
        image_processor: Processor for image data
        batch_size: Batch size for the dataloader
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        use_cache: Whether to use caching for preprocessed data
        cache_dir: Directory to use for caching
        preprocessing_fn: Optional function for preprocessing raw data
        collate_fn: Optional function for collating batches
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        modalities: List of modalities to include (e.g., ["text", "image"])
        **dataset_kwargs: Additional arguments for the dataset
        
    Returns:
        DataLoader for multimodal data
    """
    # Create multimodal-specific dataset
    dataset = MultimodalDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=max_length,
        use_cache=use_cache,
        cache_dir=cache_dir,
        preprocessing_fn=preprocessing_fn,
        modalities=modalities,
        **dataset_kwargs
    )
    
    # Use multimodal-specific collate function if not provided
    if collate_fn is None:
        collate_fn = tensor_collate["multimodal_collate_fn"]
    
    # Create and return the dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def create_dataloader_from_config(
    config: Dict[str, Any],
    tokenizer: Any,
    **override_kwargs
) -> DataLoader:
    """
    Create a DataLoader based on a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer for processing text
        **override_kwargs: Additional arguments to override config values
        
    Returns:
        Configured DataLoader
        
    Raises:
        ValueError: If the configuration is invalid or the dataset type is unknown
    """
    # Extract dataset type
    dataset_type = config.get("dataset_type", "text")
    
    # Merge config with override kwargs
    merged_kwargs = dict(config)
    merged_kwargs.update(override_kwargs)
    
    # Ensure tokenizer is set
    merged_kwargs["tokenizer"] = tokenizer
    
    # Create dataloader based on dataset type
    if dataset_type.lower() == "text":
        return create_text_dataloader(**merged_kwargs)
    elif dataset_type.lower() == "dialogue":
        return create_dialogue_dataloader(**merged_kwargs)
    elif dataset_type.lower() == "wikitext":
        return create_wikitext_dataloader(**merged_kwargs)
    elif dataset_type.lower() == "function_calling":
        return create_function_calling_dataloader(**merged_kwargs)
    elif dataset_type.lower() == "multimodal":
        return create_multimodal_dataloader(**merged_kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")