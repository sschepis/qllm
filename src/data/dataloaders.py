"""
Data loading utilities for the Quantum Resonance Language Model.
Provides functions for creating and managing dataloaders for
training, validation, and inference.
"""

import os
import logging
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset, DatasetDict, Dataset as HFDataset
from transformers import PreTrainedTokenizer
from typing import Dict, Tuple, Optional, List, Union, Callable, Any

# Get logger
logger = logging.getLogger("quantum_resonance")


def get_tokenizer(tokenizer_name: str) -> PreTrainedTokenizer:
    """
    Get a tokenizer from the specified name.
    
    Args:
        tokenizer_name: Name or path of the tokenizer
        
    Returns:
        PreTrainedTokenizer: Loaded tokenizer
    """
    from transformers import AutoTokenizer
    
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Ensure we have a padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = "[PAD]"
            logger.warning("No pad or eos token found in tokenizer. Using [PAD] instead.")
    
    return tokenizer


def load_and_prepare_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    stride: int,
    dataset_config_name: Optional[str] = None,
    train_file: Optional[str] = None,
    validation_file: Optional[str] = None,
    test_file: Optional[str] = None,
    cache_dir: Optional[str] = None,
    preprocessing_num_workers: int = 4,
    overwrite_cache: bool = False,
    train_test_split: Optional[List[float]] = None,
    subset_size: Optional[int] = None,
    seed: int = 42
) -> Dict[str, HFDataset]:
    """
    Load and tokenize a dataset for language modeling.
    
    Args:
        dataset_name: Name of the dataset or path to local files
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        stride: Stride for sliding window tokenization
        dataset_config_name: Configuration name for the dataset
        train_file: Optional path to train file (if using local files)
        validation_file: Optional path to validation file
        test_file: Optional path to test file
        cache_dir: Directory to cache datasets
        preprocessing_num_workers: Number of workers for preprocessing
        overwrite_cache: Whether to overwrite cached files
        train_test_split: Optional split ratios [train, val, test] if not predefined
        subset_size: Optional limit on dataset size (for faster iteration)
        seed: Random seed for reproducibility
        
    Returns:
        Dict[str, Dataset]: Dictionary of datasets for each split
    """
    data_files = {}
    if train_file is not None:
        data_files["train"] = train_file
    if validation_file is not None:
        data_files["validation"] = validation_file
    if test_file is not None:
        data_files["test"] = test_file
    
    # Load dataset from file or by name
    if len(data_files) > 0:
        extension = train_file.split(".")[-1] if train_file else "text"
        dataset = load_dataset(extension, data_files=data_files, cache_dir=cache_dir)
    else:
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=cache_dir
        )
    
    # Handle dataset with a single split
    if isinstance(dataset, HFDataset):
        dataset = DatasetDict({"train": dataset})
    
    # Create splits if needed
    if "train" in dataset and "validation" not in dataset and "test" not in dataset and train_test_split:
        # Calculate number of samples for each split
        train_split, val_split, test_split = train_test_split
        total = len(dataset["train"])
        val_size = int(total * val_split)
        test_size = int(total * test_split)
        train_size = total - val_size - test_size
        
        # Create splits
        splits = dataset["train"].train_test_split(
            train_size=train_size,
            test_size=val_size + test_size,
            seed=seed
        )
        
        # Further split test into validation and test
        test_splits = splits["test"].train_test_split(
            train_size=val_size,
            test_size=test_size,
            seed=seed
        )
        
        # Combine into dataset dict
        dataset = DatasetDict({
            "train": splits["train"],
            "validation": test_splits["train"],
            "test": test_splits["test"]
        })
    
    # Apply subset limit if requested (for faster iteration)
    if subset_size is not None:
        for split in dataset:
            if len(dataset[split]) > subset_size:
                logger.info(f"Limiting {split} split to {subset_size} examples")
                dataset[split] = dataset[split].select(range(min(subset_size, len(dataset[split]))))
    
    # Select column to use for tokenization
    column_names = dataset["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    
    # Tokenize function for language modeling
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])
    
    # Apply tokenization
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )
    
    # Group texts for language modeling
    def group_texts(examples):
        # Concatenate all texts and create chunks
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Calculate how many chunks we will create
        total_length = (total_length // max_length) * max_length
        
        # Create chunks with stride
        result = {}
        for k, t in concatenated_examples.items():
            chunks = []
            for i in range(0, total_length, stride):
                chunk = t[i:i + max_length]
                if len(chunk) == max_length:
                    chunks.append(chunk)
            result[k] = chunks
        
        # Create labels
        result["labels"] = result["input_ids"].copy()
        
        return result
    
    # Apply grouping
    logger.info(f"Creating chunks of length {max_length} with stride {stride}...")
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_workers,
        desc=f"Grouping texts in chunks of {max_length}",
    )
    
    return lm_dataset


def create_dataloaders(
    datasets: Dict[str, HFDataset],
    batch_size: int,
    eval_batch_size: Optional[int] = None,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for each dataset split.
    
    Args:
        datasets: Dictionary of datasets for each split
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation (defaults to training batch_size)
        shuffle_train: Whether to shuffle the training data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory (speeds up GPU transfers)
        drop_last: Whether to drop the last incomplete batch
        collate_fn: Optional custom collation function
        
    Returns:
        Dict[str, DataLoader]: Dictionary of dataloaders for each split
    """
    dataloaders = {}
    
    # Default eval_batch_size to batch_size if not specified
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    # Create dataloader for each split
    for split_name, dataset in datasets.items():
        is_train = split_name == "train"
        
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size if is_train else eval_batch_size,
            shuffle=shuffle_train if is_train else False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=drop_last if is_train else False,
            collate_fn=collate_fn
        )
    
    # Log dataset sizes
    logger.info(f"Created dataloaders with:")
    for split, loader in dataloaders.items():
        logger.info(f"  {split}: {len(loader.dataset)} samples in {len(loader)} batches")
    
    return dataloaders


def get_wikitext_dataloaders(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    stride: int = 128,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    cache_dir: Optional[str] = ".cache",
    subset_size: Optional[int] = None
) -> Dict[str, DataLoader]:
    """
    Get dataloaders for the WikiText dataset.
    
    Args:
        tokenizer: Tokenizer to use
        batch_size: Batch size for training
        max_length: Maximum sequence length
        stride: Stride for sliding window tokenization
        eval_batch_size: Batch size for evaluation (defaults to batch_size)
        num_workers: Number of worker processes for data loading
        cache_dir: Directory to cache datasets
        subset_size: Optional limit on dataset size (for faster iteration)
        
    Returns:
        Dict[str, DataLoader]: Dictionary of dataloaders for each split
    """
    # Load the dataset
    datasets = load_and_prepare_dataset(
        dataset_name="wikitext",
        dataset_config_name="wikitext-103-raw-v1",
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        cache_dir=cache_dir,
        preprocessing_num_workers=num_workers,
        subset_size=subset_size
    )
    
    # Create dataloaders
    return create_dataloaders(
        datasets=datasets,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers
    )


def prepare_optimizer_groups(
    model: torch.nn.Module,
    weight_decay: float = 0.01,
    no_decay_keywords: List[str] = ["bias", "LayerNorm", "layernorm", "layer_norm"]
) -> List[Dict[str, Any]]:
    """
    Prepare parameter groups for optimizer with weight decay exclusions.
    
    Args:
        model: Model to prepare parameters for
        weight_decay: Weight decay factor
        no_decay_keywords: Parameter names containing these substrings will not use weight decay
        
    Returns:
        List[Dict]: Parameter groups for optimizer
    """
    # Create parameter groups for optimizer
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay_keywords) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay_keywords) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    
    # Count parameters in each group
    no_decay_param_count = sum(p.numel() for n, p in model.named_parameters()
                             if any(nd in n for nd in no_decay_keywords) and p.requires_grad)
    decay_param_count = sum(p.numel() for n, p in model.named_parameters()
                          if not any(nd in n for nd in no_decay_keywords) and p.requires_grad)
    
    logger.info(f"Using optimizer groups with weight decay {weight_decay}")
    logger.info(f"  Params with weight decay: {decay_param_count:,}")
    logger.info(f"  Params without weight decay: {no_decay_param_count:,}")
    
    return optimizer_grouped_parameters


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        float: Perplexity
    """
    return float(torch.exp(torch.tensor(loss)))