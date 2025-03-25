"""
WikiText Dataset processing module.

This module provides utilities for downloading, preprocessing, and loading
the WikiText-103 dataset for training and evaluation.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class WikiTextDataset(Dataset):
    """
    Dataset class for WikiText-103 corpus.
    
    This class handles the preprocessing and tokenization of the WikiText-103
    dataset for language modeling tasks.
    """
    
    def __init__(self, tokenizer, split="train", max_length=512, stride=256, 
                 return_tensors=True, cache_dir=None, dataset=None):
        """
        Initialize the WikiText dataset.
        
        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer to use for preprocessing
            split (str): Dataset split to use (train, validation, or test)
            max_length (int): Maximum sequence length
            stride (int): Stride for overlapping sequences
            return_tensors (bool): Whether to return PyTorch tensors
            cache_dir (str, optional): Directory for caching the dataset
            dataset (Dataset, optional): Pre-loaded dataset to use instead of loading from scratch
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.return_tensors = return_tensors
        
        # Map dataset split names
        split_map = {
            "train": "train",
            "validation": "validation",
            "valid": "validation",
            "test": "test"
        }
        self.split = split_map.get(split, split)
        
        # Use provided dataset or load it
        if dataset is not None:
            self.raw_dataset = dataset
        else:
            # Load the dataset
            self.raw_dataset = load_dataset(
                "wikitext", 
                "wikitext-103-raw-v1", 
                split=self.split,
                cache_dir=cache_dir
            )
        
        # Preprocess the dataset
        self.examples = self._preprocess()
    
    def _preprocess(self):
        """
        Preprocess the dataset by tokenizing texts and creating examples.
        
        Returns:
            List[Dict]: Processed examples
        """
        tokenized_examples = []
        
        # Tokenize the entire dataset
        for article in self.raw_dataset["text"]:
            # Skip empty lines
            if not article.strip():
                continue
            
            # Tokenize the article
            tokenized_article = self.tokenizer(article, return_attention_mask=True)
            input_ids = tokenized_article["input_ids"]
            attention_mask = tokenized_article["attention_mask"]
            
            # Create examples with overlapping windows
            for i in range(0, len(input_ids), self.stride):
                begin_idx = i
                end_idx = min(i + self.max_length, len(input_ids))
                
                # Skip examples that are too short
                if end_idx - begin_idx < self.max_length // 2:
                    continue
                
                # Extract the window
                example = {
                    "input_ids": input_ids[begin_idx:end_idx],
                    "attention_mask": attention_mask[begin_idx:end_idx],
                    "labels": input_ids[begin_idx:end_idx].copy()
                }
                
                tokenized_examples.append(example)
                
                # Break if we've reached the end of the article
                if end_idx == len(input_ids):
                    break
        
        return tokenized_examples
    
    def __len__(self):
        """
        Get the number of examples in the dataset.
        
        Returns:
            int: Number of examples
        """
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        Get an example from the dataset.
        
        Args:
            idx (int): Index of the example
        
        Returns:
            Dict: Example with input_ids, attention_mask, and labels
        """
        example = self.examples[idx]
        
        if self.return_tensors:
            # Convert to PyTorch tensors
            return {
                "input_ids": torch.tensor(example["input_ids"]),
                "attention_mask": torch.tensor(example["attention_mask"]),
                "labels": torch.tensor(example["labels"])
            }
        
        return example


def create_wikitext_dataloader(tokenizer, split="train", batch_size=16, 
                               max_length=512, stride=256, shuffle=True, 
                               num_workers=4, cache_dir=None, collate_fn=None):
    """
    Create a DataLoader for the WikiText dataset.
    
    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to use for preprocessing
        split (str): Dataset split to use (train, validation, or test)
        batch_size (int): Batch size
        max_length (int): Maximum sequence length
        stride (int): Stride for overlapping sequences
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of workers for data loading
        cache_dir (str, optional): Directory for caching the dataset
        collate_fn (callable, optional): Function to collate data samples into batches
    
    Returns:
        torch.utils.data.DataLoader: DataLoader for the WikiText dataset
    """
    # Create the dataset
    dataset = WikiTextDataset(
        tokenizer=tokenizer,
        split=split,
        max_length=max_length,
        stride=stride,
        return_tensors=True,
        cache_dir=cache_dir
    )
    
    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return dataloader


def collate_fn(batch):
    """
    Collate function for padding sequences in a batch.
    
    Args:
        batch (List[Dict]): Batch of examples
    
    Returns:
        Dict: Padded batch
    """
    # Get max length in the batch
    max_len = max(len(example["input_ids"]) for example in batch)
    
    # Initialize padded arrays
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)  # -100 is ignored in loss
    
    # Fill in data
    for i, example in enumerate(batch):
        seq_len = len(example["input_ids"])
        input_ids[i, :seq_len] = example["input_ids"]
        attention_mask[i, :seq_len] = example["attention_mask"]
        labels[i, :seq_len] = example["labels"]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def get_wikitext_dataloaders(tokenizer, batch_size=16, max_length=512, stride=256,
                           num_workers=4, cache_dir=None, collate_fn=None):
    """
    Get DataLoaders for the train, validation, and test splits of WikiText.
    
    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to use for preprocessing
        batch_size (int): Batch size
        max_length (int): Maximum sequence length
        stride (int): Stride for overlapping sequences
        num_workers (int): Number of workers for data loading
        cache_dir (str, optional): Directory for caching the dataset
        collate_fn (callable, optional): Function to collate data samples into batches
    
    Returns:
        Dict: DataLoaders for train, validation, and test splits
    """
    train_loader = create_wikitext_dataloader(
        tokenizer=tokenizer,
        split="train",
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=True,
        num_workers=num_workers,
        cache_dir=cache_dir,
        collate_fn=collate_fn
    )
    
    val_loader = create_wikitext_dataloader(
        tokenizer=tokenizer,
        split="validation",
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False,
        num_workers=num_workers,
        cache_dir=cache_dir,
        collate_fn=collate_fn
    )
    
    test_loader = create_wikitext_dataloader(
        tokenizer=tokenizer,
        split="test",
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False,
        num_workers=num_workers,
        cache_dir=cache_dir,
        collate_fn=collate_fn
    )
    
    return {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader
    }