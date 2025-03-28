"""
Base interface for dataset adapters in the enhanced training system.

This module defines the abstract base class that all dataset adapters must implement.
Dataset adapters provide a consistent interface to handle various dataset types
and formats for model training.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader


class DatasetAdapter(ABC):
    """
    Abstract base class for dataset adapters.
    
    Dataset adapters handle dataset-specific operations like loading data,
    creating dataloaders, applying transformations, and preparing batches.
    """
    
    def __init__(
        self,
        config: Any,
        tokenizer: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize the dataset adapter.
        
        Args:
            config: Configuration for the dataset
            tokenizer: Tokenizer to use for processing text
            **kwargs: Additional keyword arguments
        """
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    @abstractmethod
    def prepare_datasets(self) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
        """
        Prepare train, validation, and test datasets.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
            val_dataset and test_dataset may be None
        """
        pass
    
    @abstractmethod
    def create_dataloaders(
        self,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        **kwargs
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Create dataloaders for train, validation, and test datasets.
        
        Args:
            train_batch_size: Batch size for training
            eval_batch_size: Batch size for evaluation
            **kwargs: Additional arguments for DataLoader
            
        Returns:
            Tuple of (train_dataloader, val_dataloader, test_dataloader)
            val_dataloader and test_dataloader may be None
        """
        pass
    
    @abstractmethod
    def process_batch(
        self,
        batch: Any,
        is_train: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Process a batch from the dataloader.
        
        Args:
            batch: Batch from dataloader
            is_train: Whether the batch is for training
            
        Returns:
            Processed batch ready for model input
        """
        pass
    
    def set_tokenizer(self, tokenizer: Any) -> None:
        """
        Set the tokenizer for the dataset adapter.
        
        Args:
            tokenizer: Tokenizer instance
        """
        self.tokenizer = tokenizer
    
    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size from the tokenizer.
        
        Returns:
            Vocabulary size
        """
        if self.tokenizer is None:
            return 0
        
        if hasattr(self.tokenizer, "vocab_size"):
            return self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, "get_vocab"):
            return len(self.tokenizer.get_vocab())
        else:
            return 0
    
    def get_dataset_size(self) -> Dict[str, int]:
        """
        Get the size of each dataset split.
        
        Returns:
            Dictionary with dataset split sizes
        """
        sizes = {}
        
        if self.train_dataset is not None:
            sizes["train"] = len(self.train_dataset)
        
        if self.val_dataset is not None:
            sizes["val"] = len(self.val_dataset)
        
        if self.test_dataset is not None:
            sizes["test"] = len(self.test_dataset)
        
        return sizes
    
    def get_examples(self, split: str = "train", num_examples: int = 3) -> List[Dict[str, Any]]:
        """
        Get example data samples from a dataset split.
        
        Args:
            split: Dataset split ("train", "val", or "test")
            num_examples: Number of examples to return
            
        Returns:
            List of example data samples
        """
        dataset = None
        
        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        elif split == "test":
            dataset = self.test_dataset
        
        if dataset is None or num_examples <= 0:
            return []
        
        examples = []
        indices = torch.randperm(len(dataset))[:num_examples].tolist()
        
        for idx in indices:
            examples.append(dataset[idx])
        
        return examples