"""
Standard dataset adapter for language model training.

This module implements the standard dataset adapter for language model datasets,
providing functionality for handling text data with standard language model
preprocessing and batching.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, Tuple, List, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.training.dataset_adapters.base_adapter import DatasetAdapter
from src.config.data_config import DataConfig


class StandardDatasetAdapter(DatasetAdapter):
    """
    Standard dataset adapter for language model training.
    
    This adapter handles standard language model datasets, providing
    tokenization, preprocessing, and dataloader creation for text datasets.
    """
    
    def __init__(
        self,
        config: DataConfig,
        tokenizer: Optional[Any] = None,
        max_seq_length: int = 512,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
    ):
        """
        Initialize the standard dataset adapter.
        
        Args:
            config: Data configuration
            tokenizer: Tokenizer to use for processing text
            max_seq_length: Maximum sequence length for inputs
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
            num_workers: Number of workers for dataloaders
            pin_memory: Whether to pin memory for dataloaders
            **kwargs: Additional keyword arguments
        """
        super().__init__(config, tokenizer, **kwargs)
        
        self.max_seq_length = max_seq_length
        self.train_path = train_path or getattr(config, "train_path", None)
        self.val_path = val_path or getattr(config, "val_path", None)
        self.test_path = test_path or getattr(config, "test_path", None)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.logger = logging.getLogger("quantum_resonance")
        
        # Set default collate function
        self.collate_fn = kwargs.get("collate_fn", None)
    
    def prepare_datasets(self) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
        """
        Prepare train, validation, and test datasets.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
            val_dataset and test_dataset may be None
        """
        # Check if tokenizer is available
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set before preparing datasets")
        
        # Load training dataset
        self.train_dataset = self._load_dataset(self.train_path, "train")
        
        # Load validation dataset if available
        if self.val_path:
            self.val_dataset = self._load_dataset(self.val_path, "val")
        else:
            self.val_dataset = None
        
        # Load test dataset if available
        if self.test_path:
            self.test_dataset = self._load_dataset(self.test_path, "test")
        else:
            self.test_dataset = None
        
        # Log dataset information
        dataset_sizes = self.get_dataset_size()
        self.logger.info(f"Prepared datasets - sizes: {dataset_sizes}")
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def create_dataloaders(
        self,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        **kwargs
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Create dataloaders for train, validation, and test datasets.
        
        Args:
            train_batch_size: Batch size for training
            eval_batch_size: Batch size for evaluation
            distributed: Whether to use distributed training
            world_size: Number of processes in distributed training
            rank: Process rank in distributed training
            **kwargs: Additional arguments for DataLoader
            
        Returns:
            Tuple of (train_dataloader, val_dataloader, test_dataloader)
            val_dataloader and test_dataloader may be None
        """
        # Prepare datasets if not already prepared
        if self.train_dataset is None:
            self.prepare_datasets()
        
        # Create samplers if using distributed training
        train_sampler = None
        val_sampler = None
        test_sampler = None
        
        if distributed and self.train_dataset is not None:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            
            if self.val_dataset is not None:
                val_sampler = DistributedSampler(
                    self.val_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False
                )
            
            if self.test_dataset is not None:
                test_sampler = DistributedSampler(
                    self.test_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False
                )
        
        # Create training dataloader
        train_dataloader = None
        if self.train_dataset is not None:
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=train_batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.collate_fn,
                **kwargs
            )
        
        # Create validation dataloader
        val_dataloader = None
        if self.val_dataset is not None:
            val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.collate_fn,
                **kwargs
            )
        
        # Create test dataloader
        test_dataloader = None
        if self.test_dataset is not None:
            test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                sampler=test_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.collate_fn,
                **kwargs
            )
        
        self.logger.info(f"Created dataloaders - Train batch size: {train_batch_size}, Eval batch size: {eval_batch_size}")
        
        return train_dataloader, val_dataloader, test_dataloader
    
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
        # Handle different batch formats
        if isinstance(batch, dict):
            # Batch is already a dictionary
            return batch
        elif isinstance(batch, tuple) and len(batch) == 2:
            # Assume (input_ids, labels) format
            input_ids, labels = batch
            return {
                "input_ids": input_ids,
                "labels": labels
            }
        else:
            # Try to handle as a standard language model batch
            # where inputs are the shifted labels
            if isinstance(batch, torch.Tensor):
                input_ids = batch
                # For causal language modeling, inputs are the shifted labels
                return {
                    "input_ids": input_ids[:, :-1],
                    "labels": input_ids[:, 1:]
                }
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")
    
    def _load_dataset(
        self,
        path: str,
        split: str
    ) -> Optional[Dataset]:
        """
        Load a dataset from a file path.
        
        Args:
            path: Path to dataset file or directory
            split: Dataset split ("train", "val", or "test")
            
        Returns:
            Loaded dataset or None if loading fails
        """
        if not path or not os.path.exists(path):
            self.logger.warning(f"Dataset path does not exist: {path}")
            return None
        
        try:
            # Try to determine dataset type from file extension
            if path.endswith('.txt'):
                from src.data.wikitext_dataset import WikiTextDataset
                return WikiTextDataset(
                    path,
                    self.tokenizer,
                    block_size=self.max_seq_length
                )
            elif path.endswith('.json'):
                # Try to handle as a dialogue dataset if that's available
                try:
                    from src.data.dialogue_dataset import DialogueDataset
                    return DialogueDataset(
                        path,
                        self.tokenizer,
                        max_length=self.max_seq_length
                    )
                except ImportError:
                    self.logger.warning("DialogueDataset not available. Falling back to generic handling.")
            
            # Generic fallback using custom loader
            from data.loaders.custom_loader import load_dataset
            return load_dataset(
                path,
                self.tokenizer,
                max_seq_length=self.max_seq_length,
                split=split
            )
        
        except Exception as e:
            self.logger.error(f"Error loading dataset from {path}: {e}")
            return None
    
    def set_collate_fn(self, collate_fn: Callable) -> None:
        """
        Set the collate function for dataloaders.
        
        Args:
            collate_fn: Collate function
        """
        self.collate_fn = collate_fn