"""
Base Dataset class for QLLM.

This module provides a base dataset class that implements common functionality
needed across different dataset types, reducing code duplication and ensuring
consistent behavior.
"""

import os
import json
import pickle
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Union, Optional, Callable, Iterable
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset


logger = logging.getLogger("qllm.data")


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all QLLM datasets.
    
    This class implements common functionality needed across different dataset
    types, such as data loading, caching, preprocessing, and iteration. It
    enforces a consistent interface while allowing specialized behavior through
    abstract methods that must be implemented by subclasses.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_length: int = 512,
        tokenizer: Optional[Any] = None,
        preprocessing_fn: Optional[Callable] = None,
        transform_fn: Optional[Callable] = None,
        use_cache: bool = True,
        cache_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the base dataset.
        
        Args:
            data_path: Path to the data source (file or directory)
            cache_dir: Directory to use for caching preprocessed data
            max_length: Maximum sequence length for text data
            tokenizer: Tokenizer for text processing
            preprocessing_fn: Optional function for preprocessing raw data
            transform_fn: Optional function for transforming samples at access time
            use_cache: Whether to use caching for preprocessed data
            cache_name: Name to use for the cache file
            **kwargs: Additional dataset-specific parameters
        """
        super().__init__()
        
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.preprocessing_fn = preprocessing_fn
        self.transform_fn = transform_fn
        self.use_cache = use_cache
        
        # Generate cache name if not provided
        if cache_name is None and data_path is not None:
            base_name = os.path.basename(data_path)
            name, _ = os.path.splitext(base_name)
            self.cache_name = f"{name}_cache"
        else:
            self.cache_name = cache_name or "dataset_cache"
            
        # Store additional configuration as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize data containers
        self.data = []
        self.metadata = {"dataset_type": self.__class__.__name__}
        
        # Load data if path is provided
        if data_path is not None:
            self.load_data()
    
    @abstractmethod
    def _load_raw_data(self) -> List[Any]:
        """
        Load raw data from source.
        
        This method must be implemented by subclasses to define
        how to load data from the specific source format.
        
        Returns:
            List of raw data samples
        """
        pass
    
    @abstractmethod
    def _preprocess_sample(self, sample: Any) -> Dict[str, Any]:
        """
        Preprocess a single raw data sample.
        
        This method must be implemented by subclasses to define
        how to preprocess raw data samples into model-ready format.
        
        Args:
            sample: Raw data sample
            
        Returns:
            Preprocessed sample as a dictionary
        """
        pass
    
    def load_data(self) -> None:
        """
        Load and prepare the dataset.
        
        This method handles data loading, caching, and preprocessing.
        """
        # Check if cache exists and should be used
        if self.use_cache and self._try_load_cache():
            logger.info(f"Loaded data from cache: {self._get_cache_path()}")
            return
            
        # Load raw data
        logger.info(f"Loading data from {self.data_path}")
        raw_data = self._load_raw_data()
        
        # Preprocess data
        logger.info("Preprocessing data...")
        self.data = self._preprocess_data(raw_data)
        
        # Apply global preprocessing function if provided
        if self.preprocessing_fn is not None:
            logger.info("Applying custom preprocessing function")
            self.data = self.preprocessing_fn(self.data)
        
        # Update metadata
        self.metadata.update({
            "num_samples": len(self.data),
            "max_length": self.max_length,
        })
        
        # Cache preprocessed data
        if self.use_cache:
            self._save_cache()
    
    def _preprocess_data(self, raw_data: List[Any]) -> List[Dict[str, Any]]:
        """
        Preprocess all raw data samples.
        
        Args:
            raw_data: List of raw data samples
            
        Returns:
            List of preprocessed samples
        """
        preprocessed_data = []
        
        for sample in raw_data:
            try:
                processed = self._preprocess_sample(sample)
                preprocessed_data.append(processed)
            except Exception as e:
                logger.warning(f"Error preprocessing sample: {e}")
        
        return preprocessed_data
    
    def _get_cache_path(self) -> str:
        """
        Get the path for the cache file.
        
        Returns:
            Cache file path
        """
        if self.cache_dir is None:
            # Default to a cache directory in the same location as the data
            if self.data_path is not None:
                self.cache_dir = os.path.join(os.path.dirname(self.data_path), "cache")
            else:
                self.cache_dir = "cache"
                
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        return os.path.join(self.cache_dir, f"{self.cache_name}.pkl")
    
    def _try_load_cache(self) -> bool:
        """
        Try to load data from cache.
        
        Returns:
            True if cache was successfully loaded, False otherwise
        """
        cache_path = self._get_cache_path()
        
        if not os.path.exists(cache_path):
            return False
        
        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                
            # Extract data and metadata
            self.data = cached_data.get("data", [])
            self.metadata = cached_data.get("metadata", {})
            
            # Validate cache
            if not self.data:
                logger.warning("Cache exists but contains no data")
                return False
                
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False
    
    def _save_cache(self) -> None:
        """Save preprocessed data to cache."""
        cache_path = self._get_cache_path()
        
        try:
            # Prepare data for caching
            cache_data = {
                "data": self.data,
                "metadata": self.metadata
            }
            
            # Save to cache file
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
                
            logger.info(f"Saved preprocessed data to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Sample as a dictionary of tensors or other data
        """
        # Get the raw preprocessed sample
        sample = self.data[idx]
        
        # Apply transform if provided
        if self.transform_fn is not None:
            sample = self.transform_fn(sample)
        
        return sample
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get dataset metadata.
        
        Returns:
            Dictionary of metadata
        """
        return self.metadata
    
    def filter(self, filter_fn: Callable[[Dict[str, Any]], bool]) -> 'BaseDataset':
        """
        Create a new dataset with filtered samples.
        
        Args:
            filter_fn: Function that takes a sample and returns True if it should be kept
            
        Returns:
            New dataset with filtered samples
        """
        # Create a new dataset instance
        filtered_dataset = self.__class__(
            data_path=None,  # Don't load from path
            cache_dir=self.cache_dir,
            max_length=self.max_length,
            tokenizer=self.tokenizer,
            preprocessing_fn=self.preprocessing_fn,
            transform_fn=self.transform_fn,
            use_cache=False,  # Don't use cache for filtered dataset
        )
        
        # Filter data
        filtered_dataset.data = [sample for sample in self.data if filter_fn(sample)]
        
        # Update metadata
        filtered_dataset.metadata = self.metadata.copy()
        filtered_dataset.metadata["num_samples"] = len(filtered_dataset.data)
        filtered_dataset.metadata["filtered"] = True
        
        return filtered_dataset
    
    def split(self, split_ratio: float = 0.8, shuffle: bool = True) -> Tuple['BaseDataset', 'BaseDataset']:
        """
        Split the dataset into training and validation sets.
        
        Args:
            split_ratio: Ratio of data to use for training (0 to 1)
            shuffle: Whether to shuffle data before splitting
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Create indices
        indices = list(range(len(self)))
        
        # Shuffle if requested
        if shuffle:
            np.random.shuffle(indices)
        
        # Calculate split point
        split_idx = int(len(indices) * split_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Create datasets for each split
        train_dataset = self._create_subset(train_indices, "train")
        val_dataset = self._create_subset(val_indices, "val")
        
        return train_dataset, val_dataset
    
    def _create_subset(self, indices: List[int], subset_name: str) -> 'BaseDataset':
        """
        Create a subset of the dataset using the given indices.
        
        Args:
            indices: List of indices to include in the subset
            subset_name: Name of the subset (e.g., "train", "val")
            
        Returns:
            New dataset containing only the specified samples
        """
        # Create a new dataset instance
        subset = self.__class__(
            data_path=None,  # Don't load from path
            cache_dir=self.cache_dir,
            max_length=self.max_length,
            tokenizer=self.tokenizer,
            preprocessing_fn=self.preprocessing_fn,
            transform_fn=self.transform_fn,
            use_cache=False,  # Don't use cache for subset
        )
        
        # Select data for the subset
        subset.data = [self.data[i] for i in indices]
        
        # Update metadata
        subset.metadata = self.metadata.copy()
        subset.metadata["num_samples"] = len(subset.data)
        subset.metadata["subset"] = subset_name
        
        return subset
    
    def to_disk(self, filepath: str) -> None:
        """
        Save the dataset to disk.
        
        Args:
            filepath: Path to save the dataset to
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        try:
            # Prepare data for saving
            save_data = {
                "data": self.data,
                "metadata": self.metadata,
                "config": {
                    "max_length": self.max_length,
                    "tokenizer_info": str(self.tokenizer) if self.tokenizer else None,
                    "dataset_type": self.__class__.__name__,
                }
            }
            
            # Save to file
            with open(filepath, "wb") as f:
                pickle.dump(save_data, f)
                
            logger.info(f"Saved dataset to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
    
    @classmethod
    def from_disk(cls, filepath: str) -> 'BaseDataset':
        """
        Load dataset from disk.
        
        Args:
            filepath: Path to the saved dataset
            
        Returns:
            Loaded dataset
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file doesn't contain a valid dataset
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        try:
            with open(filepath, "rb") as f:
                saved_data = pickle.load(f)
                
            # Extract data and configuration
            data = saved_data.get("data", [])
            metadata = saved_data.get("metadata", {})
            config = saved_data.get("config", {})
            
            # Create a new dataset instance
            dataset = cls(
                data_path=None,  # Don't load from path
                max_length=config.get("max_length", 512),
                use_cache=False,  # Don't use cache
            )
            
            # Set data and metadata
            dataset.data = data
            dataset.metadata = metadata
            
            return dataset
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")