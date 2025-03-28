"""
Base Loader class for QLLM.

This module provides a base loader class that implements common functionality
for loading data from various sources, reducing code duplication and ensuring
consistent behavior across different data loaders.
"""

import os
import json
import pickle
import logging
import hashlib
from typing import Dict, Any, List, Tuple, Union, Optional, Callable, Iterable, Type
from abc import ABC, abstractmethod

import torch
import numpy as np


logger = logging.getLogger("qllm.data")


class BaseLoader(ABC):
    """
    Abstract base class for all QLLM data loaders.
    
    This class implements common functionality needed across different loader
    types, such as file handling, caching, and error recovery. It enforces
    a consistent interface while allowing specialized behavior through
    abstract methods that must be implemented by subclasses.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        validate: bool = True,
        max_samples: Optional[int] = None,
        skip_bad_samples: bool = True,
        **kwargs
    ):
        """
        Initialize the base loader.
        
        Args:
            data_path: Path to the data source (file or directory)
            cache_dir: Directory to use for caching loaded data
            use_cache: Whether to use caching for loaded data
            validate: Whether to validate loaded data
            max_samples: Maximum number of samples to load (None for all)
            skip_bad_samples: Whether to skip samples that fail to load/validate
            **kwargs: Additional loader-specific parameters
        """
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.validate = validate
        self.max_samples = max_samples
        self.skip_bad_samples = skip_bad_samples
        
        # Store additional configuration as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize state
        self.is_loaded = False
        self.cached_hash = None
        self.loaded_samples = 0
        self.failed_samples = 0
        self.metadata = {"loader_type": self.__class__.__name__}
    
    @abstractmethod
    def _load_file(self, filepath: str) -> List[Any]:
        """
        Load data from a single file.
        
        This method must be implemented by subclasses to define
        how to load data from a specific file format.
        
        Args:
            filepath: Path to the file to load
            
        Returns:
            List of loaded data samples
        """
        pass
    
    @abstractmethod
    def _validate_sample(self, sample: Any) -> bool:
        """
        Validate a single data sample.
        
        This method must be implemented by subclasses to define
        how to validate samples for the specific data type.
        
        Args:
            sample: Data sample to validate
            
        Returns:
            True if the sample is valid, False otherwise
        """
        pass
    
    def load(self) -> List[Any]:
        """
        Load data from the data path.
        
        This method handles file discovery, loading, validation,
        and caching of data.
        
        Returns:
            List of loaded data samples
            
        Raises:
            FileNotFoundError: If the data path doesn't exist
            ValueError: If no valid data could be loaded
        """
        # Check if data is already loaded
        if self.is_loaded:
            return self.get_loaded_data()
        
        # Validate data path
        if self.data_path is None:
            raise ValueError("No data path provided")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        # Check if cache exists and should be used
        if self.use_cache and self._try_load_cache():
            logger.info(f"Loaded data from cache: {self._get_cache_path()}")
            self.is_loaded = True
            return self.get_loaded_data()
            
        # Load data based on whether path is file or directory
        data = []
        files_processed = 0
        
        if os.path.isfile(self.data_path):
            # Load a single file
            logger.info(f"Loading data from file: {self.data_path}")
            try:
                file_data = self._load_file(self.data_path)
                data.extend(file_data)
                files_processed += 1
            except Exception as e:
                logger.error(f"Error loading file {self.data_path}: {e}")
                raise
                
        elif os.path.isdir(self.data_path):
            # Load all files in directory
            logger.info(f"Loading data from directory: {self.data_path}")
            data_files = self._discover_files()
            
            for filepath in data_files:
                try:
                    logger.debug(f"Loading file: {filepath}")
                    file_data = self._load_file(filepath)
                    data.extend(file_data)
                    files_processed += 1
                    
                    # Check if we've reached max samples
                    if self.max_samples is not None and len(data) >= self.max_samples:
                        data = data[:self.max_samples]
                        break
                except Exception as e:
                    logger.warning(f"Error loading file {filepath}: {e}")
        
        # Validate data if requested
        if self.validate:
            logger.info("Validating loaded data...")
            data = self._validate_data(data)
        
        # Update metadata
        self.metadata.update({
            "num_samples": len(data),
            "files_processed": files_processed,
            "failed_samples": self.failed_samples
        })
        
        # Cache data if requested
        if self.use_cache:
            self._save_cache(data)
        
        # Update state
        self.loaded_samples = len(data)
        self.is_loaded = True
        
        # Store data
        self._store_loaded_data(data)
        
        return data
    
    def get_loaded_data(self) -> List[Any]:
        """
        Get the loaded data.
        
        This method must be implemented by subclasses to return
        the loaded data, which might be stored in different ways
        depending on the loader type.
        
        Returns:
            List of loaded data samples
        """
        # Default implementation; subclasses may override this
        return getattr(self, "_loaded_data", [])
    
    def _store_loaded_data(self, data: List[Any]) -> None:
        """
        Store the loaded data.
        
        Args:
            data: List of loaded data samples
        """
        # Default implementation; subclasses may override this
        self._loaded_data = data
    
    def _discover_files(self) -> List[str]:
        """
        Discover files to load in the data directory.
        
        This method can be overridden by subclasses to implement
        specific file discovery logic.
        
        Returns:
            List of file paths to load
        """
        # Default implementation: discover files with extensions that this loader supports
        supported_extensions = getattr(self, "supported_extensions", None)
        result = []
        
        for root, _, files in os.walk(self.data_path):
            for filename in files:
                filepath = os.path.join(root, filename)
                
                # Skip files that don't have a supported extension
                if supported_extensions:
                    _, ext = os.path.splitext(filename)
                    if ext.lower() not in supported_extensions:
                        continue
                        
                result.append(filepath)
        
        return result
    
    def _validate_data(self, data: List[Any]) -> List[Any]:
        """
        Validate a list of data samples.
        
        Args:
            data: List of data samples to validate
            
        Returns:
            List of valid data samples
        """
        valid_data = []
        
        for sample in data:
            try:
                if self._validate_sample(sample):
                    valid_data.append(sample)
                else:
                    self.failed_samples += 1
                    if not self.skip_bad_samples:
                        logger.warning("Invalid sample found and skip_bad_samples=False")
                        raise ValueError("Invalid sample found")
            except Exception as e:
                self.failed_samples += 1
                if not self.skip_bad_samples:
                    logger.error(f"Error validating sample: {e}")
                    raise
        
        if not valid_data:
            logger.warning("No valid data samples found")
            
        return valid_data
    
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
        
        # Create a hash based on data path and loader parameters
        data_hash = self._compute_data_hash()
        
        return os.path.join(self.cache_dir, f"{data_hash}.cache")
    
    def _compute_data_hash(self) -> str:
        """
        Compute a hash for the data source and loader parameters.
        
        Returns:
            Hash string
        """
        # Create a string representation of loader parameters
        param_str = f"path={self.data_path}|"
        param_str += f"type={self.__class__.__name__}|"
        param_str += f"max_samples={self.max_samples}|"
        
        # Add additional parameters
        for key, value in sorted(self.__dict__.items()):
            if key not in {"data_path", "cache_dir", "use_cache", "is_loaded", 
                           "loaded_samples", "failed_samples", "metadata", "cached_hash"}:
                param_str += f"{key}={value}|"
        
        # Compute hash
        hash_obj = hashlib.md5(param_str.encode())
        return hash_obj.hexdigest()
    
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
            data = cached_data.get("data", [])
            metadata = cached_data.get("metadata", {})
            cached_hash = cached_data.get("hash")
            
            # Validate cache
            if not data:
                logger.warning("Cache exists but contains no data")
                return False
            
            # Store loaded data
            self._store_loaded_data(data)
            self.metadata = metadata
            self.cached_hash = cached_hash
            self.loaded_samples = len(data)
            
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False
    
    def _save_cache(self, data: List[Any]) -> None:
        """
        Save loaded data to cache.
        
        Args:
            data: List of data samples to cache
        """
        cache_path = self._get_cache_path()
        data_hash = self._compute_data_hash()
        
        try:
            # Prepare data for caching
            cache_data = {
                "data": data,
                "metadata": self.metadata,
                "hash": data_hash
            }
            
            # Save to cache file
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
                
            logger.info(f"Saved loaded data to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get loader metadata.
        
        Returns:
            Dictionary of metadata
        """
        return self.metadata
    
    def reset(self) -> None:
        """Reset the loader state."""
        self.is_loaded = False
        self.loaded_samples = 0
        self.failed_samples = 0
        self._store_loaded_data([])
        self.metadata = {"loader_type": self.__class__.__name__}
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseLoader':
        """
        Create a loader from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Loader instance
            
        Raises:
            ValueError: If the configuration is invalid
        """
        # Extract required parameters
        data_path = config.get("data_path")
        
        # Create loader instance
        return cls(**config)