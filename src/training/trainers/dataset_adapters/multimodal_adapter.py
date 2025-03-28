"""
Multimodal dataset adapter for vision-language model training.

This module implements a dataset adapter for multimodal datasets, 
providing functionality for handling image-text pairs and other
multimodal data formats.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, Tuple, List, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.training.dataset_adapters.base_adapter import DatasetAdapter
from src.config.data_config import DataConfig


class MultimodalDatasetAdapter(DatasetAdapter):
    """
    Dataset adapter for multimodal (image-text) datasets.
    
    This adapter handles multimodal dataset operations, including
    image preprocessing, text tokenization, and combined batching for
    vision-language models.
    """
    
    def __init__(
        self,
        config: DataConfig,
        tokenizer: Optional[Any] = None,
        image_processor: Optional[Any] = None,
        max_text_length: int = 77,
        image_size: Union[int, Tuple[int, int]] = 224,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
    ):
        """
        Initialize the multimodal dataset adapter.
        
        Args:
            config: Data configuration
            tokenizer: Tokenizer to use for text processing
            image_processor: Image processor for image transformation
            max_text_length: Maximum text token length
            image_size: Target image size (single int or tuple of (height, width))
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
            num_workers: Number of workers for dataloaders
            pin_memory: Whether to pin memory for dataloaders
            **kwargs: Additional keyword arguments
        """
        super().__init__(config, tokenizer, **kwargs)
        
        self.image_processor = image_processor
        self.max_text_length = max_text_length
        
        # Handle image size parameter
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        
        self.train_path = train_path or getattr(config, "train_path", None)
        self.val_path = val_path or getattr(config, "val_path", None)
        self.test_path = test_path or getattr(config, "test_path", None)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.logger = logging.getLogger("quantum_resonance.multimodal")
        
        # Set default collate function
        self.collate_fn = kwargs.get("collate_fn", self._default_collate_fn)
    
    def prepare_datasets(self) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
        """
        Prepare train, validation, and test multimodal datasets.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
            val_dataset and test_dataset may be None
        """
        # Check if tokenizer and image processor are available
        if self.tokenizer is None:
            self.logger.warning("Tokenizer not set for multimodal dataset adapter")
        
        if self.image_processor is None:
            self.logger.warning("Image processor not set for multimodal dataset adapter")
        
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
        self.logger.info(f"Prepared multimodal datasets - sizes: {dataset_sizes}")
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def create_dataloaders(
        self,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        **kwargs
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Create dataloaders for train, validation, and test multimodal datasets.
        
        Args:
            train_batch_size: Batch size for training
            eval_batch_size: Batch size for evaluation
            **kwargs: Additional arguments for DataLoader
            
        Returns:
            Tuple of (train_dataloader, val_dataloader, test_dataloader)
            val_dataloader and test_dataloader may be None
        """
        # Prepare datasets if not already prepared
        if self.train_dataset is None:
            self.prepare_datasets()
        
        # Create training dataloader
        train_dataloader = None
        if self.train_dataset is not None:
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
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
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.collate_fn,
                **kwargs
            )
        
        self.logger.info(f"Created multimodal dataloaders - Train batch size: {train_batch_size}, Eval batch size: {eval_batch_size}")
        
        return train_dataloader, val_dataloader, test_dataloader
    
    def process_batch(
        self,
        batch: Any,
        is_train: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Process a multimodal batch from the dataloader.
        
        Args:
            batch: Batch from multimodal dataloader
            is_train: Whether the batch is for training
            
        Returns:
            Processed batch ready for model input
        """
        # Handle different multimodal batch formats
        if isinstance(batch, dict):
            # Process different multimodal batch dictionary formats
            if "pixel_values" in batch and "input_ids" in batch:
                # Standard format with pixel_values and input_ids
                return batch
            elif "images" in batch and "texts" in batch:
                # Format with raw images and texts
                return self._process_raw_batch(batch)
            elif "image" in batch and "text" in batch:
                # Singular form format
                return self._process_raw_batch({
                    "images": batch["image"],
                    "texts": batch["text"]
                })
        
        # Handle tuple format (image, text)
        elif isinstance(batch, tuple) and len(batch) == 2:
            images, texts = batch
            return self._process_raw_batch({
                "images": images,
                "texts": texts
            })
        
        # Unknown format
        self.logger.warning(f"Unknown multimodal batch format: {type(batch)}")
        return batch
    
    def _process_raw_batch(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Process a raw batch with images and texts.
        
        Args:
            batch: Batch containing raw images and texts
            
        Returns:
            Processed batch with pixel_values and input_ids
        """
        result = {}
        
        # Process images
        if "images" in batch and self.image_processor is not None:
            images = batch["images"]
            try:
                # Process images with image processor
                pixel_values = self.image_processor(images)
                
                # Handle different return types
                if isinstance(pixel_values, dict):
                    result.update(pixel_values)
                else:
                    result["pixel_values"] = pixel_values
            except Exception as e:
                self.logger.error(f"Error processing images: {e}")
        
        # Process texts
        if "texts" in batch and self.tokenizer is not None:
            texts = batch["texts"]
            try:
                # Tokenize texts
                text_inputs = self.tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_text_length,
                    return_tensors="pt"
                )
                
                # Add tokenized text to result
                for key, value in text_inputs.items():
                    result[key] = value
            except Exception as e:
                self.logger.error(f"Error processing texts: {e}")
        
        return result
    
    def _load_dataset(
        self,
        path: str,
        split: str
    ) -> Optional[Dataset]:
        """
        Load a multimodal dataset from a file path.
        
        Args:
            path: Path to dataset file or directory
            split: Dataset split ("train", "val", or "test")
            
        Returns:
            Loaded multimodal dataset or None if loading fails
        """
        if not path or not os.path.exists(path):
            self.logger.warning(f"Dataset path does not exist: {path}")
            return None
        
        try:
            # Try specialized loaders first
            
            # Check for image-text datasets in src.data if they exist
            # These are example imports that might not exist in the actual codebase
            try:
                # Try to find multimodal datasets in the project
                # First check if we have a function_calling_dataset that might handle multimodal data
                from src.data.function_calling_dataset import FunctionCallingDataset
                if "multimodal" in path.lower() or "image" in path.lower():
                    return FunctionCallingDataset(
                        path,
                        self.tokenizer,
                        image_processor=self.image_processor,
                        max_length=self.max_text_length,
                        image_size=self.image_size[0],
                        split=split
                    )
            except ImportError:
                pass
            
            # Generic fallback
            self.logger.warning(f"No specialized multimodal dataset loader found for {path}")
            self.logger.warning("Attempting to use custom_loader as fallback")
            
            # Try to fall back to custom loader
            try:
                from data.loaders.custom_loader import load_dataset
                return load_dataset(
                    path,
                    self.tokenizer,
                    image_processor=self.image_processor,
                    max_seq_length=self.max_text_length,
                    image_size=self.image_size,
                    split=split,
                    is_multimodal=True
                )
            except Exception as e:
                self.logger.error(f"Error loading with custom_loader: {e}")
            
            # If all else fails, raise an error
            raise ValueError(f"No suitable multimodal dataset loader found for {path}")
        
        except Exception as e:
            self.logger.error(f"Error loading multimodal dataset from {path}: {e}")
            return None
    
    def set_image_processor(self, image_processor: Any) -> None:
        """
        Set the image processor for the dataset adapter.
        
        Args:
            image_processor: Image processor instance
        """
        self.image_processor = image_processor
    
    def _default_collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Default collate function for multimodal batches.
        
        Args:
            batch: List of samples
            
        Returns:
            Collated batch
        """
        if not batch:
            return {}
        
        result = {}
        
        # Process each key in the batch
        for key in batch[0].keys():
            values = [sample[key] for sample in batch if key in sample]
            
            if not values:
                continue
            
            if isinstance(values[0], torch.Tensor):
                # Handle tensor values
                if key == "pixel_values" or key.endswith("_pixel_values"):
                    # For images, we need to ensure they have the same shape
                    # or use a proper batching approach
                    try:
                        result[key] = torch.stack(values)
                    except Exception:
                        # Fall back to padding if shapes differ
                        max_h = max(img.shape[-2] for img in values)
                        max_w = max(img.shape[-1] for img in values)
                        
                        # Pad and stack
                        padded = []
                        for img in values:
                            pad_h = max_h - img.shape[-2]
                            pad_w = max_w - img.shape[-1]
                            padded_img = F.pad(img, (0, pad_w, 0, pad_h))
                            padded.append(padded_img)
                        
                        result[key] = torch.stack(padded)
                else:
                    # For other tensors, try simple stacking
                    result[key] = torch.stack(values)
            elif isinstance(values[0], (str, int, float, bool)):
                # Simple types can be returned as lists
                result[key] = values
            elif values[0] is None:
                # Skip None values
                continue
            else:
                # For other types, return as list and log a warning
                result[key] = values
                self.logger.warning(f"Unsupported batch type for key {key}: {type(values[0])}")
        
        return result
    
    def get_examples(
        self,
        split: str = "train",
        num_examples: int = 1,
        return_tensors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get example multimodal data from a dataset split.
        
        Args:
            split: Dataset split ("train", "val", or "test")
            num_examples: Number of examples to return
            return_tensors: Whether to return tensor data (False returns descriptions)
            
        Returns:
            List of example multimodal data
        """
        examples = super().get_examples(split, num_examples)
        
        if return_tensors or not examples:
            return examples
        
        # Format multimodal examples for readability
        formatted_examples = []
        
        for example in examples:
            formatted_example = {}
            
            # Format image data
            if "pixel_values" in example:
                image_tensor = example["pixel_values"]
                formatted_example["image"] = {
                    "shape": tuple(image_tensor.shape),
                    "type": str(image_tensor.dtype),
                    "min": float(image_tensor.min()),
                    "max": float(image_tensor.max()),
                    "mean": float(image_tensor.float().mean()),
                }
            
            # Format text data
            if "input_ids" in example and self.tokenizer is not None:
                input_text = self.tokenizer.decode(example["input_ids"])
                formatted_example["text"] = input_text
            
            # Add other metadata
            for key, value in example.items():
                if key not in ["pixel_values", "input_ids", "attention_mask", "labels"]:
                    formatted_example[key] = value
            
            formatted_examples.append(formatted_example)
        
        return formatted_examples