"""
Base Processor class for QLLM.

This module provides a base processor class that implements common functionality
for data transformation operations, reducing code duplication and ensuring
consistent behavior across different data processors.
"""

import os
import json
import pickle
import logging
from typing import Dict, Any, List, Tuple, Union, Optional, Callable, Iterable, Type
from abc import ABC, abstractmethod


logger = logging.getLogger("qllm.data")


class BaseProcessor(ABC):
    """
    Abstract base class for all QLLM data processors.
    
    This class implements common functionality needed across different processor
    types, such as transformation pipelines, batch processing, and serialization.
    It enforces a consistent interface while allowing specialized behavior through
    abstract methods that must be implemented by subclasses.
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        batch_size: int = 32,
        **kwargs
    ):
        """
        Initialize the base processor.
        
        Args:
            name: Name of the processor for identification
            batch_size: Batch size for batch processing
            **kwargs: Additional processor-specific parameters
        """
        self.name = name or self.__class__.__name__
        self.batch_size = batch_size
        
        # Store additional configuration as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize transformation pipeline
        self.pipeline = []
        self.metadata = {"processor_type": self.__class__.__name__}
        
        # Set up default transformations
        self._setup_pipeline()
    
    @abstractmethod
    def _setup_pipeline(self) -> None:
        """
        Set up the transformation pipeline.
        
        This method must be implemented by subclasses to define
        the specific transformations to apply.
        """
        pass
    
    @abstractmethod
    def _validate_input(self, data: Any) -> bool:
        """
        Validate input data.
        
        This method must be implemented by subclasses to define
        how to validate input data for the specific processor type.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if the input is valid, False otherwise
        """
        pass
    
    def process(self, data: Any) -> Any:
        """
        Process input data.
        
        This method applies all transformations in the pipeline to the input data.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
            
        Raises:
            ValueError: If the input data is invalid
        """
        # Validate input
        if not self._validate_input(data):
            raise ValueError(f"Invalid input data for {self.name}")
        
        # Apply each transformation in the pipeline
        result = data
        for transform_fn in self.pipeline:
            result = transform_fn(result)
            
        return result
    
    def process_batch(self, batch: List[Any]) -> List[Any]:
        """
        Process a batch of input data.
        
        This method applies the processor to each item in the batch.
        
        Args:
            batch: List of input data to process
            
        Returns:
            List of processed data
        """
        results = []
        
        for item in batch:
            try:
                processed = self.process(item)
                results.append(processed)
            except Exception as e:
                logger.warning(f"Error processing item: {e}")
                # Skip or add a placeholder depending on the specific processor
                if hasattr(self, "error_value"):
                    results.append(self.error_value)
        
        return results
    
    def add_transform(self, transform_fn: Callable[[Any], Any]) -> 'BaseProcessor':
        """
        Add a transformation to the pipeline.
        
        Args:
            transform_fn: Function that takes input data and returns transformed data
            
        Returns:
            Self for method chaining
        """
        self.pipeline.append(transform_fn)
        return self
    
    def reset_pipeline(self) -> None:
        """Reset the transformation pipeline."""
        self.pipeline = []
        self._setup_pipeline()
    
    def __call__(self, data: Any) -> Any:
        """
        Call the processor as a function.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
        """
        # Check if input is a batch or single item
        if isinstance(data, list):
            return self.process_batch(data)
        else:
            return self.process(data)
    
    def chain(self, next_processor: 'BaseProcessor') -> 'ChainedProcessor':
        """
        Chain this processor with another one.
        
        Args:
            next_processor: Next processor in the chain
            
        Returns:
            ChainedProcessor combining this processor with the next one
        """
        return ChainedProcessor(self, next_processor)
    
    def save(self, filepath: str) -> None:
        """
        Save the processor to a file.
        
        Args:
            filepath: Path to save the processor to
            
        Raises:
            IOError: If saving fails
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Extract serializable attributes
            state = {
                "name": self.name,
                "batch_size": self.batch_size,
                "metadata": self.metadata,
                "processor_type": self.__class__.__name__,
            }
            
            # Add custom serialization
            self._add_custom_serialization(state)
            
            # Save to file
            with open(filepath, "wb") as f:
                pickle.dump(state, f)
                
            logger.info(f"Saved processor to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save processor: {e}")
            raise IOError(f"Failed to save processor: {e}")
    
    def _add_custom_serialization(self, state: Dict[str, Any]) -> None:
        """
        Add custom serialization for processor-specific attributes.
        
        This method can be overridden by subclasses to add
        custom serialization logic.
        
        Args:
            state: Dictionary to add serialized attributes to
        """
        # Default implementation does nothing
        pass
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseProcessor':
        """
        Load a processor from a file.
        
        Args:
            filepath: Path to the saved processor
            
        Returns:
            Loaded processor
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file doesn't contain a valid processor
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Processor file not found: {filepath}")
        
        try:
            with open(filepath, "rb") as f:
                state = pickle.load(f)
                
            # Check if processor type matches
            processor_type = state.get("processor_type")
            if processor_type != cls.__name__:
                logger.warning(
                    f"Loaded processor type {processor_type} doesn't match expected type {cls.__name__}"
                )
            
            # Create a new processor instance
            processor = cls(
                name=state.get("name"),
                batch_size=state.get("batch_size", 32)
            )
            
            # Restore metadata
            processor.metadata = state.get("metadata", {})
            
            # Add custom deserialization
            processor._add_custom_deserialization(state)
            
            return processor
        except Exception as e:
            raise ValueError(f"Failed to load processor: {e}")
    
    def _add_custom_deserialization(self, state: Dict[str, Any]) -> None:
        """
        Add custom deserialization for processor-specific attributes.
        
        This method can be overridden by subclasses to add
        custom deserialization logic.
        
        Args:
            state: Dictionary containing serialized attributes
        """
        # Default implementation does nothing
        pass


class ChainedProcessor(BaseProcessor):
    """
    Processor that chains multiple processors together.
    
    This processor applies multiple processors in sequence.
    """
    
    def __init__(
        self,
        first_processor: BaseProcessor,
        second_processor: BaseProcessor,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the chained processor.
        
        Args:
            first_processor: First processor in the chain
            second_processor: Second processor in the chain
            name: Name of the processor (default: "ChainedProcessor")
            **kwargs: Additional parameters
        """
        self.first_processor = first_processor
        self.second_processor = second_processor
        
        # Set default name if not provided
        if name is None:
            name = f"Chain({first_processor.name}->{second_processor.name})"
            
        super().__init__(name=name, **kwargs)
    
    def _setup_pipeline(self) -> None:
        """Set up the transformation pipeline."""
        # The chained processor doesn't use the standard pipeline
        pass
    
    def _validate_input(self, data: Any) -> bool:
        """
        Validate input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if the input is valid for the first processor
        """
        return self.first_processor._validate_input(data)
    
    def process(self, data: Any) -> Any:
        """
        Process input data through both processors.
        
        Args:
            data: Input data to process
            
        Returns:
            Data processed through both processors
        """
        # Apply first processor
        intermediate = self.first_processor.process(data)
        
        # Apply second processor
        return self.second_processor.process(intermediate)
    
    def _add_custom_serialization(self, state: Dict[str, Any]) -> None:
        """
        Add custom serialization for chained processor.
        
        Args:
            state: Dictionary to add serialized attributes to
        """
        # Store processor chain information
        state["first_processor_info"] = {
            "type": self.first_processor.__class__.__name__,
            "name": self.first_processor.name
        }
        state["second_processor_info"] = {
            "type": self.second_processor.__class__.__name__,
            "name": self.second_processor.name
        }
        
        # We don't store the actual processors - they should be saved separately