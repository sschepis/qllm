"""
WikiText Loader for QLLM.

This module provides a loader for WikiText data, extending the
BaseLoader with functionality specific to loading and processing WikiText.
"""

import os
import re
import logging
from typing import Dict, Any, List, Tuple, Union, Optional, Callable

from src.data.base import BaseLoader


logger = logging.getLogger("qllm.data")


class WikitextLoader(BaseLoader):
    """
    Loader for WikiText data.
    
    This loader implements functionality for loading and processing
    WikiText data, which is a collection of Wikipedia articles
    commonly used for language modeling tasks.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        validate: bool = True,
        max_samples: Optional[int] = None,
        skip_bad_samples: bool = True,
        min_length: int = 10,
        clean_whitespace: bool = True,
        remove_citations: bool = True,
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 0,
        **kwargs
    ):
        """
        Initialize the WikiText loader.
        
        Args:
            data_path: Path to the WikiText data file or directory
            cache_dir: Directory to use for caching loaded data
            use_cache: Whether to use caching for loaded data
            validate: Whether to validate loaded data
            max_samples: Maximum number of samples to load (None for all)
            skip_bad_samples: Whether to skip samples that fail to load/validate
            min_length: Minimum length of text to be considered valid
            clean_whitespace: Whether to clean excessive whitespace
            remove_citations: Whether to remove citation brackets
            chunk_size: Size of text chunks (None to keep articles whole)
            chunk_overlap: Overlap between chunks
            **kwargs: Additional loader-specific parameters
        """
        # WikiText-specific parameters
        self.min_length = min_length
        self.clean_whitespace = clean_whitespace
        self.remove_citations = remove_citations
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Define supported file extensions
        self.supported_extensions = [".txt", ".tokens", ".raw"]
        
        # Initialize base loader
        super().__init__(
            data_path=data_path,
            cache_dir=cache_dir,
            use_cache=use_cache,
            validate=validate,
            max_samples=max_samples,
            skip_bad_samples=skip_bad_samples,
            **kwargs
        )
    
    def _load_file(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load data from a single WikiText file.
        
        Args:
            filepath: Path to the file to load
            
        Returns:
            List of loaded data samples
        """
        samples = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # WikiText files contain Wikipedia articles with headers and paragraphs
                current_article = None
                current_text = []
                
                for line in f:
                    # Process the line
                    line = line.rstrip('\n')
                    
                    # Handle article headers (= Title =)
                    if line.startswith('= ') and line.endswith(' ='):
                        # If we have accumulated text from a previous article, save it
                        if current_article is not None and current_text:
                            article_text = '\n'.join(current_text)
                            article_samples = self._process_article(
                                current_article, article_text)
                            samples.extend(article_samples)
                        
                        # Start a new article
                        current_article = line.strip('= ')
                        current_text = []
                    else:
                        # Add line to current article text
                        current_text.append(line)
                
                # Don't forget the last article
                if current_article is not None and current_text:
                    article_text = '\n'.join(current_text)
                    article_samples = self._process_article(
                        current_article, article_text)
                    samples.extend(article_samples)
        
        except Exception as e:
            logger.error(f"Error loading WikiText file {filepath}: {e}")
            if not self.skip_bad_samples:
                raise
        
        return samples
    
    def _process_article(self, title: str, text: str) -> List[Dict[str, Any]]:
        """
        Process a single WikiText article.
        
        Args:
            title: Article title
            text: Article text
            
        Returns:
            List of processed samples from the article
        """
        # Clean the text if requested
        if self.clean_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove citation brackets if requested
        if self.remove_citations:
            text = re.sub(r'\[\d+\]', '', text)
        
        # If chunk size is not specified, return the whole article as one sample
        if self.chunk_size is None:
            return [{
                'title': title,
                'text': text,
                'source': 'wikitext'
            }]
        
        # Otherwise, split into chunks
        chunks = []
        
        # Split text into tokens (simple word-based tokenization)
        tokens = text.split()
        
        # Skip if too short
        if len(tokens) < self.min_length:
            return []
        
        # Create chunks with overlap
        for i in range(0, len(tokens) - self.min_length, 
                       self.chunk_size - self.chunk_overlap):
            # Get chunk tokens
            end_idx = min(i + self.chunk_size, len(tokens))
            chunk_tokens = tokens[i:end_idx]
            
            # Convert back to text
            chunk_text = ' '.join(chunk_tokens)
            
            # Add to chunks
            chunks.append({
                'title': title,
                'text': chunk_text,
                'source': 'wikitext',
                'chunk_id': len(chunks),
                'total_chunks': (len(tokens) - self.chunk_overlap) // 
                                (self.chunk_size - self.chunk_overlap) + 1
            })
            
            # Stop if we've reached the end
            if end_idx >= len(tokens):
                break
        
        return chunks
    
    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate a single data sample.
        
        Args:
            sample: Data sample to validate
            
        Returns:
            True if the sample is valid, False otherwise
        """
        # Check if sample has required fields
        if not isinstance(sample, dict):
            return False
        
        if 'text' not in sample:
            return False
        
        # Check text length
        text = sample['text']
        if not isinstance(text, str) or len(text.split()) < self.min_length:
            return False
        
        return True
    
    def get_loaded_data(self) -> List[Dict[str, Any]]:
        """
        Get the loaded data.
        
        Returns:
            List of loaded data samples
        """
        return getattr(self, "_loaded_data", [])
    
    @classmethod
    def from_directory(
        cls, 
        directory_path: str, 
        **kwargs
    ) -> 'WikitextLoader':
        """
        Create a WikitextLoader from a directory of text files.
        
        Args:
            directory_path: Path to directory containing WikiText files
            **kwargs: Additional parameters for the loader
            
        Returns:
            WikitextLoader instance
            
        Raises:
            FileNotFoundError: If the directory doesn't exist
        """
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        return cls(data_path=directory_path, **kwargs)
    
    @classmethod
    def from_standard_dataset(
        cls,
        dataset_name: str = "wikitext-103",
        subset: str = "train",
        cache_dir: Optional[str] = None,
        **kwargs
    ) -> 'WikitextLoader':
        """
        Create a WikitextLoader for a standard WikiText dataset.
        
        Args:
            dataset_name: Name of the dataset ('wikitext-2', 'wikitext-103')
            subset: Subset to load ('train', 'valid', 'test')
            cache_dir: Directory to use for caching
            **kwargs: Additional parameters for the loader
            
        Returns:
            WikitextLoader instance
        """
        # Validate dataset name and subset
        valid_names = ["wikitext-2", "wikitext-103"]
        valid_subsets = ["train", "valid", "test"]
        
        if dataset_name not in valid_names:
            raise ValueError(f"Invalid dataset name: {dataset_name}. Expected one of {valid_names}")
        
        if subset not in valid_subsets:
            raise ValueError(f"Invalid subset: {subset}. Expected one of {valid_subsets}")
        
        # Try to download the dataset if it's not already available
        try:
            from datasets import load_dataset
            
            # Load the dataset using Hugging Face datasets
            dataset = load_dataset(
                "wikitext", 
                name=dataset_name.replace("wikitext-", ""),
                split=subset,
                cache_dir=cache_dir
            )
            
            # Extract texts and save to temporary file
            import tempfile
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, f"{dataset_name}-{subset}.txt")
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(item['text'] + '\n')
            
            # Create loader from the temporary file
            loader = cls(data_path=temp_file, cache_dir=cache_dir, **kwargs)
            
            # Set cleanup function to remove temporary files
            def cleanup():
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
            
            # Store cleanup function
            loader._cleanup = cleanup
            
            return loader
        
        except ImportError:
            # Datasets library not available
            logger.warning("Hugging Face datasets library not available. "
                          "Cannot download standard WikiText dataset.")
            raise ImportError("Hugging Face datasets library is required to download "
                             "standard WikiText datasets. Please install it with "
                             "pip install datasets")