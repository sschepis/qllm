"""
Text Processor for QLLM.

This module provides a processor for textual data, extending the
BaseProcessor with functionality specific to text preprocessing
and transformation.
"""

import re
import string
import logging
from typing import Dict, Any, List, Tuple, Union, Optional, Callable, Set

from src.data.base import BaseProcessor


logger = logging.getLogger("qllm.data")


class TextProcessor(BaseProcessor):
    """
    Processor for text data.
    
    This processor implements common text preprocessing operations
    such as lowercasing, punctuation removal, whitespace normalization,
    and more.
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        batch_size: int = 32,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        normalize_whitespace: bool = False,
        remove_html: bool = False,
        expand_contractions: bool = False,
        remove_stopwords: bool = False,
        stopwords: Optional[Set[str]] = None,
        stemming: bool = False,
        lemmatization: bool = False,
        custom_transformations: Optional[List[Callable[[str], str]]] = None,
        input_key: str = "text",
        output_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the text processor.
        
        Args:
            name: Name of the processor for identification
            batch_size: Batch size for batch processing
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            normalize_whitespace: Whether to normalize whitespace
            remove_html: Whether to remove HTML tags
            expand_contractions: Whether to expand contractions
            remove_stopwords: Whether to remove stop words
            stopwords: Set of stop words to remove
            stemming: Whether to apply stemming
            lemmatization: Whether to apply lemmatization
            custom_transformations: List of custom transformation functions
            input_key: Key for input text in data dictionaries
            output_key: Key for output text in data dictionaries (defaults to input_key)
            **kwargs: Additional processor-specific parameters
        """
        # Store text processing options
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.normalize_whitespace = normalize_whitespace
        self.remove_html = remove_html
        self.expand_contractions = expand_contractions
        self.remove_stopwords = remove_stopwords
        self.stopwords = stopwords or set()
        self.stemming = stemming
        self.lemmatization = lemmatization
        self.custom_transformations = custom_transformations or []
        self.input_key = input_key
        self.output_key = output_key or input_key
        
        # Load natural language processing components if needed
        self._load_nlp_components()
        
        # Initialize base processor
        super().__init__(
            name=name or "TextProcessor",
            batch_size=batch_size,
            **kwargs
        )
        
        # Prepare error value
        self.error_value = ""
    
    def _load_nlp_components(self) -> None:
        """Load required NLP components."""
        # Load stemmer if needed
        if self.stemming:
            try:
                import nltk
                from nltk.stem import PorterStemmer
                try:
                    # Ensure the required resources are downloaded
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt')
                self.stemmer = PorterStemmer()
            except ImportError:
                logger.warning("NLTK is required for stemming but not installed. Stemming will be disabled.")
                self.stemming = False
        
        # Load lemmatizer if needed
        if self.lemmatization:
            try:
                import nltk
                from nltk.stem import WordNetLemmatizer
                try:
                    # Ensure the required resources are downloaded
                    nltk.data.find('corpora/wordnet')
                except LookupError:
                    nltk.download('wordnet')
                self.lemmatizer = WordNetLemmatizer()
            except ImportError:
                logger.warning("NLTK is required for lemmatization but not installed. Lemmatization will be disabled.")
                self.lemmatization = False
        
        # Load stopwords if needed
        if self.remove_stopwords and not self.stopwords:
            try:
                import nltk
                try:
                    # Ensure the required resources are downloaded
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords')
                from nltk.corpus import stopwords
                self.stopwords = set(stopwords.words('english'))
            except ImportError:
                logger.warning("NLTK is required for stopwords but not installed. Stopword removal will be disabled.")
                self.remove_stopwords = False
    
    def _setup_pipeline(self) -> None:
        """Set up the text transformation pipeline."""
        # Add each transformation to the pipeline in the appropriate order
        
        # HTML removal should come first
        if self.remove_html:
            self.pipeline.append(self._remove_html_tags)
        
        # Expand contractions before other processing
        if self.expand_contractions:
            self.pipeline.append(self._expand_contractions)
        
        # Case normalization
        if self.lowercase:
            self.pipeline.append(self._lowercase)
        
        # Punctuation removal
        if self.remove_punctuation:
            self.pipeline.append(self._remove_punctuation)
        
        # Whitespace normalization
        if self.normalize_whitespace:
            self.pipeline.append(self._normalize_whitespace)
        
        # Stopword removal
        if self.remove_stopwords:
            self.pipeline.append(self._remove_stopwords)
        
        # Stemming/lemmatization (mutually exclusive)
        if self.stemming:
            self.pipeline.append(self._stem_text)
        elif self.lemmatization:
            self.pipeline.append(self._lemmatize_text)
        
        # Add custom transformations
        for transform_fn in self.custom_transformations:
            self.pipeline.append(transform_fn)
    
    def _validate_input(self, data: Any) -> bool:
        """
        Validate input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if the input is valid, False otherwise
        """
        # If input is a string, it's valid
        if isinstance(data, str):
            return True
        
        # If input is a dictionary, check for input key
        if isinstance(data, dict) and self.input_key in data:
            # Ensure the value is a string
            return isinstance(data[self.input_key], str)
        
        return False
    
    def process(self, data: Any) -> Any:
        """
        Process input data.
        
        This method applies all transformations in the pipeline to the input text.
        
        Args:
            data: Input data to process (string or dictionary)
            
        Returns:
            Processed data (string or dictionary)
            
        Raises:
            ValueError: If the input data is invalid
        """
        # Validate input
        if not self._validate_input(data):
            raise ValueError(f"Invalid input data for {self.name}")
        
        # Extract text from input
        if isinstance(data, str):
            text = data
            is_dict = False
        else:
            text = data[self.input_key]
            is_dict = True
        
        # Apply transformations
        for transform_fn in self.pipeline:
            text = transform_fn(text)
        
        # Return result in the same format as input
        if is_dict:
            result = data.copy()
            result[self.output_key] = text
            return result
        else:
            return text
    
    # Text transformation methods
    
    def _lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple whitespace characters with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        return text.strip()
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        # Simple regex to remove HTML tags
        return re.sub(r'<[^>]+>', '', text)
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions in text."""
        # Dictionary of common contractions
        contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'll": "he will",
            "he's": "he is",
            "i'd": "I would",
            "i'll": "I will",
            "i'm": "I am",
            "i've": "I have",
            "isn't": "is not",
            "it's": "it is",
            "let's": "let us",
            "mightn't": "might not",
            "mustn't": "must not",
            "shan't": "shall not",
            "she'd": "she would",
            "she'll": "she will",
            "she's": "she is",
            "shouldn't": "should not",
            "that's": "that is",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "we'd": "we would",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "where's": "where is",
            "who'd": "who would",
            "who'll": "who will",
            "who're": "who are",
            "who's": "who is",
            "who've": "who have",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have"
        }
        
        # Replace contractions
        for contraction, expansion in contractions.items():
            text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove stop words from text."""
        if not self.stopwords:
            return text
        
        # Split text into words
        words = text.split()
        
        # Filter out stop words
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        
        # Join words back into text
        return ' '.join(filtered_words)
    
    def _stem_text(self, text: str) -> str:
        """Apply stemming to text."""
        if not hasattr(self, 'stemmer'):
            return text
        
        # Split text into words
        words = text.split()
        
        # Apply stemming to each word
        stemmed_words = [self.stemmer.stem(word) for word in words]
        
        # Join words back into text
        return ' '.join(stemmed_words)
    
    def _lemmatize_text(self, text: str) -> str:
        """Apply lemmatization to text."""
        if not hasattr(self, 'lemmatizer'):
            return text
        
        # Split text into words
        words = text.split()
        
        # Apply lemmatization to each word
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        
        # Join words back into text
        return ' '.join(lemmatized_words)
    
    def _add_custom_serialization(self, state: Dict[str, Any]) -> None:
        """Add custom serialization for processor-specific attributes."""
        # Save processor configuration
        state["processor_config"] = {
            "lowercase": self.lowercase,
            "remove_punctuation": self.remove_punctuation,
            "normalize_whitespace": self.normalize_whitespace,
            "remove_html": self.remove_html,
            "expand_contractions": self.expand_contractions,
            "remove_stopwords": self.remove_stopwords,
            "stemming": self.stemming,
            "lemmatization": self.lemmatization,
            "input_key": self.input_key,
            "output_key": self.output_key
        }
        
        # Save stopwords
        if self.stopwords:
            state["stopwords"] = list(self.stopwords)
    
    def _add_custom_deserialization(self, state: Dict[str, Any]) -> None:
        """Add custom deserialization for processor-specific attributes."""
        # Load processor configuration
        processor_config = state.get("processor_config", {})
        for key, value in processor_config.items():
            setattr(self, key, value)
        
        # Load stopwords
        if "stopwords" in state:
            self.stopwords = set(state["stopwords"])
        
        # Load NLP components if needed
        self._load_nlp_components()
        
        # Reset pipeline with loaded configuration
        self.reset_pipeline()
    
    @classmethod
    def create_standard_processor(cls, processor_type: str = "basic") -> 'TextProcessor':
        """
        Create a standard text processor with predefined configurations.
        
        Args:
            processor_type: Type of processor to create:
                - 'basic': Lowercase and normalize whitespace
                - 'clean': Basic + remove punctuation and HTML
                - 'nlp': Clean + expand contractions and remove stopwords
                - 'stem': NLP + stemming
                - 'lemma': NLP + lemmatization
                
        Returns:
            Configured TextProcessor
        """
        if processor_type == "basic":
            return cls(
                lowercase=True,
                normalize_whitespace=True
            )
        elif processor_type == "clean":
            return cls(
                lowercase=True,
                normalize_whitespace=True,
                remove_punctuation=True,
                remove_html=True
            )
        elif processor_type == "nlp":
            return cls(
                lowercase=True,
                normalize_whitespace=True,
                remove_punctuation=True,
                remove_html=True,
                expand_contractions=True,
                remove_stopwords=True
            )
        elif processor_type == "stem":
            return cls(
                lowercase=True,
                normalize_whitespace=True,
                remove_punctuation=True,
                remove_html=True,
                expand_contractions=True,
                remove_stopwords=True,
                stemming=True
            )
        elif processor_type == "lemma":
            return cls(
                lowercase=True,
                normalize_whitespace=True,
                remove_punctuation=True,
                remove_html=True,
                expand_contractions=True,
                remove_stopwords=True,
                lemmatization=True
            )
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")