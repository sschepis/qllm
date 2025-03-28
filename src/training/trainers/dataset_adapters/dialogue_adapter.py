"""
Dialogue dataset adapter for conversational language model training.

This module implements a dataset adapter for dialogue datasets, providing
functionality for handling conversational data with appropriate formatting
and processing for dialogue-focused language models.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, Tuple, List, Callable

import torch
from torch.utils.data import Dataset, DataLoader

from src.training.dataset_adapters.base_adapter import DatasetAdapter
from src.training.dataset_adapters.standard_adapter import StandardDatasetAdapter
from src.config.data_config import DataConfig


class DialogueDatasetAdapter(StandardDatasetAdapter):
    """
    Dataset adapter for dialogue/conversational datasets.
    
    This adapter handles dialogue-specific dataset operations, including
    conversation formatting, speaker tracking, and specialized processing
    for conversational language models.
    """
    
    def __init__(
        self,
        config: DataConfig,
        tokenizer: Optional[Any] = None,
        max_seq_length: int = 1024,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        conversation_template: Optional[str] = None,
        user_token: str = "<user>",
        assistant_token: str = "<assistant>",
        system_token: str = "<system>",
        add_special_tokens: bool = True,
        **kwargs
    ):
        """
        Initialize the dialogue dataset adapter.
        
        Args:
            config: Data configuration
            tokenizer: Tokenizer to use for processing text
            max_seq_length: Maximum sequence length for inputs
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
            conversation_template: Template format for conversations
            user_token: Token to indicate user messages
            assistant_token: Token to indicate assistant messages
            system_token: Token to indicate system messages
            add_special_tokens: Whether to add special tokens to the tokenizer
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            config,
            tokenizer,
            max_seq_length,
            train_path,
            val_path,
            test_path,
            **kwargs
        )
        
        self.conversation_template = conversation_template
        self.user_token = user_token
        self.assistant_token = assistant_token
        self.system_token = system_token
        self.add_special_tokens = add_special_tokens
        
        # Try to extract format from config if not provided
        if self.conversation_template is None:
            self.conversation_template = getattr(config, "conversation_template", "default")
        
        self.logger = logging.getLogger("quantum_resonance.dialogue")
    
    def prepare_datasets(self) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
        """
        Prepare train, validation, and test dialogue datasets.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
            val_dataset and test_dataset may be None
        """
        # Add special tokens if requested
        if self.add_special_tokens and self.tokenizer is not None:
            self._add_dialogue_tokens()
        
        # Load datasets using parent method
        return super().prepare_datasets()
    
    def process_batch(
        self,
        batch: Any,
        is_train: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Process a dialogue batch from the dataloader.
        
        Args:
            batch: Batch from dialogue dataloader
            is_train: Whether the batch is for training
            
        Returns:
            Processed batch ready for model input
        """
        # Handle different dialogue batch formats
        if isinstance(batch, dict):
            # Handle common dialogue batch formats
            if "input_ids" in batch and "labels" in batch:
                # Standard format with input_ids and labels
                return batch
            elif "input_ids" in batch and "attention_mask" in batch:
                # Format with attention mask but no labels
                # For dialogue models, we often use input_ids as labels
                result = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                }
                
                # Add labels if they don't exist
                if "labels" not in batch:
                    result["labels"] = batch["input_ids"].clone()
                else:
                    result["labels"] = batch["labels"]
                
                return result
            elif "conversation" in batch:
                # Batch contains raw conversation data
                return self._format_conversations(batch["conversation"])
        
        # Fall back to standard processing
        return super().process_batch(batch, is_train)
    
    def _load_dataset(
        self,
        path: str,
        split: str
    ) -> Optional[Dataset]:
        """
        Load a dialogue dataset from a file path.
        
        Args:
            path: Path to dataset file or directory
            split: Dataset split ("train", "val", or "test")
            
        Returns:
            Loaded dialogue dataset or None if loading fails
        """
        if not path or not os.path.exists(path):
            self.logger.warning(f"Dataset path does not exist: {path}")
            return None
        
        try:
            # Try to load as a dialogue dataset first
            try:
                from src.data.dialogue_dataset import DialogueDataset
                return DialogueDataset(
                    path,
                    self.tokenizer,
                    max_length=self.max_seq_length,
                    user_token=self.user_token,
                    assistant_token=self.assistant_token,
                    system_token=self.system_token
                )
            except ImportError:
                self.logger.warning("DialogueDataset not available, trying alternative loaders.")
            
            # Try daily dialog loader
            try:
                from data.loaders.daily_dialog_loader import DailyDialogDataset
                if os.path.basename(path).startswith("daily_dialog"):
                    return DailyDialogDataset(
                        path,
                        self.tokenizer,
                        max_length=self.max_seq_length
                    )
            except ImportError:
                pass
            
            # Fall back to custom loader
            from data.loaders.custom_loader import load_dataset
            return load_dataset(
                path,
                self.tokenizer,
                max_seq_length=self.max_seq_length,
                split=split,
                is_dialogue=True
            )
        
        except Exception as e:
            self.logger.error(f"Error loading dialogue dataset from {path}: {e}")
            return None
    
    def _add_dialogue_tokens(self) -> None:
        """
        Add dialogue-specific special tokens to the tokenizer.
        """
        if self.tokenizer is None:
            return
        
        # Define special tokens to add
        special_tokens = {
            "additional_special_tokens": []
        }
        
        # Add role tokens if they don't exist
        for token in [self.user_token, self.assistant_token, self.system_token]:
            if token not in special_tokens["additional_special_tokens"]:
                special_tokens["additional_special_tokens"].append(token)
        
        # Add special tokens to tokenizer
        try:
            num_added = self.tokenizer.add_special_tokens(special_tokens)
            if num_added > 0:
                self.logger.info(f"Added {num_added} dialogue special tokens to tokenizer")
        except Exception as e:
            self.logger.warning(f"Failed to add dialogue special tokens: {e}")
    
    def _format_conversations(
        self,
        conversations: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Format conversation data into model inputs.
        
        Args:
            conversations: List of conversation dictionaries
            
        Returns:
            Formatted input batch
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set to format conversations")
        
        # Apply conversation formatting based on template
        formatted_inputs = []
        
        for conv in conversations:
            if self.conversation_template == "default":
                # Default format: "<system> {system} <user> {user} <assistant> {assistant}"
                formatted_text = ""
                
                # Add system message if present
                if "system" in conv and conv["system"]:
                    formatted_text += f"{self.system_token} {conv['system']} "
                
                # Add conversation turns
                for turn in conv.get("turns", []):
                    if turn.get("role") == "user":
                        formatted_text += f"{self.user_token} {turn.get('content', '')} "
                    elif turn.get("role") == "assistant":
                        formatted_text += f"{self.assistant_token} {turn.get('content', '')} "
                
                formatted_inputs.append(formatted_text)
            
            elif self.conversation_template == "chatml":
                # ChatML format
                formatted_text = ""
                
                # Add system message if present
                if "system" in conv and conv["system"]:
                    formatted_text += f"<|im_start|>system\n{conv['system']}<|im_end|>\n"
                
                # Add conversation turns
                for turn in conv.get("turns", []):
                    if turn.get("role") == "user":
                        formatted_text += f"<|im_start|>user\n{turn.get('content', '')}<|im_end|>\n"
                    elif turn.get("role") == "assistant":
                        formatted_text += f"<|im_start|>assistant\n{turn.get('content', '')}<|im_end|>\n"
                
                formatted_inputs.append(formatted_text)
            
            else:
                # Custom handling for other templates can be added here
                self.logger.warning(f"Unknown conversation template: {self.conversation_template}")
                # Fall back to default format
                formatted_text = ""
                for turn in conv.get("turns", []):
                    formatted_text += f"{turn.get('role', 'user')}: {turn.get('content', '')}\n"
                
                formatted_inputs.append(formatted_text)
        
        # Tokenize formatted conversations
        encoded_inputs = self.tokenizer(
            formatted_inputs,
            padding="longest",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        # For causal language modeling, create labels from input_ids
        encoded_inputs["labels"] = encoded_inputs["input_ids"].clone()
        
        return encoded_inputs
    
    def get_examples(
        self,
        split: str = "train",
        num_examples: int = 1,
        format_output: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get example dialogue data from a dataset split.
        
        Args:
            split: Dataset split ("train", "val", or "test")
            num_examples: Number of examples to return
            format_output: Whether to format conversations for readability
            
        Returns:
            List of example dialogue data
        """
        examples = super().get_examples(split, num_examples)
        
        if not format_output or not examples:
            return examples
        
        # Format dialogue examples for readability
        formatted_examples = []
        
        for example in examples:
            formatted_example = {}
            
            # Try to decode input_ids if present
            if "input_ids" in example and self.tokenizer is not None:
                input_text = self.tokenizer.decode(example["input_ids"])
                formatted_example["input_text"] = input_text
            
            # Add original example data
            for key, value in example.items():
                if key != "input_ids" and key != "labels" and key != "attention_mask":
                    formatted_example[key] = value
            
            formatted_examples.append(formatted_example)
            
            return formatted_examples
        
        def get_train_dataloader(self) -> Optional[DataLoader]:
            """
            Get the training dataloader.
            
            Returns:
                Training dataloader or None if not available
            """
            # Create dataloaders if we don't have them already
            if not hasattr(self, '_train_dataloader') or self._train_dataloader is None:
                train_batch_size = getattr(self.config, "batch_size", 8)
                eval_batch_size = getattr(self.config, "eval_batch_size", train_batch_size)
                
                self._train_dataloader, self._val_dataloader, self._test_dataloader = self.create_dataloaders(
                    train_batch_size=train_batch_size,
                    eval_batch_size=eval_batch_size
                )
            
            return self._train_dataloader
        
        def get_val_dataloader(self) -> Optional[DataLoader]:
            """
            Get the validation dataloader.
            
            Returns:
                Validation dataloader or None if not available
            """
            # Make sure we've initialized the dataloaders
            if not hasattr(self, '_val_dataloader') or self._val_dataloader is None:
                self.get_train_dataloader()
                
            return self._val_dataloader
        
        def prepare_batch(self, batch: Dict[str, Any], is_train: bool = True) -> Dict[str, torch.Tensor]:
            """
            Prepare a batch for the model, ensuring all tensors have the correct type.
            
            Args:
                batch: Input batch to prepare
                is_train: Whether the batch is for training
                
            Returns:
                Prepared batch with correct tensor types
            """
            # First process the batch with the standard method
            processed_batch = self.process_batch(batch, is_train)
            
            # Ensure input_ids and labels are integers, not floats
            for key in ['input_ids', 'labels']:
                if key in processed_batch and isinstance(processed_batch[key], torch.Tensor):
                    if processed_batch[key].dtype != torch.long:
                        # Convert to LongTensor for embedding layers
                        self.logger.info(f"Converting {key} from {processed_batch[key].dtype} to torch.long")
                        processed_batch[key] = processed_batch[key].long()
            
            return processed_batch