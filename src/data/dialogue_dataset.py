"""
Dialogue Dataset processing module.

This module provides utilities for loading, preprocessing, and training on
dialogue datasets for conversation-based language modeling.
"""

import os
import json
import torch
import random
from typing import Dict, List, Optional, Union, Any, Tuple
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class DialogueDataset(Dataset):
    """
    Dataset class for dialogue and conversation data.
    
    This class handles various dialogue formats, including:
    - Standard dialogue datasets (DailyDialog, ConvAI2, etc.)
    - Custom conversation data
    - User feedback and interactions
    """
    
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 data_path: Optional[str] = None,
                 dataset_name: Optional[str] = None,
                 split: str = "train",
                 max_length: int = 1024,
                 speaker_tokens: Optional[Dict[str, str]] = None,
                 system_prompt: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 return_tensors: bool = True,
                 include_history: bool = True,
                 history_length: int = 5,
                 learning_samples: Optional[List[Dict]] = None):
        """
        Initialize the DialogueDataset.
        
        Args:
            tokenizer: Tokenizer to use for preprocessing
            data_path: Path to custom dialogue data (JSON format)
            dataset_name: Name of the HF dataset to load (e.g., "daily_dialog")
            split: Dataset split (train, validation, test)
            max_length: Maximum sequence length
            speaker_tokens: Dictionary mapping speaker roles to tokens
            system_prompt: Optional system prompt to include at the start of conversations
            cache_dir: Directory for caching the dataset
            return_tensors: Whether to return PyTorch tensors
            include_history: Whether to include conversation history
            history_length: Maximum number of turns to include in history
            learning_samples: Optional additional samples for continuous learning
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
        self.include_history = include_history
        self.history_length = history_length
        
        # Set up speaker tokens
        self.speaker_tokens = speaker_tokens or {
            "system": "<|system|>",
            "user": "<|user|>",
            "assistant": "<|assistant|>",
            "end": "<|end|>"
        }
        
        # Add special tokens to tokenizer if they don't exist
        self._add_special_tokens()
        
        # Set system prompt
        self.system_prompt = system_prompt
        
        # Load the dataset
        self.conversations = []
        
        if data_path and os.path.exists(data_path):
            # Load custom data from file
            self._load_custom_data(data_path)
        elif dataset_name:
            # Load from Hugging Face datasets
            self._load_hf_dataset(dataset_name, split, cache_dir)
        
        # Add learning samples if provided
        if learning_samples:
            self.conversations.extend(learning_samples)
        
        # Prepare examples
        self.examples = self._prepare_examples()
    
    def _add_special_tokens(self):
        """Add special tokens to the tokenizer if they don't exist."""
        special_tokens = list(self.speaker_tokens.values())
        
        # Check if we need to add these tokens
        tokens_to_add = []
        for token in special_tokens:
            if token not in self.tokenizer.get_vocab():
                tokens_to_add.append(token)
        
        if tokens_to_add:
            special_tokens_dict = {"additional_special_tokens": tokens_to_add}
            self.tokenizer.add_special_tokens(special_tokens_dict)
    
    def _load_custom_data(self, data_path: str):
        """Load custom dialogue data from file."""
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle different JSON formats
                if isinstance(data, list):
                    if data and isinstance(data[0], dict):
                        if "conversations" in data[0]:
                            # ShareGPT format
                            for item in data:
                                self.conversations.append(item["conversations"])
                        else:
                            # List of direct conversations
                            self.conversations.extend(data)
                elif isinstance(data, dict) and "conversations" in data:
                    # Dataset with a conversations field
                    self.conversations.extend(data["conversations"])
        else:
            # Handle other file formats here
            raise ValueError(f"Unsupported file format for {data_path}")
    
    def _load_hf_dataset(self, dataset_name: str, split: str, cache_dir: Optional[str] = None):
        """Load dataset from Hugging Face datasets."""
        # Map common dataset names to proper identifiers
        dataset_mapping = {
            "daily_dialog": ("daily_dialog", self._process_daily_dialog),
            "convai2": ("conv_ai_2", self._process_convai),
            "empathetic_dialogues": ("empathetic_dialogues", self._process_empathetic)
        }
        
        if dataset_name in dataset_mapping:
            hf_name, processor = dataset_mapping[dataset_name]
            data = load_dataset(hf_name, split=split, cache_dir=cache_dir)
            self.conversations = processor(data)
        else:
            # Try to load directly and use default processing
            try:
                data = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
                if "dialog" in data.column_names:
                    self.conversations = self._process_general_dialog(data)
                else:
                    raise ValueError(f"Unsupported dataset format for {dataset_name}")
            except Exception as e:
                raise ValueError(f"Failed to load dataset {dataset_name}: {str(e)}")
    
    def _process_daily_dialog(self, data):
        """Process the DailyDialog dataset format."""
        conversations = []
        for item in data:
            dialog = []
            for i, utterance in enumerate(item["dialog"]):
                # Alternating between user and assistant
                speaker = "user" if i % 2 == 0 else "assistant"
                dialog.append({"role": speaker, "content": utterance})
            conversations.append(dialog)
        return conversations
    
    def _process_convai(self, data):
        """Process the ConvAI2 dataset format."""
        conversations = []
        for item in data:
            dialog = []
            for i, utterance in enumerate(item["dialog"]):
                role = "user" if utterance["id"] % 2 == 0 else "assistant"
                dialog.append({"role": role, "content": utterance["text"]})
            conversations.append(dialog)
        return conversations
    
    def _process_empathetic(self, data):
        """Process the Empathetic Dialogues dataset format."""
        conversations = []
        current_conv = []
        current_conv_id = None
        
        # Sort by conversation ID and turn ID
        sorted_data = sorted(data, key=lambda x: (x["conv_id"], x["utterance_idx"]))
        
        for item in sorted_data:
            if current_conv_id is None:
                current_conv_id = item["conv_id"]
            
            if item["conv_id"] != current_conv_id:
                if current_conv:
                    conversations.append(current_conv)
                current_conv = []
                current_conv_id = item["conv_id"]
            
            role = "user" if item["speaker_idx"] % 2 == 0 else "assistant"
            current_conv.append({"role": role, "content": item["utterance"]})
        
        if current_conv:
            conversations.append(current_conv)
            
        return conversations
    
    def _process_general_dialog(self, data):
        """Process a general dialogue dataset with a 'dialog' field."""
        conversations = []
        for item in data:
            if "dialog" in item:
                dialog = []
                for i, utterance in enumerate(item["dialog"]):
                    if isinstance(utterance, dict) and "text" in utterance:
                        speaker = utterance.get("speaker", "user" if i % 2 == 0 else "assistant")
                        dialog.append({"role": speaker, "content": utterance["text"]})
                    elif isinstance(utterance, str):
                        speaker = "user" if i % 2 == 0 else "assistant"
                        dialog.append({"role": speaker, "content": utterance})
                conversations.append(dialog)
        return conversations
    
    def _prepare_examples(self):
        """Prepare examples for training by formatting and tokenizing the conversations."""
        examples = []
        
        for conversation in self.conversations:
            # Skip empty conversations
            if not conversation:
                continue
            
            # Add system prompt if provided
            formatted_text = ""
            if self.system_prompt:
                formatted_text = f"{self.speaker_tokens['system']} {self.system_prompt} {self.speaker_tokens['end']}\n"
            
            # Process conversation turns
            history = []
            for i, turn in enumerate(conversation):
                role = turn.get("role", "user" if i % 2 == 0 else "assistant")
                content = turn.get("content", "").strip()
                
                if not content:
                    continue
                
                turn_text = f"{self.speaker_tokens.get(role, role)} {content} {self.speaker_tokens['end']}\n"
                
                # For each turn, create an example with appropriate history
                if role == "assistant":
                    if self.include_history:
                        # Include conversation history up to this point
                        context = formatted_text + "".join(history[-self.history_length*2:])
                        full_text = context + turn_text
                    else:
                        # Only include the immediate user query and assistant response
                        if history:
                            full_text = formatted_text + history[-1] + turn_text
                        else:
                            full_text = formatted_text + turn_text
                    
                    # Tokenize the text
                    tokenized = self.tokenizer(full_text, truncation=True, max_length=self.max_length)
                    
                    # Find where the assistant response starts to create labels
                    assistant_token = self.speaker_tokens["assistant"]
                    assistant_token_ids = self.tokenizer.encode(assistant_token, add_special_tokens=False)
                    
                    # Find the last occurrence of the assistant token
                    input_ids = tokenized["input_ids"]
                    attention_mask = tokenized["attention_mask"]
                    
                    # First token of the last assistant turn
                    for i in range(len(input_ids) - len(assistant_token_ids)):
                        if input_ids[i:i+len(assistant_token_ids)] == assistant_token_ids:
                            assistant_start = i
                    
                    # Create labels: -100 for non-assistant tokens (ignored in loss)
                    labels = [-100] * len(input_ids)
                    labels[assistant_start:] = input_ids[assistant_start:]
                    
                    examples.append({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    })
                
                # Add this turn to history
                history.append(turn_text)
        
        return examples
    
    def add_learning_samples(self, samples: List[Dict]):
        """
        Add new samples for continuous learning.
        
        Args:
            samples: List of conversation samples to add
        
        Returns:
            Number of new examples added
        """
        original_len = len(self.examples)
        
        # Add samples to conversations
        self.conversations.extend(samples)
        
        # Update examples
        self.examples = self._prepare_examples()
        
        return len(self.examples) - original_len
    
    def __len__(self):
        """Get the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get an example from the dataset."""
        example = self.examples[idx]
        
        if self.return_tensors:
            # Convert to PyTorch tensors
            return {
                "input_ids": torch.tensor(example["input_ids"]),
                "attention_mask": torch.tensor(example["attention_mask"]),
                "labels": torch.tensor(example["labels"])
            }
        
        return example


def create_dialogue_dataloader(tokenizer, data_path=None, dataset_name=None, split="train", 
                              batch_size=8, max_length=1024, speaker_tokens=None,
                              system_prompt=None, shuffle=True, num_workers=4, 
                              cache_dir=None, collate_fn=None, learning_samples=None):
    """
    Create a DataLoader for dialogue data.
    
    Args:
        tokenizer: Tokenizer to use for preprocessing
        data_path: Path to custom dialogue data (JSON format)
        dataset_name: Name of the HF dataset to load (e.g., "daily_dialog")
        split: Dataset split (train, validation, test)
        batch_size: Batch size
        max_length: Maximum sequence length
        speaker_tokens: Dictionary mapping speaker roles to tokens
        system_prompt: Optional system prompt to prepend to conversations
        shuffle: Whether to shuffle the dataset
        num_workers: Number of workers for data loading
        cache_dir: Directory for caching the dataset
        collate_fn: Function to collate data samples into batches
        learning_samples: Optional additional samples for continuous learning
    
    Returns:
        torch.utils.data.DataLoader: DataLoader for the dialogue dataset
    """
    # Create the dataset
    dataset = DialogueDataset(
        tokenizer=tokenizer,
        data_path=data_path,
        dataset_name=dataset_name,
        split=split,
        max_length=max_length,
        speaker_tokens=speaker_tokens,
        system_prompt=system_prompt,
        cache_dir=cache_dir,
        return_tensors=True,
        learning_samples=learning_samples
    )
    
    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn or dialogue_collate_fn
    )
    
    return dataloader


def dialogue_collate_fn(batch):
    """
    Collate function for padding sequences in a dialogue batch.
    
    Args:
        batch: Batch of examples
    
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


def get_dialogue_dataloaders(tokenizer, data_path=None, dataset_name="daily_dialog", 
                            batch_size=8, max_length=1024, speaker_tokens=None,
                            system_prompt=None, num_workers=4, cache_dir=None, 
                            collate_fn=None, learning_samples=None):
    """
    Get DataLoaders for the train, validation, and test splits of dialogue data.
    
    Args:
        tokenizer: Tokenizer to use for preprocessing
        data_path: Path to custom dialogue data (JSON format)
        dataset_name: Name of the HF dataset to load (e.g., "daily_dialog")
        batch_size: Batch size
        max_length: Maximum sequence length
        speaker_tokens: Dictionary mapping speaker roles to tokens
        system_prompt: Optional system prompt to prepend to conversations
        num_workers: Number of workers for data loading
        cache_dir: Directory for caching the dataset
        collate_fn: Function to collate data samples into batches
        learning_samples: Optional additional samples for continuous learning
    
    Returns:
        Dict: DataLoaders for train, validation, and test splits
    """
    train_loader = create_dialogue_dataloader(
        tokenizer=tokenizer,
        data_path=data_path,
        dataset_name=dataset_name,
        split="train",
        batch_size=batch_size,
        max_length=max_length,
        speaker_tokens=speaker_tokens,
        system_prompt=system_prompt,
        shuffle=True,
        num_workers=num_workers,
        cache_dir=cache_dir,
        collate_fn=collate_fn,
        learning_samples=learning_samples
    )
    
    val_loader = create_dialogue_dataloader(
        tokenizer=tokenizer,
        data_path=data_path,
        dataset_name=dataset_name,
        split="validation",
        batch_size=batch_size,
        max_length=max_length,
        speaker_tokens=speaker_tokens,
        system_prompt=system_prompt,
        shuffle=False,
        num_workers=num_workers,
        cache_dir=cache_dir,
        collate_fn=collate_fn
    )
    
    # Not all datasets have a test split, so handle gracefully
    try:
        test_loader = create_dialogue_dataloader(
            tokenizer=tokenizer,
            data_path=data_path,
            dataset_name=dataset_name,
            split="test",
            batch_size=batch_size,
            max_length=max_length,
            speaker_tokens=speaker_tokens,
            system_prompt=system_prompt,
            shuffle=False,
            num_workers=num_workers,
            cache_dir=cache_dir,
            collate_fn=collate_fn
        )
    except Exception:
        # Fall back to validation set if test is not available
        test_loader = val_loader
    
    return {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader
    }