"""
Custom dataset loader.

This module provides functions for loading and processing custom datasets
from files in various formats.
"""

import os
import torch
import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Dataset as HFDataset
from transformers import PreTrainedTokenizer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from src.data.tensor_collate import default_collate_fn, dialogue_collate_fn
from src.data.dataloader_utils import setup_cache_dir

# Set up logging
logger = logging.getLogger("qllm_dataloaders")


def get_custom_dataloaders(
    tokenizer: PreTrainedTokenizer,
    train_file: str,
    validation_file: Optional[str] = None,
    test_file: Optional[str] = None,
    is_dialogue: bool = False,
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    max_length: int = 512,
    num_workers: int = 4,
    system_prompt: Optional[str] = None,
    timeout: int = 180
) -> Dict[str, DataLoader]:
    """
    Create data loaders for custom datasets.
    
    Args:
        tokenizer: Tokenizer to use for tokenizing the dataset
        train_file: Path to training data file
        validation_file: Path to validation data file
        test_file: Path to test data file
        is_dialogue: Whether the dataset is a dialogue dataset
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation (if None, use batch_size)
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        system_prompt: System prompt for dialogue datasets
        timeout: Timeout in seconds for dataset loading operations
        
    Returns:
        Dictionary of data loaders for train, validation, and test splits
    """
    # Set evaluation batch size
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    # Check if files exist
    if not os.path.exists(train_file):
        logger.error(f"Training file not found: {train_file}")
        logger.warning("Using dummy datasets for testing")
        if is_dialogue:
            from src.data.dummy_loaders import create_dummy_dialogue_dataloaders
            return create_dummy_dialogue_dataloaders(
                tokenizer, batch_size, eval_batch_size, num_workers, system_prompt
            )
        else:
            from src.data.dummy_loaders import create_dummy_dataloaders
            return create_dummy_dataloaders(
                tokenizer, batch_size, eval_batch_size, num_workers
            )
    
    # Load datasets with timeout protection
    logger.info(f"Loading custom dataset from {train_file}")
    
    # Determine file extension
    file_ext = os.path.splitext(train_file)[1].lower()
    
    try:
        # Load based on file extension with timeout protection
        if file_ext == ".json":
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(load_dataset, "json", data_files={"train": train_file})
                try:
                    train_data = future.result(timeout=timeout)["train"]
                except TimeoutError:
                    logger.error(f"Timeout ({timeout}s) while loading JSON dataset")
                    raise ValueError(f"Dataset loading timed out after {timeout} seconds")
            
            if validation_file and os.path.exists(validation_file):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(load_dataset, "json", data_files={"validation": validation_file})
                    val_data = future.result(timeout=timeout)["validation"]
            else:
                # Split train data for validation
                train_size = int(0.9 * len(train_data))
                val_size = len(train_data) - train_size
                train_data, val_data = train_data.train_test_split(
                    test_size=val_size, seed=42
                ).values()
            
            if test_file and os.path.exists(test_file):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(load_dataset, "json", data_files={"test": test_file})
                    test_data = future.result(timeout=timeout)["test"]
            else:
                # Use validation data for testing if no test file
                test_data = val_data
                
        elif file_ext == ".csv":
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(load_dataset, "csv", data_files={"train": train_file})
                try:
                    train_data = future.result(timeout=timeout)["train"]
                except TimeoutError:
                    logger.error(f"Timeout ({timeout}s) while loading CSV dataset")
                    raise ValueError(f"Dataset loading timed out after {timeout} seconds")
            
            if validation_file and os.path.exists(validation_file):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(load_dataset, "csv", data_files={"validation": validation_file})
                    val_data = future.result(timeout=timeout)["validation"]
            else:
                # Split train data for validation
                train_size = int(0.9 * len(train_data))
                val_size = len(train_data) - train_size
                train_data, val_data = train_data.train_test_split(
                    test_size=val_size, seed=42
                ).values()
            
            if test_file and os.path.exists(test_file):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(load_dataset, "csv", data_files={"test": test_file})
                    test_data = future.result(timeout=timeout)["test"]
            else:
                # Use validation data for testing if no test file
                test_data = val_data
                
        elif file_ext == ".txt":
            # For text files, use text dataset - read directly
            start_time = time.time()
            with open(train_file, "r", encoding="utf-8") as f:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Reading text file timed out after {timeout} seconds")
                train_text = f.read()
            
            if validation_file and os.path.exists(validation_file):
                start_time = time.time()
                with open(validation_file, "r", encoding="utf-8") as f:
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Reading validation file timed out after {timeout} seconds")
                    val_text = f.read()
            else:
                # Split train text for validation (simple approach)
                split_idx = int(len(train_text) * 0.9)
                val_text = train_text[split_idx:]
                train_text = train_text[:split_idx]
            
            if test_file and os.path.exists(test_file):
                start_time = time.time()
                with open(test_file, "r", encoding="utf-8") as f:
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Reading test file timed out after {timeout} seconds")
                    test_text = f.read()
            else:
                # Use validation text for testing
                test_text = val_text
            
            # Create Huggingface datasets
            train_data = HFDataset.from_dict({"text": [train_text]})
            val_data = HFDataset.from_dict({"text": [val_text]})
            test_data = HFDataset.from_dict({"text": [test_text]})
            
        else:
            logger.error(f"Unsupported file extension: {file_ext}")
            logger.warning("Using dummy datasets for testing")
            if is_dialogue:
                from src.data.dummy_loaders import create_dummy_dialogue_dataloaders
                return create_dummy_dialogue_dataloaders(
                    tokenizer, batch_size, eval_batch_size, num_workers, system_prompt
                )
            else:
                from src.data.dummy_loaders import create_dummy_dataloaders
                return create_dummy_dataloaders(
                    tokenizer, batch_size, eval_batch_size, num_workers
                )
    
    except Exception as e:
        logger.error(f"Error loading custom dataset: {e}")
        logger.warning("Using dummy datasets for testing")
        if is_dialogue:
            from src.data.dummy_loaders import create_dummy_dialogue_dataloaders
            return create_dummy_dialogue_dataloaders(
                tokenizer, batch_size, eval_batch_size, num_workers, system_prompt
            )
        else:
            from src.data.dummy_loaders import create_dummy_dataloaders
            return create_dummy_dataloaders(
                tokenizer, batch_size, eval_batch_size, num_workers
            )
    
    # Create appropriate datasets and dataloaders based on whether it's dialogue data
    if is_dialogue:
        # Import with performance optimizations
        from src.data.dialogue_dataset import DialogueDataset
        from src.data.daily_dialog_loader import create_optimized_dialogue_dataset
        
        # Pre-compute helpful mappings for efficiency
        role_mapping = {
            "system": "<system>",
            "user": "<user>",
            "assistant": "<assistant>",
            "human": "<user>",
            "bot": "<assistant>"
        }
        
        # Process JSON files for dialogue
        if file_ext == ".json":
            logger.info("Processing JSON files for dialogue dataset...")
            
            # Try to extract dialogues from the data
            train_dialogues = extract_dialogues_from_json(train_data, system_prompt)
            val_dialogues = extract_dialogues_from_json(val_data, system_prompt)
            test_dialogues = extract_dialogues_from_json(test_data, system_prompt)
            
            # Create dialogue datasets with optimization
            logger.info(f"Creating training DialogueDataset with {len(train_dialogues)} dialogues")
            train_dataset = create_optimized_dialogue_dataset(
                tokenizer=tokenizer,
                dialogues=train_dialogues,
                max_length=max_length,
                system_prompt=system_prompt,
                role_mapping=role_mapping
            )
            
            logger.info(f"Creating validation DialogueDataset with {len(val_dialogues)} dialogues")
            val_dataset = create_optimized_dialogue_dataset(
                tokenizer=tokenizer,
                dialogues=val_dialogues,
                max_length=max_length,
                system_prompt=system_prompt,
                role_mapping=role_mapping
            )
            
            logger.info(f"Creating test DialogueDataset with {len(test_dialogues)} dialogues")
            test_dataset = create_optimized_dialogue_dataset(
                tokenizer=tokenizer,
                dialogues=test_dialogues,
                max_length=max_length,
                system_prompt=system_prompt,
                role_mapping=role_mapping
            )
            
        else:
            # For non-JSON files, use the raw text approach with DialogueDataset
            train_dataset = DialogueDataset(
                tokenizer=tokenizer,
                data_path=train_file if file_ext == ".json" else None,
                dialogues=None if file_ext == ".json" else [extract_simple_dialogue(train_data)],
                max_length=max_length,
                system_prompt=system_prompt,
                role_mapping=role_mapping,
                processing_batch_size=8  # Use smaller batches
            )
            
            val_dataset = DialogueDataset(
                tokenizer=tokenizer,
                data_path=validation_file if validation_file and file_ext == ".json" else None,
                dialogues=None if file_ext == ".json" else [extract_simple_dialogue(val_data)],
                max_length=max_length,
                system_prompt=system_prompt,
                role_mapping=role_mapping,
                processing_batch_size=8
            )
            
            test_dataset = DialogueDataset(
                tokenizer=tokenizer,
                data_path=test_file if test_file and file_ext == ".json" else None,
                dialogues=None if file_ext == ".json" else [extract_simple_dialogue(test_data)],
                max_length=max_length,
                system_prompt=system_prompt,
                role_mapping=role_mapping,
                processing_batch_size=8
            )
        
        # Create dataloaders
        dataloaders = {}
        
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=dialogue_collate_fn,
        )
        
        dataloaders["validation"] = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=dialogue_collate_fn,
        )
        
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=dialogue_collate_fn,
        )
        
    else:
        # Language modeling dataset
        
        # Tokenization function
        def tokenize_function(examples):
            # Check what column to use (text, content, etc.)
            text_column = None
            for column in ["text", "content", "document"]:
                if column in examples:
                    text_column = column
                    break
            
            if text_column is None:
                # Use the first string column
                string_columns = [col for col, values in examples.items() 
                                if isinstance(values[0], str)]
                if string_columns:
                    text_column = string_columns[0]
                else:
                    raise ValueError("No text column found in dataset")
            
            # Tokenize all texts
            tokenized = tokenizer(
                examples[text_column],
                truncation=True,
                max_length=max_length,
                return_token_type_ids=False,
            )
            return tokenized
        
        # Apply tokenization
        logger.info("Tokenizing datasets...")
        train_tokenized = train_data.map(
            tokenize_function,
            batched=True,
            num_proc=num_workers,
            remove_columns=train_data.column_names,
            desc="Tokenizing training data",
        )
        
        val_tokenized = val_data.map(
            tokenize_function,
            batched=True,
            num_proc=num_workers,
            remove_columns=val_data.column_names,
            desc="Tokenizing validation data",
        )
        
        test_tokenized = test_data.map(
            tokenize_function,
            batched=True,
            num_proc=num_workers,
            remove_columns=test_data.column_names,
            desc="Tokenizing test data",
        )
        
        # Add labels for language modeling
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples
        
        train_dataset = train_tokenized.map(
            add_labels, batched=True, desc="Adding labels to training data"
        )
        val_dataset = val_tokenized.map(
            add_labels, batched=True, desc="Adding labels to validation data"
        )
        test_dataset = test_tokenized.map(
            add_labels, batched=True, desc="Adding labels to test data"
        )
        
        # Create dataloaders
        dataloaders = {}
        
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=default_collate_fn,
        )
        
        dataloaders["validation"] = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=default_collate_fn,
        )
        
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=default_collate_fn,
        )
    
    logger.info(f"Created dataloaders: {len(dataloaders['train'])} training batches")
    return dataloaders


def extract_dialogues_from_json(dataset: HFDataset, system_prompt: Optional[str] = None) -> List[List[Dict[str, str]]]:
    """
    Extract dialogues from a JSON dataset.
    
    This function attempts to extract dialogue data from various JSON formats.
    
    Args:
        dataset: The dataset to extract dialogues from
        system_prompt: Optional system prompt to add
        
    Returns:
        List of dialogues
    """
    dialogues = []
    
    # Check for dialogue-specific columns
    if "dialog" in dataset.column_names or "dialogue" in dataset.column_names or "conversation" in dataset.column_names:
        dialogue_col = "dialog" if "dialog" in dataset.column_names else (
            "dialogue" if "dialogue" in dataset.column_names else "conversation"
        )
        
        # Process each row
        for i in range(len(dataset)):
            item = dataset[i]
            dialogue = []
            
            # Add system prompt if provided
            if system_prompt:
                dialogue.append({"role": "system", "content": system_prompt})
            
            # Extract dialogue turns
            turns = item[dialogue_col]
            if isinstance(turns, list):
                for j, turn in enumerate(turns):
                    if isinstance(turn, dict):
                        # Handle dictionary format with role/content
                        if "role" in turn and "content" in turn:
                            dialogue.append({
                                "role": turn["role"],
                                "content": turn["content"]
                            })
                        elif "speaker" in turn and "text" in turn:
                            dialogue.append({
                                "role": turn["speaker"],
                                "content": turn["text"]
                            })
                    elif isinstance(turn, str):
                        # Handle string format, alternating user/assistant
                        role = "user" if j % 2 == 0 else "assistant"
                        dialogue.append({
                            "role": role,
                            "content": turn
                        })
            
            if dialogue:
                dialogues.append(dialogue)
                
    # Try messages format
    elif "messages" in dataset.column_names:
        for i in range(len(dataset)):
            item = dataset[i]
            dialogue = []
            
            # Add system prompt if provided and not already present
            if system_prompt:
                has_system = False
                for msg in item["messages"]:
                    if isinstance(msg, dict) and msg.get("role") == "system":
                        has_system = True
                        break
                
                if not has_system:
                    dialogue.append({"role": "system", "content": system_prompt})
            
            # Add messages
            for msg in item["messages"]:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    dialogue.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            if dialogue:
                dialogues.append(dialogue)
    
    # Try simpler approach with alternating turns
    elif len(dataset.column_names) > 0:
        # Use first column for text if nothing else found
        text_col = dataset.column_names[0]
        
        for i in range(len(dataset)):
            item = dataset[i]
            dialogue = []
            
            # Add system prompt if provided
            if system_prompt:
                dialogue.append({"role": "system", "content": system_prompt})
            
            # Add a single turn
            dialogue.append({
                "role": "user",
                "content": str(item[text_col])
            })
            
            if len(dialogue) > 1 or (len(dialogue) == 1 and dialogue[0]["role"] != "system"):
                dialogues.append(dialogue)
    
    logger.info(f"Extracted {len(dialogues)} dialogues from dataset")
    return dialogues


def extract_simple_dialogue(dataset: HFDataset) -> List[Dict[str, str]]:
    """
    Extract a simple dialogue from text data.
    
    Args:
        dataset: The dataset to extract dialogue from
        
    Returns:
        A simple dialogue with alternating user/assistant turns
    """
    dialogue = []
    
    # Find text column
    text_col = None
    for col in ["text", "content", "document"]:
        if col in dataset.column_names:
            text_col = col
            break
    
    if text_col is None and len(dataset.column_names) > 0:
        text_col = dataset.column_names[0]
    
    if text_col:
        # Extract text and split into paragraphs
        text = dataset[0][text_col]
        paragraphs = text.split("\n\n")
        
        # Create alternating user/assistant dialogue
        for i, para in enumerate(paragraphs):
            if para.strip():
                role = "user" if i % 2 == 0 else "assistant"
                dialogue.append({
                    "role": role,
                    "content": para.strip()
                })
    
    return dialogue