"""
Daily Dialog dataset loader.

This module provides functions for loading and processing the Daily Dialog dataset
for conversational language modeling tasks.
"""

import os
import logging
import time
from typing import Dict, Any, Optional, List, Set
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from src.data.dialogue_dataset import DialogueDataset
from src.data.utils.tensor_collate import dialogue_collate_fn
from src.data.utils.caching import setup_cache_dir, load_from_cache, save_to_cache
from src.data.base.base_loader import BaseLoader
# Set up logging
logger = logging.getLogger("qllm_dataloaders")

# Performance tuning constants - critical for fast processing
PROCESSING_BATCH_SIZE = 8  # Reduced from 64 to prevent memory issues
DEBUG_PROCESSING = False   # Set to True to enable detailed timing logs


class DailyDialogLoader(BaseLoader):
    """
    Loader for the Daily Dialog dataset.
    
    This class provides functionality for loading and processing the Daily Dialog dataset,
    which contains conversations between two speakers.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 16,
        eval_batch_size: Optional[int] = None,
        max_length: int = 512,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
        timeout: int = 300,
        use_local_fallback: bool = True,
        **kwargs
    ):
        """
        Initialize the Daily Dialog loader.
        
        Args:
            tokenizer: Tokenizer to use for tokenizing the dataset
            batch_size: Batch size for training
            eval_batch_size: Batch size for evaluation (if None, use batch_size)
            max_length: Maximum sequence length
            num_workers: Number of workers for data loading
            cache_dir: Cache directory for datasets
            system_prompt: System prompt to prepend to dialogues
            timeout: Timeout in seconds for dataset download
            use_local_fallback: Whether to use local fallback if download fails
            **kwargs: Additional parameters to pass to the base loader
        """
        super().__init__(
            cache_dir=cache_dir,
            use_cache=True,
            validate=True,
            **kwargs
        )
        
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.use_local_fallback = use_local_fallback
        
        # Initialize attributes to store loaded data
        self.daily_dialog = None
        self.dataloaders = {}
        
    def _load_file(self, filepath: str) -> List[Any]:
        """
        Load data from a file.
        
        This method is not used as the Daily Dialog dataset is loaded from the HuggingFace datasets.
        It's implemented as a placeholder to satisfy the abstract method requirement.
        
        Args:
            filepath: Path to the file to load
            
        Returns:
            List of loaded data samples
        """
        # This loader doesn't use local files directly
        # It loads from HuggingFace datasets
        return []
    
    def _validate_sample(self, sample: Any) -> bool:
        """
        Validate a single data sample.
        
        Args:
            sample: Data sample to validate
            
        Returns:
            True if the sample is valid, False otherwise
        """
        # For dialogue samples, check if it contains valid dialogues
        if not isinstance(sample, dict):
            return False
            
        if "dialog" not in sample:
            return False
            
        if not sample["dialog"]:
            return False
            
        return True
    
    def _store_loaded_data(self, data: List[Any]) -> None:
        """
        Store the loaded data.
        
        Args:
            data: List of loaded data samples
        """
        self._loaded_data = data
        
    def get_loaded_data(self) -> List[Any]:
        """
        Get the loaded data.
        
        Returns:
            List of loaded data samples
        """
        return self._loaded_data
    
    def load(self) -> Dict[str, DataLoader]:
        """
        Load and prepare the Daily Dialog dataset.
        
        Returns:
            Dictionary of data loaders for train, validation, and test splits
        """
        # Use the existing implementation to create data loaders
        self.dataloaders = get_daily_dialog_dataloaders(
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            eval_batch_size=self.eval_batch_size,
            max_length=self.max_length,
            num_workers=self.num_workers,
            cache_dir=self.cache_dir,
            system_prompt=self.system_prompt,
            timeout=self.timeout,
            use_local_fallback=self.use_local_fallback
        )
        
        self.is_loaded = True
        return self.dataloaders
    
    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Get the prepared data loaders.
        
        Returns:
            Dictionary of data loaders for train, validation, and test splits
        """
        if not self.is_loaded:
            return self.load()
        return self.dataloaders




def get_daily_dialog_dataloaders(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    max_length: int = 512,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    system_prompt: str = "You are a helpful assistant.",
    timeout: int = 300,
    use_local_fallback: bool = True
) -> Dict[str, DataLoader]:
    """
    Create data loaders for the Daily Dialog dataset.
    
    Args:
        tokenizer: Tokenizer to use for tokenizing the dataset
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation (if None, use batch_size)
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        cache_dir: Cache directory for datasets
        system_prompt: System prompt to prepend to dialogues
        timeout: Timeout in seconds for dataset download
        use_local_fallback: Whether to use local fallback if download fails
        
    Returns:
        Dictionary of data loaders for train, validation, and test splits
    """
    # Set evaluation batch size
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    # Make sure cache directory exists
    cache_dir = setup_cache_dir(cache_dir)
    
    # Check for cached dataset to avoid redundant downloads
    cached_path = os.path.join(cache_dir or ".cache", "daily_dialog.cached")
    daily_dialog = load_from_cache(cached_path)
    
    if daily_dialog is None:
        # Load Daily Dialog dataset with timeout protection
        logger.info("Downloading Daily Dialog dataset...")
        
        try:
            # Use executor with timeout to prevent hanging
            with ThreadPoolExecutor(max_workers=1) as executor:
                # Add trust_remote_code=True to handle custom code in the dataset
                future = executor.submit(load_dataset, "daily_dialog", cache_dir=cache_dir, trust_remote_code=True)
                try:
                    daily_dialog = future.result(timeout=timeout)
                    logger.info("Daily Dialog dataset downloaded successfully")
                    # Cache the dataset for future use
                    if cache_dir:
                        save_to_cache(daily_dialog, cached_path)
                except TimeoutError:
                    logger.error(f"Timeout ({timeout}s) while downloading Daily Dialog dataset")
                    if use_local_fallback:
                        logger.info("Using built-in fallback dataset...")
                        daily_dialog = create_fallback_daily_dialog()
                    else:
                        raise ValueError(f"Dataset download timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Error loading Daily Dialog dataset: {e}")
            if use_local_fallback:
                logger.info("Using built-in fallback dataset...")
                daily_dialog = create_fallback_daily_dialog()
            else:
                logger.warning("Using dummy datasets for testing")
                from data.loaders.dummy_loaders import create_dummy_dialogue_dataloaders
                return create_dummy_dialogue_dataloaders(
                    tokenizer, batch_size, eval_batch_size, num_workers, system_prompt
                )
    
    # Display dataset sizes
    logger.info(f"Daily Dialog dataset loaded: {len(daily_dialog['train'])} dialogues")
    
    # Process dataset into dialogues
    def process_daily_dialog(examples):
        dialogues = []
        for dial_idx in range(len(examples["dialog"])):
            dialog = examples["dialog"][dial_idx]
            dialogue = []
            
            # Add system message
            if system_prompt:
                dialogue.append({"role": "system", "content": system_prompt})
            
            # Add utterances
            for i, utterance in enumerate(dialog):
                role = "user" if i % 2 == 0 else "assistant"
                dialogue.append({"role": role, "content": utterance})
            
            dialogues.append(dialogue)
        
        return {"dialogues": dialogues}
    
    # Convert to dialogue format - output progress
    logger.info("Processing dialogues...")
    processed_dataset = daily_dialog.map(
        process_daily_dialog,
        batched=True,
        num_proc=num_workers,
        remove_columns=daily_dialog["train"].column_names,
        desc="Processing dialogues",
    )

    # Performance optimization: Create a optimized DialogueDataset that processes in small batches
    logger.info("Creating optimized DialogueDataset instances...")
    
    # Pre-compute all role token IDs to avoid repeated lookups
    role_mapping = {
        "system": "<system>",
        "user": "<user>",
        "assistant": "<assistant>",
        "human": "<user>",
        "bot": "<assistant>"
    }
    
    # Log the start of creation for each dataset to monitor progress
    logger.info("Creating training dataset...")
    train_start = time.time()
    train_dataset = create_optimized_dialogue_dataset(
        tokenizer=tokenizer,
        dialogues=processed_dataset["train"]["dialogues"],
        max_length=max_length,
        system_prompt=system_prompt,
        role_mapping=role_mapping
    )
    logger.info(f"Training dataset created in {time.time() - train_start:.2f} seconds")
    
    logger.info("Creating validation dataset...")
    val_start = time.time()
    val_dataset = create_optimized_dialogue_dataset(
        tokenizer=tokenizer,
        dialogues=processed_dataset["validation"]["dialogues"],
        max_length=max_length,
        system_prompt=system_prompt,
        role_mapping=role_mapping
    )
    logger.info(f"Validation dataset created in {time.time() - val_start:.2f} seconds")
    
    logger.info("Creating test dataset...")
    test_start = time.time()
    test_dataset = create_optimized_dialogue_dataset(
        tokenizer=tokenizer,
        dialogues=processed_dataset["test"]["dialogues"],
        max_length=max_length,
        system_prompt=system_prompt,
        role_mapping=role_mapping
    )
    logger.info(f"Test dataset created in {time.time() - test_start:.2f} seconds")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
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
    
    logger.info(f"Created dataloaders: {len(dataloaders['train'])} training batches")
    
    return dataloaders


def create_optimized_dialogue_dataset(
    tokenizer: PreTrainedTokenizer,
    dialogues: List[List[Dict[str, str]]],
    max_length: int,
    system_prompt: Optional[str] = None,
    role_mapping: Optional[Dict[str, str]] = None
) -> DialogueDataset:
    """
    Create an optimized DialogueDataset with batched processing.
    
    Args:
        tokenizer: Tokenizer to use
        dialogues: List of dialogues
        max_length: Maximum sequence length
        system_prompt: System prompt to prepend
        role_mapping: Role mapping for special tokens
        
    Returns:
        DialogueDataset instance with pre-processed data
    """
    # Create a single system prompt object if provided
    system_prompt_obj = None
    if system_prompt:
        system_prompt_obj = {"role": "system", "content": system_prompt}
    
    # Pre-process dialogues in smaller batches to show progress
    total_dialogues = len(dialogues)
    batch_size = PROCESSING_BATCH_SIZE  # Use a smaller batch size than the default 64
    num_batches = (total_dialogues + batch_size - 1) // batch_size
    
    # Pre-compute role token IDs for faster processing
    role_tokens = {}
    for role, token in role_mapping.items():
        role_tokens[role] = tokenizer.convert_tokens_to_ids(token)
    
    # Store the token IDs for special tokens
    assistant_token_id = role_tokens.get('assistant')
    role_token_ids = set(role_tokens.values())
    
    processed_dialogues = []
    
    for batch_idx in tqdm(range(num_batches), desc="Preprocessing dialogues"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_dialogues)
        batch_dialogues = dialogues[start_idx:end_idx]
        
        # Process each dialogue in the batch
        for dialogue in batch_dialogues:
            # Add system prompt if not already present
            if system_prompt_obj and (not dialogue or dialogue[0].get("role") != "system"):
                dialogue = [system_prompt_obj] + dialogue
            processed_dialogues.append(dialogue)
    
    # Create the dataset with our processed dialogues
    logger.info(f"Creating DialogueDataset with {len(processed_dialogues)} dialogues")
    
    # Pass the pre-computed role tokens to DialogueDataset for faster processing
    dataset = DialogueDataset(
        tokenizer=tokenizer,
        dialogues=processed_dialogues,
        max_length=max_length,
        # Use optimized parameters:
        processing_batch_size=batch_size,  # Smaller batch size
        role_token_ids=role_token_ids,     # Pre-computed role token IDs
        assistant_token_id=assistant_token_id  # Pre-computed assistant token ID
    )
    
    return dataset


def create_fallback_daily_dialog():
    """
    Create a built-in fallback version of the Daily Dialog dataset.
    This avoids having to download the dataset from the internet when it fails.
    
    Returns:
        HuggingFace Dataset object with the same structure as daily_dialog
    """
    # Create a minimal version of Daily Dialog
    fallback_dialogs = [
        # Sample dialog 1
        [
            "Hello, how are you doing today?",
            "I'm doing well, thank you for asking. How about you?",
            "Pretty good. I just got back from a vacation.",
            "That sounds amazing! Where did you go?",
            "I went to Hawaii. The beaches were incredible."
        ],
        # Sample dialog 2
        [
            "Do you have any recommendations for a good restaurant?",
            "Yes, there's a new Italian place downtown called Bella Pasta.",
            "What kind of dishes do they serve?",
            "They have amazing handmade pasta and authentic wood-fired pizza.",
            "That sounds delicious. I'll try it this weekend."
        ],
        # Sample dialog 3
        [
            "I've been thinking about learning a new language.",
            "That's a great idea! What language are you interested in?",
            "I'm considering either Spanish or Japanese.",
            "Both are good choices. Spanish might be easier to start with.",
            "You're probably right. I'll look into Spanish courses."
        ]
    ]
    
    # Add more sample dialogs
    more_dialogs = [
        [
            "Have you seen the latest superhero movie?",
            "Yes, I watched it last weekend. The special effects were amazing.",
            "Who was your favorite character?",
            "Definitely the villain. The actor's performance was outstanding.",
            "I agree. The villain had much more depth than in previous films."
        ],
        [
            "I'm having trouble with my computer.",
            "What seems to be the problem?",
            "It keeps crashing whenever I open multiple applications.",
            "Have you tried updating your drivers?",
            "No, I haven't. I'll try that. Thanks for the suggestion."
        ]
    ]
    fallback_dialogs.extend(more_dialogs)
    
    # Create 20 more dialogues by mixing and matching sentences
    sample_sentences = [
        "I really appreciate your help.",
        "What do you think about the current situation?",
        "Have you been to the new mall?",
        "The weather is beautiful today.",
        "I'm looking forward to the weekend.",
        "That's an interesting perspective.",
        "Could you explain that in more detail?",
        "I'd like to hear more about your project.",
        "Let's meet up for coffee sometime.",
        "How long have you been working on this?"
    ]
    
    responses = [
        "That's great to hear.",
        "I think it's quite complex, but promising.",
        "Yes, it's impressive. The architecture is stunning.",
        "I know! Perfect for a walk in the park.",
        "Me too. I have some exciting plans.",
        "Thank you. I've given it a lot of thought.",
        "Of course. What specific aspects are you curious about?",
        "I'd be happy to tell you more about it next time we meet.",
        "That sounds wonderful. How about next Tuesday?",
        "About three months now. It's been challenging but rewarding."
    ]
    
    import random
    random.seed(42)  # For reproducibility
    
    for _ in range(20):
        dialog_length = random.randint(3, 6) * 2  # Even number of turns
        dialog = []
        
        for i in range(dialog_length):
            if i % 2 == 0:
                dialog.append(random.choice(sample_sentences))
            else:
                dialog.append(random.choice(responses))
        
        fallback_dialogs.append(dialog)
    
    # Create splits
    train_size = int(len(fallback_dialogs) * 0.7)
    val_size = int(len(fallback_dialogs) * 0.15)
    test_size = len(fallback_dialogs) - train_size - val_size
    
    train_dialogs = fallback_dialogs[:train_size]
    val_dialogs = fallback_dialogs[train_size:train_size+val_size]
    test_dialogs = fallback_dialogs[train_size+val_size:]
    
    # Create dataset in the format expected by the daily_dialog dataset
    def create_split(dialogs):
        return Dataset.from_dict({
            "dialog": dialogs,
            "act": [[0] * len(d) for d in dialogs],  # Dummy act labels
            "emotion": [[0] * len(d) for d in dialogs],  # Dummy emotion labels
        })
    
    dataset_dict = DatasetDict({
        "train": create_split(train_dialogs),
        "validation": create_split(val_dialogs),
        "test": create_split(test_dialogs)
    })
    
    logger.info(f"Created fallback dataset with {len(train_dialogs)} training dialogs")
    return dataset_dict