"""
Function Calling Dataset module for QLLM.

This module extends the dialogue dataset to support function calling capabilities,
allowing the model to generate structured JSON outputs and handle API-like interactions.
"""

import os
import json
import torch
import logging
import random
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

from src.data.dialogue_dataset import DialogueDataset, dialogue_collate_fn


logger = logging.getLogger(__name__)


@dataclass
class FunctionSchema:
    """Schema for a callable function."""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]
    required_parameters: List[str]


class FunctionCallingDataset(DialogueDataset):
    """
    Dataset for training models with function calling capabilities.
    
    This dataset extends DialogueDataset to support function definitions,
    function calls, and structured JSON responses.
    """
    
    def __init__(self, 
                 tokenizer,
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
                 learning_samples: Optional[List[Dict]] = None,
                 function_definitions: Optional[List[Dict]] = None,
                 json_format_probability: float = 0.5):
        """
        Initialize the FunctionCallingDataset.
        
        Args:
            tokenizer: Tokenizer to use for preprocessing
            data_path: Path to custom dialogue data (JSON format)
            dataset_name: Name of the HF dataset to load
            split: Dataset split (train, validation, test)
            max_length: Maximum sequence length
            speaker_tokens: Dictionary mapping speaker roles to tokens
            system_prompt: Optional system prompt to include at the start of conversations
            cache_dir: Directory for caching the dataset
            return_tensors: Whether to return PyTorch tensors
            include_history: Whether to include conversation history
            history_length: Maximum number of turns to include in history
            learning_samples: Optional additional samples for continuous learning
            function_definitions: List of function definitions (name, description, parameters)
            json_format_probability: Probability of formatting assistant responses as JSON
        """
        # Set up function calling specific tokens
        extended_speaker_tokens = speaker_tokens or {
            "system": "<|system|>",
            "user": "<|user|>",
            "assistant": "<|assistant|>",
            "end": "<|end|>",
        }
        
        # Add function-related special tokens
        function_tokens = {
            "function": "<|function|>",
            "json_response": "<|json_response|>",
            "function_call": "<|function_call|>",
            "function_result": "<|function_result|>",
        }
        extended_speaker_tokens.update(function_tokens)
        
        self.function_tokens = function_tokens
        self.json_format_probability = json_format_probability
        
        # Initialize with base DialogueDataset
        super().__init__(
            tokenizer=tokenizer,
            data_path=data_path,
            dataset_name=dataset_name,
            split=split,
            max_length=max_length,
            speaker_tokens=extended_speaker_tokens,
            system_prompt=system_prompt,
            cache_dir=cache_dir,
            return_tensors=return_tensors,
            include_history=include_history,
            history_length=history_length,
            learning_samples=learning_samples
        )
        
        # Process function definitions
        self.function_schemas = []
        if function_definitions:
            for func_def in function_definitions:
                self._add_function_schema(func_def)
        
        # Process function calling examples separately
        self._prepare_function_examples()
    
    def _add_function_schema(self, function_def: Dict[str, Any]) -> FunctionSchema:
        """
        Add a function schema from a definition dictionary.
        
        Args:
            function_def: Function definition dictionary
            
        Returns:
            FunctionSchema object
        """
        required = function_def.get("required", [])
        schema = FunctionSchema(
            name=function_def["name"],
            description=function_def.get("description", ""),
            parameters=function_def.get("parameters", {}),
            required_parameters=required if isinstance(required, list) else []
        )
        self.function_schemas.append(schema)
        return schema
    
    def _format_function_definitions(self, selected_functions: Optional[List[FunctionSchema]] = None) -> str:
        """
        Format function definitions as a string for inclusion in prompts.
        
        Args:
            selected_functions: List of function schemas to include (None = all)
            
        Returns:
            Formatted function definitions string
        """
        functions = selected_functions if selected_functions else self.function_schemas
        if not functions:
            return ""
            
        result = f"{self.function_tokens['function']}\n"
        result += "Available functions:\n"
        
        for func in functions:
            result += f"- {func.name}: {func.description}\n"
            result += "  Parameters:\n"
            
            for param_name, param_info in func.parameters.items():
                param_type = param_info.get("type", "string")
                description = param_info.get("description", "")
                required = "required" if param_name in func.required_parameters else "optional"
                result += f"  - {param_name} ({param_type}, {required}): {description}\n"
            
            result += "\n"
        
        result += f"{self.speaker_tokens['end']}\n"
        return result
    
    def _format_function_call(self, function_call: Dict[str, Any]) -> str:
        """
        Format a function call as a string for inclusion in examples.
        
        Args:
            function_call: Dictionary with function call details
            
        Returns:
            Formatted function call string
        """
        function_name = function_call.get("name", "")
        arguments = function_call.get("arguments", {})
        
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except:
                arguments = {"raw_input": arguments}
        
        result = f"{self.function_tokens['function_call']}\n"
        result += "{\n"
        result += f'  "name": "{function_name}",\n'
        result += '  "arguments": {\n'
        
        for i, (key, value) in enumerate(arguments.items()):
            if isinstance(value, str):
                value_str = f'"{value}"'
            else:
                value_str = json.dumps(value)
                
            if i < len(arguments) - 1:
                result += f'    "{key}": {value_str},\n'
            else:
                result += f'    "{key}": {value_str}\n'
                
        result += "  }\n"
        result += "}\n"
        result += f"{self.speaker_tokens['end']}\n"
        
        return result
    
    def _format_function_result(self, function_result: Any) -> str:
        """
        Format a function result as a string for inclusion in examples.
        
        Args:
            function_result: The result returned from a function
            
        Returns:
            Formatted function result string
        """
        result = f"{self.function_tokens['function_result']}\n"
        
        if isinstance(function_result, str):
            try:
                # Try to parse as JSON for prettier formatting
                parsed = json.loads(function_result)
                result += json.dumps(parsed, indent=2)
            except:
                # Use as raw string
                result += function_result
        else:
            result += json.dumps(function_result, indent=2)
            
        result += f"\n{self.speaker_tokens['end']}\n"
        
        return result
    
    def _format_json_response(self, content: str) -> str:
        """
        Format a regular assistant response as JSON.
        
        Args:
            content: Original response content
            
        Returns:
            JSON-formatted response
        """
        result = f"{self.function_tokens['json_response']}\n"
        result += "{\n"
        result += f'  "response": "{content.replace(chr(34), chr(92)+chr(34))}"\n'
        result += "}\n"
        result += f"{self.speaker_tokens['end']}\n"
        
        return result
    
    def _prepare_function_examples(self):
        """Process conversation examples to include function calling patterns."""
        # This function is called by __init__ after processing regular dialogue examples
        function_examples = []
        
        for conversation in self.conversations:
            # Skip conversations that don't have enough turns
            if len(conversation) < 2:
                continue
                
            # Find conversations that might be suitable for function calling
            # For example, conversations with questions or instructions
            user_turns = [turn for turn in conversation if turn.get("role") == "user"]
            if not user_turns:
                continue
                
            # Only convert some conversations to function calling examples
            if random.random() > 0.3:  # 30% probability for function calling examples
                continue
                
            # Create a modified conversation with function calls
            modified_conversation = []
            
            # Maybe add a system message explaining function capabilities
            if not any(turn.get("role") == "system" for turn in conversation):
                if random.random() > 0.5 and self.function_schemas:
                    # Randomly select a subset of available functions
                    num_functions = min(len(self.function_schemas), 
                                      max(1, int(random.random() * len(self.function_schemas))))
                    selected_functions = random.sample(self.function_schemas, num_functions)
                    
                    # Create system message with function definitions
                    system_message = {
                        "role": "system",
                        "content": f"You can use functions to perform actions or retrieve information. "
                                  f"{self._format_function_definitions(selected_functions)}"
                    }
                    modified_conversation.append(system_message)
            
            # Process the conversation, potentially converting some assistant responses to function calls
            for i, turn in enumerate(conversation):
                role = turn.get("role", "")
                content = turn.get("content", "")
                
                if not content.strip():
                    continue
                
                if role == "assistant" and i > 0 and random.random() < 0.3 and self.function_schemas:
                    # Convert this assistant response to a function call
                    function_schema = random.choice(self.function_schemas)
                    
                    # Create a plausible function call based on the schema
                    arguments = {}
                    for param_name, param_info in function_schema.parameters.items():
                        if param_name in function_schema.required_parameters or random.random() > 0.3:
                            param_type = param_info.get("type", "string")
                            
                            # Generate a plausible value based on the type
                            if param_type == "string":
                                arguments[param_name] = f"sample_{param_name}"
                            elif param_type == "number" or param_type == "integer":
                                arguments[param_name] = random.randint(1, 100)
                            elif param_type == "boolean":
                                arguments[param_name] = random.choice([True, False])
                    
                    # Add the function call
                    function_call = {
                        "name": function_schema.name,
                        "arguments": arguments
                    }
                    
                    # Replace the assistant's message with a function call
                    modified_conversation.append({
                        "role": "assistant",
                        "content": self._format_function_call(function_call),
                        "is_function_call": True
                    })
                    
                    # Add a simulated function result
                    function_result = {
                        "result": f"Simulated result for {function_schema.name}",
                        "status": "success"
                    }
                    
                    modified_conversation.append({
                        "role": "function",
                        "content": self._format_function_result(function_result),
                        "name": function_schema.name
                    })
                    
                    # Add a follow-up assistant response
                    modified_conversation.append({
                        "role": "assistant",
                        "content": f"Based on the {function_schema.name} result, I can provide an answer."
                    })
                    
                elif role == "assistant" and random.random() < self.json_format_probability:
                    # Format this response as JSON
                    modified_conversation.append({
                        "role": role,
                        "content": self._format_json_response(content)
                    })
                    
                else:
                    # Keep the original turn
                    modified_conversation.append(turn)
            
            # Add to function examples if it contains function calls
            if any(turn.get("is_function_call", False) for turn in modified_conversation):
                function_examples.append(modified_conversation)
            elif any(self.function_tokens["json_response"] in turn.get("content", "") 
                    for turn in modified_conversation):
                function_examples.append(modified_conversation)
        
        # Add function examples to conversations
        self.conversations.extend(function_examples)
        
        # Regenerate examples with the new conversations
        self.examples = self._prepare_examples()


def create_function_calling_dataset(
    tokenizer,
    function_definitions=None,
    data_path=None,
    dataset_name="daily_dialog",
    split="train",
    max_length=1024,
    speaker_tokens=None,
    system_prompt=None,
    cache_dir=None,
    json_format_probability=0.5
):
    """
    Create a dataset for training function calling capabilities.
    
    Args:
        tokenizer: Tokenizer to use for preprocessing
        function_definitions: List of function definitions
        data_path: Path to custom dialogue data
        dataset_name: Name of the HF dataset to load
        split: Dataset split
        max_length: Maximum sequence length
        speaker_tokens: Dictionary mapping speaker roles to tokens
        system_prompt: Optional system prompt
        cache_dir: Directory for caching the dataset
        json_format_probability: Probability of JSON formatting
        
    Returns:
        FunctionCallingDataset instance
    """
    # Default function definitions if none provided
    if function_definitions is None:
        function_definitions = get_default_function_definitions()
    
    # Create dataset
    dataset = FunctionCallingDataset(
        tokenizer=tokenizer,
        data_path=data_path,
        dataset_name=dataset_name,
        split=split,
        max_length=max_length,
        speaker_tokens=speaker_tokens,
        system_prompt=system_prompt,
        cache_dir=cache_dir,
        function_definitions=function_definitions,
        json_format_probability=json_format_probability
    )
    
    return dataset


def get_default_function_definitions():
    """
    Get a set of default function definitions for training.
    
    Returns:
        List of function definitions
    """
    return [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "location": {
                    "type": "string",
                    "description": "The city and state or country (e.g., 'San Francisco, CA')"
                },
                "unit": {
                    "type": "string",
                    "description": "The unit of temperature, either 'celsius' or 'fahrenheit'",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        },
        {
            "name": "search_knowledge_base",
            "description": "Search the knowledge base for information",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return"
                }
            },
            "required": ["query"]
        },
        {
            "name": "create_calendar_event",
            "description": "Create a calendar event",
            "parameters": {
                "title": {
                    "type": "string",
                    "description": "The title of the event"
                },
                "start_time": {
                    "type": "string",
                    "description": "The start time of the event (ISO format)"
                },
                "end_time": {
                    "type": "string",
                    "description": "The end time of the event (ISO format)"
                },
                "description": {
                    "type": "string",
                    "description": "Description of the event"
                },
                "attendees": {
                    "type": "array",
                    "description": "List of attendees' email addresses",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["title", "start_time"]
        },
        {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "operation": {
                    "type": "string",
                    "description": "The mathematical operation to perform",
                    "enum": ["add", "subtract", "multiply", "divide", "power"]
                },
                "a": {
                    "type": "number",
                    "description": "The first operand"
                },
                "b": {
                    "type": "number",
                    "description": "The second operand"
                }
            },
            "required": ["operation", "a", "b"]
        }
    ]


def create_function_calling_dataloader(
    tokenizer,
    function_definitions=None,
    data_path=None,
    dataset_name="daily_dialog",
    split="train",
    batch_size=8,
    max_length=1024,
    speaker_tokens=None,
    system_prompt=None,
    shuffle=True,
    num_workers=4,
    cache_dir=None,
    collate_fn=None,
    json_format_probability=0.5
):
    """
    Create a DataLoader for function calling training.
    
    Args:
        See create_function_calling_dataset for parameters
        
    Returns:
        DataLoader for function calling dataset
    """
    # Create dataset
    dataset = create_function_calling_dataset(
        tokenizer=tokenizer,
        function_definitions=function_definitions,
        data_path=data_path,
        dataset_name=dataset_name,
        split=split,
        max_length=max_length,
        speaker_tokens=speaker_tokens,
        system_prompt=system_prompt,
        cache_dir=cache_dir,
        json_format_probability=json_format_probability
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn or dialogue_collate_fn
    )
    
    return dataloader


def get_function_calling_dataloaders(
    tokenizer,
    function_definitions=None,
    data_path=None,
    dataset_name="daily_dialog",
    batch_size=8,
    max_length=1024,
    speaker_tokens=None,
    system_prompt=None,
    num_workers=4,
    cache_dir=None,
    collate_fn=None,
    json_format_probability=0.5
):
    """
    Get DataLoaders for training, validation, and test for function calling.
    
    Args:
        See create_function_calling_dataloader for parameters
        
    Returns:
        Dict of DataLoaders for train, validation, and test splits
    """
    # Training dataloader
    train_loader = create_function_calling_dataloader(
        tokenizer=tokenizer,
        function_definitions=function_definitions,
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
        json_format_probability=json_format_probability
    )
    
    # Validation dataloader
    val_loader = create_function_calling_dataloader(
        tokenizer=tokenizer,
        function_definitions=function_definitions,
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
        collate_fn=collate_fn,
        json_format_probability=json_format_probability
    )
    
    # Test dataloader
    try:
        test_loader = create_function_calling_dataloader(
            tokenizer=tokenizer,
            function_definitions=function_definitions,
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
            collate_fn=collate_fn,
            json_format_probability=json_format_probability
        )
    except Exception:
        # Fall back to validation set if test set is not available
        test_loader = val_loader
    
    return {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader
    }