#!/usr/bin/env python3
"""
Interactive chat interface for Semantic Resonance Language Model.

This script provides a simple command-line chat interface for
interacting with a trained model.
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer

# Add the repository root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ModelConfig
from src.model.semantic_resonance_model import SemanticResonanceModel
from src.training.checkpoint import find_latest_checkpoint, load_checkpoint
from src.utils.config import load_dataclass_from_json
from src.utils.device import get_device


def create_parser():
    """Create argument parser for the chat interface."""
    parser = argparse.ArgumentParser(
        description="Interactive chat interface for Semantic Resonance Language Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the model directory or checkpoint file")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to model configuration file")
    parser.add_argument("--tokenizer_path", type=str, default="gpt2",
                        help="Path to tokenizer or name of pretrained tokenizer")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum generation length")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling parameter")
    parser.add_argument("--history_size", type=int, default=5,
                        help="Number of past exchanges to keep in context")
    
    return parser


def load_model(args):
    """Load the model and tokenizer."""
    # Determine device
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model config
    if args.config_path and os.path.exists(args.config_path):
        print(f"Loading model configuration from {args.config_path}")
        model_config = load_dataclass_from_json(ModelConfig, args.config_path)
    else:
        # Try to find config in model path
        config_path = None
        if args.model_path:
            potential_config = os.path.join(args.model_path, "model_config.json")
            if os.path.exists(potential_config):
                config_path = potential_config
        
        if config_path:
            print(f"Loading model configuration from {config_path}")
            model_config = load_dataclass_from_json(ModelConfig, config_path)
        else:
            print("Using default model configuration")
            model_config = ModelConfig()
    
    # Update vocab size
    model_config.vocab_size = len(tokenizer)
    
    # Initialize model
    print("Initializing model...")
    model = SemanticResonanceModel(model_config)
    
    # Load checkpoint if available
    if args.model_path:
        checkpoint_path = args.model_path
        
        # Check if it's a directory and find latest checkpoint
        if os.path.isdir(checkpoint_path):
            latest_checkpoint = find_latest_checkpoint(checkpoint_path)
            if latest_checkpoint:
                checkpoint_path = latest_checkpoint
        
        if os.path.exists(checkpoint_path):
            print(f"Loading model from checkpoint: {checkpoint_path}")
            load_checkpoint(model, checkpoint_path, map_location=device)
        else:
            print(f"Checkpoint not found at {checkpoint_path}, using random initialization")
    
    model.to(device)
    model.eval()
    
    return model, tokenizer, device


def format_conversation(conversation, system_prompt=None):
    """Format the conversation history into a prompt."""
    formatted = ""
    
    # Add system prompt if provided
    if system_prompt:
        formatted += f"System: {system_prompt}\n\n"
    
    # Add conversation history
    for entry in conversation:
        if entry["role"] == "user":
            formatted += f"User: {entry['content']}\n"
        else:
            formatted += f"Assistant: {entry['content']}\n"
    
    # Add final prompt for assistant response
    formatted += "Assistant:"
    
    return formatted


def run_chat_interface(args):
    """Run the interactive chat interface."""
    model, tokenizer, device = load_model(args)
    
    # Initialize conversation history
    conversation = []
    system_prompt = "You are a helpful AI assistant based on the Semantic Resonance Language Model."
    
    print("\n" + "="*60)
    print("Semantic Resonance Language Model - Chat Interface")
    print("="*60)
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("Type 'clear' to clear the conversation history.")
    print("Type '/system <prompt>' to change the system prompt.")
    print("="*60 + "\n")
    
    print(f"System: {system_prompt}\n")
    
    while True:
        # Get user input
        user_input = input("User: ").strip()
        
        # Check for special commands
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\nGoodbye!")
            break
        elif user_input.lower() == "clear":
            conversation = []
            print("\nConversation history cleared.\n")
            continue
        elif user_input.lower().startswith("/system "):
            system_prompt = user_input[8:].strip()
            print(f"\nSystem prompt updated: {system_prompt}\n")
            continue
        
        # Add user message to conversation
        conversation.append({"role": "user", "content": user_input})
        
        # Limit conversation history based on history_size
        if len(conversation) > args.history_size * 2:
            # Keep the most recent exchanges, but always keep the latest user message
            conversation = conversation[-(args.history_size * 2):]
        
        # Format conversation into a prompt
        prompt = format_conversation(conversation, system_prompt)
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate response
        print("Assistant: ", end="", flush=True)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + args.max_length,
                temperature=args.temperature,
                do_sample=True,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode the response
        generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Clean up the response (stop at certain tokens)
        cutoff_tokens = ["User:", "System:", "\n\n"]
        for token in cutoff_tokens:
            if token in generated_text:
                generated_text = generated_text.split(token)[0]
        
        print(generated_text.strip())
        
        # Add assistant response to conversation
        conversation.append({"role": "assistant", "content": generated_text.strip()})
        print()  # Add empty line for better readability


def main():
    """Main function to run the chat interface."""
    parser = create_parser()
    args = parser.parse_args()
    run_chat_interface(args)


if __name__ == "__main__":
    main()