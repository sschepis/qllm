#!/usr/bin/env python3
"""
Test script for the QLLM configuration system.

This script creates a sample configuration with extensions enabled,
validates it, and saves it to a file.
"""

import os
import json
from src.config.config_manager import ConfigManager


def main():
    """Test the configuration system."""
    print("Testing QLLM Configuration System")
    print("=================================")
    
    # Create configuration manager
    config_manager = ConfigManager()
    
    # Create a default configuration
    config = config_manager.create_default_config()
    
    # Enable extensions
    if "extensions" not in config["model"]:
        config["model"]["extensions"] = {}
    
    config["model"]["extensions"]["extensions_enabled"] = True
    config["model"]["extensions"]["enable_memory"] = True
    config["model"]["extensions"]["enable_quantum"] = True
    
    # Configure memory extension
    if "memory_config" not in config["model"]["extensions"]:
        config["model"]["extensions"]["memory_config"] = {}
    
    config["model"]["extensions"]["memory_config"]["memory_size"] = 2000
    config["model"]["extensions"]["memory_config"]["entity_dim"] = 512
    
    # Configure quantum extension
    if "quantum_config" not in config["model"]["extensions"]:
        config["model"]["extensions"]["quantum_config"] = {}
    
    config["model"]["extensions"]["quantum_config"]["pattern_type"] = "prime"
    config["model"]["extensions"]["quantum_config"]["base_sparsity"] = 0.7
    
    # Set some training parameters
    config["training"]["batch_size"] = 32
    config["training"]["learning_rate"] = 3e-5
    config["training"]["training_type"] = "dialogue"
    
    # Set some data parameters
    config["data"]["dataset_name"] = "daily_dialog"
    config["data"]["max_length"] = 256
    
    # Validate configuration
    print("\nValidating configuration...")
    errors = config_manager.validate_config(config)
    
    if errors:
        print("Validation failed with errors:")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    print("Configuration validation passed!")
    
    # Create a test config directory
    test_dir = "test_configs"
    os.makedirs(test_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(test_dir, "test_config.json")
    try:
        config_manager.save_config(config, config_path)
        print(f"\nConfiguration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return 1
    
    # Load configuration back
    try:
        loaded_config = config_manager.load_config(config_path)
        print("\nConfiguration loaded successfully!")
        
        # Convert to config classes
        config_classes = config_manager.to_config_classes(loaded_config)
        
        # Print some info
        print("\nModel configuration:")
        print(f"  Hidden dimension: {config_classes['model'].hidden_dim}")
        print(f"  Layers: {config_classes['model'].num_layers}")
        
        print("\nExtensions:")
        extensions = config_classes["model"].extensions
        print(f"  Extensions enabled: {extensions.get('extensions_enabled', False)}")
        print(f"  Memory enabled: {extensions.get('enable_memory', False)}")
        print(f"  Quantum enabled: {extensions.get('enable_quantum', False)}")
        
        if extensions.get("enable_memory", False):
            memory_config = extensions.get("memory_config", {})
            print("\nMemory configuration:")
            print(f"  Memory size: {memory_config.get('memory_size', 'N/A')}")
            print(f"  Entity dimension: {memory_config.get('entity_dim', 'N/A')}")
        
        if extensions.get("enable_quantum", False):
            quantum_config = extensions.get("quantum_config", {})
            print("\nQuantum configuration:")
            print(f"  Pattern type: {quantum_config.get('pattern_type', 'N/A')}")
            print(f"  Base sparsity: {quantum_config.get('base_sparsity', 'N/A')}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    print("\nTest completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())