"""
Configuration wizard for the QLLM CLI.

This module provides a step-by-step wizard for configuring training parameters,
guiding users through the configuration process with validation and help text.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional, Union

from src.cli.user_interface import TerminalUI
from src.config.config_manager import ConfigManager
from src.config.config_schema import get_schema


class ConfigWizard:
    """Interactive wizard for configuring training parameters."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the configuration wizard.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.schema = get_schema()
        self.ui = TerminalUI()
        
    def run_wizard(
        self, 
        initial_config: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run the configuration wizard.
        
        Args:
            initial_config: Initial configuration to start with
            
        Returns:
            The configured parameters
        """
        # Start with default or provided configuration
        config = initial_config
        if config is None:
            config = self.config_manager.create_default_config()
        
        # Display welcome screen
        self.ui.clear_screen()
        self.ui.print_header("QLLM Configuration Wizard")
        self.ui.print_info(
            "This wizard will guide you through configuring the Quantum Resonance "
            "Language Model training parameters. You can navigate options with arrow keys "
            "and accept defaults by pressing Enter."
        )
        self.ui.wait_for_any_key()
        
        # Select configuration sections to edit
        while True:
            self.ui.clear_screen()
            self.ui.print_header("Configuration Sections")
            
            sections = ["model", "training", "data"]
            section_names = ["Model Configuration", "Training Configuration", "Data Configuration"]
            options = section_names + ["Review Configuration", "Save Configuration", "Exit Wizard"]
            
            choice = self.ui.menu(
                "Select a section to configure:",
                options
            )
            
            if choice < len(sections):
                # Edit selected section
                section = sections[choice]
                config[section] = self.configure_section(section, config[section])
                
                # Automatically update training_type based on dataset selection
                if section == "data" and config["data"].get("dataset_name") == "daily_dialog":
                    config["training"]["training_type"] = "dialogue"
                    self.ui.print_success("Training type automatically set to 'dialogue' for Daily Dialog dataset")
                    self.ui.wait_for_any_key()
                
            elif choice == len(sections):
                # Review configuration
                self.review_configuration(config)
            elif choice == len(sections) + 1:
                # Save configuration
                saved = self.save_configuration(config)
                if saved:
                    return config
            else:
                # Exit wizard
                if self.ui.prompt_bool("Exit without saving?", default=False):
                    break
        
        return config
    
    def configure_section(
        self,
        section: str,
        section_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Configure a specific section.
        
        Args:
            section: Section name (model, training, data)
            section_config: Current section configuration
            
        Returns:
            Updated section configuration
        """
        # Create a working copy of the configuration
        config = section_config.copy()
        
        # Get schema for this section
        section_schema = self.schema.schema.get(section, {})
        
        # Display section header
        self.ui.clear_screen()
        self.ui.print_header(f"{section.title()} Configuration")
        
        # Special handling for dataset selection in data section
        if section == "data":
            return self._configure_dataset_section(config)
        
        # Organize parameters into categories (basic, advanced)
        basic_params = []
        advanced_params = []
        
        for param_name in section_schema:
            # Skip "extra" fields that collect arbitrary parameters
            if param_name.startswith('extra_') or param_name.endswith('_params'):
                continue
                
            # Simple heuristic for categorizing parameters
            if param_name in [
                # Model basic params
                "hidden_dim", "num_layers", "num_heads", "dropout", "max_seq_length",
                # Training basic params
                "batch_size", "learning_rate", "max_epochs", "training_type",
                # Data basic params
                "dataset_name", "tokenizer_name", "max_length"
            ]:
                basic_params.append(param_name)
            else:
                advanced_params.append(param_name)
        
        # Configure basic parameters
        self.ui.print_section("Basic Parameters")
        for param in basic_params:
            config[param] = self._configure_parameter(section, param, config.get(param))
        
        # Configure advanced parameters
        configure_advanced = self.ui.prompt_bool(
            "Configure advanced parameters?",
            default=False
        )
        
        if configure_advanced:
            self.ui.print_section("Advanced Parameters")
            for param in advanced_params:
                config[param] = self._configure_parameter(section, param, config.get(param))
        
        return config
    
    def _configure_dataset_section(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure the dataset section with special handling for dataset types.
        
        Args:
            config: Current data configuration
            
        Returns:
            Updated data configuration
        """
        self.ui.print_section("Dataset Selection")
        
        # Select dataset type
        dataset_options = ["wikitext", "daily_dialog", "custom"]
        dataset_descriptions = [
            "WikiText (language modeling corpus)",
            "Daily Dialog (conversation dataset)",
            "Custom dataset (from local files)"
        ]
        
        self.ui.print_info("Select a dataset type:")
        for i, desc in enumerate(dataset_descriptions):
            self.ui.print_info(f"  {i+1}. {desc}")
        print()
        
        current_idx = dataset_options.index(config.get("dataset_name", "wikitext")) if config.get("dataset_name") in dataset_options else 0
        choice = self.ui.prompt_choice("Dataset type", dataset_options, default=current_idx)
        config["dataset_name"] = dataset_options[choice]
        
        # Dataset-specific configuration
        if config["dataset_name"] == "wikitext":
            # WikiText variant
            variant_options = ["wikitext-103-v1", "wikitext-2-v1"]
            current_variant_idx = variant_options.index(config.get("dataset_variant", "wikitext-103-v1")) if config.get("dataset_variant") in variant_options else 0
            variant_choice = self.ui.prompt_choice("WikiText variant", variant_options, default=current_variant_idx)
            config["dataset_variant"] = variant_options[variant_choice]
            
            # Other WikiText settings
            config["tokenizer_name"] = self.ui.prompt("Tokenizer name", default=config.get("tokenizer_name", "gpt2"))
            config["max_length"] = self.ui.prompt_int("Max sequence length", default=config.get("max_length", 512), min_value=64, max_value=4096)
            config["stride"] = self.ui.prompt_int("Stride length", default=config.get("stride", 256), min_value=32, max_value=2048)
            
        elif config["dataset_name"] == "daily_dialog":
            # Daily Dialog settings
            config["tokenizer_name"] = self.ui.prompt("Tokenizer name", default=config.get("tokenizer_name", "gpt2"))
            config["system_prompt"] = self.ui.prompt("System prompt", default=config.get("system_prompt", "You are a helpful assistant."))
            config["max_length"] = self.ui.prompt_int("Max sequence length", default=config.get("max_length", 512), min_value=64, max_value=4096)
            
        elif config["dataset_name"] == "custom":
            # Custom dataset settings
            config["train_file"] = self.ui.prompt("Training data file path", default=config.get("train_file", ""))
            config["validation_file"] = self.ui.prompt("Validation data file path", default=config.get("validation_file", ""))
            config["tokenizer_name"] = self.ui.prompt("Tokenizer name", default=config.get("tokenizer_name", "gpt2"))
            config["max_length"] = self.ui.prompt_int("Max sequence length", default=config.get("max_length", 512), min_value=64, max_value=4096)
            
            # Check if it's dialogue data
            is_dialogue = self.ui.prompt_bool("Is this a dialogue dataset?", default=False)
            if is_dialogue:
                config["system_prompt"] = self.ui.prompt("System prompt", default=config.get("system_prompt", "You are a helpful assistant."))
        
        # Common settings
        self.ui.print_section("Advanced Data Parameters")
        
        # Allow advanced configuration
        if self.ui.prompt_bool("Configure advanced data parameters?", default=False):
            config["preprocessing_num_workers"] = self.ui.prompt_int(
                "Preprocessing workers",
                default=config.get("preprocessing_num_workers", 4),
                min_value=1,
                max_value=32
            )
            
            config["cache_dir"] = self.ui.prompt(
                "Cache directory",
                default=config.get("cache_dir", ".cache")
            )
            
            config["subset_size"] = self.ui.prompt_int(
                "Subset size (0 for full dataset)",
                default=config.get("subset_size", 0),
                min_value=0
            )
            if config["subset_size"] == 0:
                config["subset_size"] = None
        
        return config
    
    def _configure_parameter(
        self,
        section: str,
        param: str,
        current_value: Any
    ) -> Any:
        """
        Configure a single parameter.
        
        Args:
            section: Configuration section
            param: Parameter name
            current_value: Current parameter value
            
        Returns:
            Updated parameter value
        """
        # Get parameter schema
        param_schema = self.schema.schema.get(section, {}).get(param, {})
        
        if not param_schema:
            return current_value
        
        # Get parameter metadata
        param_type = param_schema.get("type", "str")
        help_text = param_schema.get("help", f"{param} parameter")
        default = param_schema.get("default")
        
        # Display parameter information
        print(f"\n{param}:")
        self.ui.print_info(f"  {help_text}")
        self.ui.print_info(f"  Current value: {current_value}")
        
        # Handle different parameter types
        if param_type == "int":
            min_val, max_val = param_schema.get("range", [None, None])
            range_info = ""
            if min_val is not None and max_val is not None:
                range_info = f" (range: {min_val}-{max_val})"
            elif min_val is not None:
                range_info = f" (min: {min_val})"
            elif max_val is not None:
                range_info = f" (max: {max_val})"
                
            return self.ui.prompt_int(
                f"Enter value for {param}{range_info}",
                default=current_value,
                min_value=min_val,
                max_value=max_val
            )
            
        elif param_type == "float":
            min_val, max_val = param_schema.get("range", [None, None])
            range_info = ""
            if min_val is not None and max_val is not None:
                range_info = f" (range: {min_val}-{max_val})"
            elif min_val is not None:
                range_info = f" (min: {min_val})"
            elif max_val is not None:
                range_info = f" (max: {max_val})"
                
            return self.ui.prompt_float(
                f"Enter value for {param}{range_info}",
                default=current_value,
                min_value=min_val,
                max_value=max_val
            )
            
        elif param_type == "bool":
            return self.ui.prompt_bool(
                f"Enable {param}?",
                default=current_value
            )
            
        elif param_type == "str":
            choices = param_schema.get("choices", [])
            if choices:
                idx = self.ui.prompt_choice(
                    f"Select {param}:",
                    choices
                )
                return choices[idx]
            else:
                return self.ui.prompt(
                    f"Enter value for {param}",
                    default=current_value or ""
                )
                
        elif param_type == "list":
            if param == "primes":  # Special handling for primes
                return self.ui.prompt_list(
                    f"Enter comma-separated list of prime numbers for {param}",
                    default=current_value,
                    item_type="int"
                )
            else:
                item_type = "str"
                if all(isinstance(x, int) for x in current_value or []):
                    item_type = "int"
                elif all(isinstance(x, float) for x in current_value or []):
                    item_type = "float"
                    
                return self.ui.prompt_list(
                    f"Enter comma-separated values for {param}",
                    default=current_value,
                    item_type=item_type
                )
                
        else:
            # Fall back to string for unknown types
            return self.ui.prompt(
                f"Enter value for {param}",
                default=str(current_value) if current_value is not None else ""
            )
    
    def review_configuration(self, config: Dict[str, Dict[str, Any]]) -> None:
        """
        Display the current configuration for review.
        
        Args:
            config: Complete configuration
        """
        self.ui.clear_screen()
        self.ui.print_header("Configuration Review")
        
        for section, section_config in config.items():
            self.ui.print_section(f"{section.title()} Configuration")
            
            # Get schema for this section
            section_schema = self.schema.schema.get(section, {})
            
            # Convert section config to a table-friendly format
            review_data = {}
            for param, value in sorted(section_config.items()):
                # Skip "extra" fields
                if param.startswith('extra_') or param.endswith('_params'):
                    continue
                    
                # Get parameter info
                param_schema = section_schema.get(param, {})
                
                # Format value for display
                if isinstance(value, list):
                    display_value = ", ".join(map(str, value))
                else:
                    display_value = str(value)
                
                # Add to review data
                review_data[param] = display_value
            
            # Display as a table
            self.ui.print_values_table(review_data)
            print()
        
        # Wait for keypress before returning
        self.ui.wait_for_any_key("Press any key to return to the menu...")
    
    def save_configuration(self, config: Dict[str, Dict[str, Any]]) -> bool:
        """
        Save the configuration to a file.
        
        Args:
            config: Configuration to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        self.ui.clear_screen()
        self.ui.print_header("Save Configuration")
        
        # Validate configuration
        errors = self.schema.validate(config)
        if errors:
            self.ui.print_error("Configuration validation failed:")
            for error in errors:
                self.ui.print_info(f"- {error}")
            
            proceed = self.ui.prompt_bool(
                "Continue with saving despite validation errors?",
                default=False
            )
            
            if not proceed:
                self.ui.print_info("Aborting save operation.")
                self.ui.wait_for_any_key()
                return False
        
        # Get save path
        default_path = "configs"
        os.makedirs(default_path, exist_ok=True)
        
        config_name = self.ui.prompt(
            "Enter configuration name (without extension)",
            default="my_config"
        )
        
        file_path = os.path.join(default_path, f"{config_name}.json")
        
        # Check for existing file
        if os.path.exists(file_path):
            overwrite = self.ui.prompt_bool(
                f"Configuration file {file_path} already exists. Overwrite?",
                default=False
            )
            
            if not overwrite:
                self.ui.print_info("Aborting save operation.")
                self.ui.wait_for_any_key()
                return False
        
        # Save configuration
        try:
            self.config_manager.save_config(config, file_path)
            self.ui.print_success(f"Configuration saved to {file_path}")
            self.ui.wait_for_any_key()
            return True
        except Exception as e:
            self.ui.print_error(f"Error saving configuration: {e}")
            self.ui.wait_for_any_key()
            return False
    
    def load_configuration(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Load a configuration from a file.
        
        Returns:
            Loaded configuration or None if cancelled
        """
        self.ui.clear_screen()
        self.ui.print_header("Load Configuration")
        
        # Find configuration files
        configs_dir = "configs"
        if not os.path.exists(configs_dir):
            self.ui.print_info("No configurations found.")
            os.makedirs(configs_dir, exist_ok=True)
            self.ui.print_info(f"Created directory: {configs_dir}")
            self.ui.wait_for_any_key()
            return None
        
        config_files = [f for f in os.listdir(configs_dir) if f.endswith('.json')]
        
        if not config_files:
            self.ui.print_info("No configuration files found.")
            self.ui.wait_for_any_key()
            return None
        
        # Let user select a configuration file
        self.ui.print_info("Available configurations:")
        
        options = [f.replace('.json', '') for f in config_files]
        options.append("Cancel")
        
        choice = self.ui.menu("Select a configuration to load:", options)
        
        if choice >= len(config_files):
            return None
        
        # Load the selected configuration
        file_path = os.path.join(configs_dir, config_files[choice])
        
        try:
            config = self.config_manager.load_config(file_path)
            self.ui.print_success(f"Configuration loaded from {file_path}")
            
            # Validate configuration
            errors = self.schema.validate(config)
            if errors:
                self.ui.print_warning("Configuration validation warnings:")
                for error in errors:
                    self.ui.print_info(f"- {error}")
            
            self.ui.wait_for_any_key()
            return config
        except Exception as e:
            self.ui.print_error(f"Error loading configuration: {e}")
            self.ui.wait_for_any_key()
            return None