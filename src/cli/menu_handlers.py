"""
Menu handlers for the QLLM CLI.

This module contains handlers for menu options, which connect the menu system
to the actual functionality (training, evaluation, etc.).
"""

import os
import sys
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from src.cli.menu_system import Menu, MenuOption
from src.cli.user_interface import TerminalUI
from src.cli.config_wizard import ConfigWizard
from src.config.config_manager import ConfigManager


class MenuHandler:
    """Handlers for menu options."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the menu handler.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.config_wizard = ConfigWizard(self.config_manager)
        self.ui = TerminalUI()
        self.current_config = None
        
        # For progress monitoring
        self._progress_message = None
        self._progress_steps = {}
        
    def build_main_menu(self) -> Menu:
        """
        Build the main menu structure.
        
        Returns:
            Main menu
        """
        main_menu = Menu(title="QLLM - Quantum Resonance Language Model")
        
        # Add main menu options
        main_menu.add_option(MenuOption(
            text="Training",
            handler=self.handle_training_menu
        ))
        
        main_menu.add_option(MenuOption(
            text="Evaluation",
            handler=self.handle_evaluation_menu
        ))
        
        main_menu.add_option(MenuOption(
            text="Generation",
            handler=self.handle_generation_menu
        ))
        
        main_menu.add_option(MenuOption(
            text="Extensions",
            handler=self.handle_extensions_menu
        ))
        
        main_menu.add_option(MenuOption(
            text="Compression",
            handler=self.handle_compression_menu
        ))
        
        main_menu.add_option(MenuOption(
            text="Exit",
            should_exit=True
        ))
        
        return main_menu
    
    def handle_training_menu(self) -> Menu:
        """
        Handle the training menu.
        
        Returns:
            Training menu
        """
        training_menu = Menu(
            title="Training Menu",
            parent=None  # Will be set by the menu system
        )
        
        # Add training menu options
        training_menu.add_option(MenuOption(
            text="Configure Training",
            handler=self.handle_configure_menu
        ))
        
        training_menu.add_option(MenuOption(
            text="Load Configuration",
            handler=self.handle_load_configuration
        ))
        
        training_menu.add_option(MenuOption(
            text="Start Training",
            handler=self.handle_start_training,
            is_enabled=lambda: self.current_config is not None
        ))
        
        training_menu.add_option(MenuOption(
            text="Resume Training",
            handler=self.handle_resume_training
        ))
        
        training_menu.add_option(MenuOption(
            text="Back to Main Menu",
            should_exit=True
        ))
        
        return training_menu
    
    def handle_extensions_menu(self) -> Menu:
        """
        Handle the extensions menu.
        
        Returns:
            Extensions menu
        """
        extensions_menu = Menu(
            title="Extensions Menu",
            parent=None  # Will be set by the menu system
        )
        
        # Add extensions menu options
        extensions_menu.add_option(MenuOption(
            text="Configure Extensions",
            handler=self.handle_configure_extensions
        ))
        
        extensions_menu.add_option(MenuOption(
            text="Memory Extensions",
            handler=self.handle_memory_extensions
        ))
        
        extensions_menu.add_option(MenuOption(
            text="Multimodal Extensions",
            handler=self.handle_multimodal_extensions
        ))
        
        extensions_menu.add_option(MenuOption(
            text="Quantum Extensions",
            handler=self.handle_quantum_extensions
        ))
        
        extensions_menu.add_option(MenuOption(
            text="List Enabled Extensions",
            handler=self.handle_list_extensions
        ))
        
        extensions_menu.add_option(MenuOption(
            text="Back to Main Menu",
            should_exit=True
        ))
        
        return extensions_menu
    
    def handle_evaluation_menu(self) -> Menu:
        """
        Handle the evaluation menu.
        
        Returns:
            Evaluation menu
        """
        eval_menu = Menu(
            title="Evaluation Menu",
            parent=None  # Will be set by the menu system
        )
        
        # Add evaluation menu options
        eval_menu.add_option(MenuOption(
            text="Configure Evaluation",
            handler=self.handle_configure_evaluation
        ))
        
        eval_menu.add_option(MenuOption(
            text="Load Model",
            handler=self.handle_load_model
        ))
        
        eval_menu.add_option(MenuOption(
            text="Run Evaluation",
            handler=self.handle_run_evaluation
        ))
        
        eval_menu.add_option(MenuOption(
            text="Visualize Results",
            handler=self.handle_visualize_results
        ))
        
        eval_menu.add_option(MenuOption(
            text="Back to Main Menu",
            should_exit=True
        ))
        
        return eval_menu
    
    def handle_generation_menu(self) -> Menu:
        """
        Handle the generation menu.
        
        Returns:
            Generation menu
        """
        gen_menu = Menu(
            title="Generation Menu",
            parent=None  # Will be set by the menu system
        )
        
        # Add generation menu options
        gen_menu.add_option(MenuOption(
            text="Configure Generation",
            handler=self.handle_configure_generation
        ))
        
        gen_menu.add_option(MenuOption(
            text="Load Model",
            handler=self.handle_load_model
        ))
        
        gen_menu.add_option(MenuOption(
            text="Interactive Generation",
            handler=self.handle_interactive_generation
        ))
        
        gen_menu.add_option(MenuOption(
            text="Batch Generation",
            handler=self.handle_batch_generation
        ))
        
        gen_menu.add_option(MenuOption(
            text="Back to Main Menu",
            should_exit=True
        ))
        
        return gen_menu
    
    def handle_compression_menu(self) -> Menu:
        """
        Handle the compression menu.
        
        Returns:
            Compression menu
        """
        comp_menu = Menu(
            title="Compression Menu",
            parent=None  # Will be set by the menu system
        )
        
        # Add compression menu options
        comp_menu.add_option(MenuOption(
            text="Configure Compression",
            handler=self.handle_configure_compression
        ))
        
        comp_menu.add_option(MenuOption(
            text="Load Model",
            handler=self.handle_load_model
        ))
        
        comp_menu.add_option(MenuOption(
            text="Compress Model",
            handler=self.handle_compress_model
        ))
        
        comp_menu.add_option(MenuOption(
            text="Compare Before/After",
            handler=self.handle_compare_compression
        ))
        
        comp_menu.add_option(MenuOption(
            text="Back to Main Menu",
            should_exit=True
        ))
        
        return comp_menu
    
    def handle_configure_menu(self) -> Menu:
        """
        Handle the configuration menu.
        
        Returns:
            Configuration menu
        """
        # Create a new configuration if none exists
        if self.current_config is None:
            self.current_config = self.config_manager.create_default_config()
        
        # Run the configuration wizard
        self.current_config = self.config_wizard.run_wizard(self.current_config)
        
        # Return to the training menu
        return None  # Stay in current menu
    
    def handle_load_configuration(self) -> None:
        """Handle loading a configuration."""
        config = self.config_wizard.load_configuration()
        if config:
            self.current_config = config
            self.ui.print_success("Configuration loaded successfully.")
        else:
            self.ui.print_info("Configuration loading cancelled or failed.")
        
        self.ui.wait_for_any_key()
    
    def handle_start_training(self) -> None:
        """Handle starting a new training session."""
        if self.current_config is None:
            self.ui.print_error("No configuration loaded. Please configure or load a configuration first.")
            self.ui.wait_for_any_key()
            return
        
        # Display training configuration summary
        self.ui.clear_screen()
        self.ui.print_header("Start Training")
        
        # Get training type
        training_type = self.current_config["training"].get("training_type", "standard")
        
        self.ui.print_info(f"Training Type: {training_type}")
        self.ui.print_info(f"Model:")
        self.ui.print_info(f"  Hidden Dimension: {self.current_config['model']['hidden_dim']}")
        self.ui.print_info(f"  Layers: {self.current_config['model']['num_layers']}")
        self.ui.print_info(f"  Heads: {self.current_config['model']['num_heads']}")
        
        self.ui.print_info(f"Training:")
        self.ui.print_info(f"  Batch Size: {self.current_config['training']['batch_size']}")
        self.ui.print_info(f"  Learning Rate: {self.current_config['training']['learning_rate']}")
        self.ui.print_info(f"  Max Epochs: {self.current_config['training']['max_epochs']}")
        
        # Display dataset information
        dataset_name = self.current_config["data"].get("dataset_name", "wikitext")
        self.ui.print_info(f"Dataset: {dataset_name}")
        
        if dataset_name == "daily_dialog":
            system_prompt = self.current_config["data"].get("system_prompt", "")
            if system_prompt:
                self.ui.print_info(f"  System Prompt: \"{system_prompt[:50]}...\"" if len(system_prompt) > 50 else f"  System Prompt: \"{system_prompt}\"")
        elif dataset_name == "custom":
            self.ui.print_info(f"  Training File: {self.current_config['data'].get('train_file', 'Not specified')}")
            self.ui.print_info(f"  Validation File: {self.current_config['data'].get('validation_file', 'Not specified')}")
        
        # Check for enabled extensions
        extensions_enabled = False
        if "model" in self.current_config and "extensions" in self.current_config["model"]:
            extensions_config = self.current_config["model"]["extensions"]
            if extensions_config.get("extensions_enabled", False):
                extensions_enabled = True
                self.ui.print_info("Extensions:")
                if extensions_config.get("enable_memory", False):
                    self.ui.print_info("  Memory: Enabled")
                if extensions_config.get("enable_multimodal", False):
                    self.ui.print_info("  Multimodal: Enabled")
                if extensions_config.get("enable_quantum", False):
                    self.ui.print_info("  Quantum: Enabled")
        
        # Check if user wants to proceed
        proceed = self.ui.prompt_bool("Start training with these settings?", default=True)
        if not proceed:
            return
        
        # Create trainer
        try:
            # Convert dict config to config classes
            config_classes = self.config_manager.to_config_classes(self.current_config)
            
            # Import TrainerFactory here to avoid circular imports
            from src.training import TrainerFactory
            
            # Create appropriate trainer
            trainer_factory = TrainerFactory()
            
            # Clear screen and show initialization message
            self.ui.clear_screen()
            self.ui.print_header("QLLM Training")
            
            # Initialize training components with progress indicators
            self.ui.print_section("Initialization")
            
            # Display dataset type message
            dataset_type = config_classes["data"].dataset_name
            self.ui.print_info(f"Preparing for {dataset_type} dataset training...")
            
            # Reset progress steps
            self._progress_steps = {
                "trainer": False,
                "model": False,
                "tokenizer": False,
                "dataloaders": False,
                "optimizer": False
            }
            
            # Create a progress monitor thread
            stop_monitor = threading.Event()
            progress_thread = threading.Thread(
                target=self._monitor_progress,
                args=(stop_monitor, dataset_type)
            )
            progress_thread.daemon = True
            self._set_progress_message("Initializing")
            progress_thread.start()
            
            try:
                # Create the trainer instance
                self._set_progress_message("Creating trainer")
                
                # Ensure tokenizer information is available to model_config
                # This fixes issues with the enhanced trainer
                if not hasattr(config_classes["model"], "tokenizer_name"):
                    if not hasattr(config_classes["model"], "extra_model_params"):
                        config_classes["model"].extra_model_params = {}
                    config_classes["model"].extra_model_params["tokenizer_name"] = config_classes["data"].tokenizer_name
                
                trainer = trainer_factory.create_trainer(
                    model_config=config_classes["model"],
                    training_config=config_classes["training"],
                    data_config=config_classes["data"]
                )
                self._progress_steps["trainer"] = True
                
                # Initialize model with progress update
                self._set_progress_message("Initializing model")
                trainer.initialize_model()
                self._progress_steps["model"] = True
                self.ui.print_success("Model initialized")
                
                # Initialize tokenizer with progress update
                self._set_progress_message("Initializing tokenizer")
                trainer.initialize_tokenizer()
                self._progress_steps["tokenizer"] = True
                self.ui.print_success("Tokenizer initialized")
                
                # Initialize dataloaders with detailed progress update
                self._set_progress_message("Loading dataset")
                self.ui.print_info(f"Downloading and processing {dataset_type} dataset...")
                self.ui.print_info("This may take some time depending on the dataset size.")
                self.ui.print_info("Please wait while the data is being prepared...")
                
                # Notify that dataloaders are initializing
                trainer.initialize_dataloaders()
                self._progress_steps["dataloaders"] = True
                self.ui.print_success("Data loaders initialized")
                
                # Initialize optimizer with progress update
                self._set_progress_message("Initializing optimizer")
                trainer.initialize_optimizer()
                self._progress_steps["optimizer"] = True
                self.ui.print_success("Optimizer initialized")
                
                # Stop the progress monitor
                stop_monitor.set()
                progress_thread.join(timeout=1.0)
                
                # Run training for specified number of epochs
                max_epochs = config_classes["training"].max_epochs
                self.ui.print_section(f"Starting Training ({max_epochs} epochs)")
                
                # Initialize training progress
                total_batches = len(trainer.dataloaders["train"])
                self.ui.print_info(f"Training on {len(trainer.dataloaders['train'].dataset)} examples "
                                  f"({total_batches} batches per epoch)")
                
                # Train for each epoch
                for epoch in range(max_epochs):
                    trainer.current_epoch = epoch
                    self.ui.print_section(f"Epoch {epoch+1}/{max_epochs}")
                    
                    # Show progress
                    start_time = time.time()
                    epoch_metrics = trainer.train_epoch()
                    elapsed_time = time.time() - start_time
                    
                    # Display epoch summary
                    self.ui.print_info(f"Epoch {epoch+1} completed in {elapsed_time:.1f}s:")
                    self.ui.print_info(f"  Train Loss: {epoch_metrics.get('train_loss', 'N/A')}")
                    
                    # Run validation
                    self.ui.print_info("Running validation...")
                    start_time = time.time()
                    val_metrics = trainer.evaluate()
                    elapsed_time = time.time() - start_time
                    
                    self.ui.print_info(f"  Validation Loss: {val_metrics.get('loss', 'N/A')}")
                    self.ui.print_info(f"  Validation Perplexity: {val_metrics.get('perplexity', 'N/A')}")
                    self.ui.print_info(f"  Validation completed in {elapsed_time:.1f}s")
                    
                    # Save checkpoint
                    if epoch == max_epochs - 1 or trainer.training_config.save_every_epoch:
                        checkpoint_path = os.path.join(trainer.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
                        trainer.save_checkpoint(checkpoint_path)
                        self.ui.print_success(f"Saved checkpoint to {checkpoint_path}")
                
                # Final message
                self.ui.print_success(f"Training completed after {max_epochs} epochs")
                self.ui.print_info(f"Model saved to {trainer.output_dir}")
                
            except Exception as e:
                # Stop the progress monitor
                stop_monitor.set()
                if progress_thread.is_alive():
                    progress_thread.join(timeout=1.0)
                raise e
                
        except Exception as e:
            import traceback
            self.ui.print_error(f"Error during training: {str(e)}")
            self.ui.print_info("Detailed error trace:")
            self.ui.print_info(traceback.format_exc())
        
        self.ui.wait_for_any_key("Press any key to return to the menu...")
    
    def _set_progress_message(self, message: str) -> None:
        """Set the current progress message."""
        self._progress_message = message
    
    def _monitor_progress(self, stop_event, dataset_type: str = None):
        """
        Monitor progress in a separate thread.
        
        Args:
            stop_event: Event to signal the thread to stop
            dataset_type: Type of dataset being processed
        """
        animation = "|/-\\"
        idx = 0
        start_time = time.time()
        
        # Define phase-specific messages
        phase_messages = {
            "Creating trainer": "Setting up the training environment...",
            "Initializing model": "Loading model architecture...",
            "Initializing tokenizer": "Configuring tokenizer...",
            "Loading dataset": f"Preparing {dataset_type} dataset...",
            "Initializing optimizer": "Setting up optimization parameters..."
        }
        
        # Track last displayed message
        last_displayed_msg = ""
        last_time_check = start_time
        
        # Clear initial line
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()
        
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            current_time = time.time()
            
            # Get current message with additional details based on phase
            current_msg = self._progress_message or "Initializing"
            detail_msg = phase_messages.get(current_msg, "")
            
            # Format elapsed time nicely
            if elapsed < 60:
                time_str = f"{int(elapsed)}s"
            else:
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                time_str = f"{minutes}m {seconds}s"
            
            # Determine if we should show a detailed message
            if current_time - last_time_check > 5.0:  # Show details every 5 seconds
                display_msg = f"{animation[idx//4 % len(animation)]} {current_msg}... ({time_str}) {detail_msg}"
                last_time_check = current_time
            else:
                display_msg = f"{animation[idx//4 % len(animation)]} {current_msg}... ({time_str})"
            
            # Only update if message changed to reduce flickering
            if display_msg != last_displayed_msg:
                sys.stdout.write("\r" + " " * 80 + "\r")  # Clear line
                sys.stdout.write(display_msg)
                sys.stdout.flush()
                last_displayed_msg = display_msg
            
            # Update every quarter second
            idx += 1
            time.sleep(0.25)
            
            # For "Loading dataset" phase that tends to hang, provide more feedback
            if current_msg == "Loading dataset" and elapsed > 30 and elapsed % 10 < 0.5:
                sys.stdout.write("\r" + " " * 80 + "\r")  # Clear line
                progress_info = ""
                
                if dataset_type == "daily_dialog":
                    progress_info = "This dataset is typically small and should finish soon."
                elif dataset_type == "wikitext":
                    progress_info = "This is a large dataset and may take several minutes to prepare."
                
                sys.stdout.write(f"Still loading... ({time_str}) {progress_info}")
                sys.stdout.flush()
                last_displayed_msg = ""  # Force update on next iteration
        
        # Final cleanup: clear the progress indicator line
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()
    
    def handle_resume_training(self) -> None:
        """Handle resuming a previous training session."""
        self.ui.clear_screen()
        self.ui.print_header("Resume Training")
        
        # Find checkpoints
        runs_dir = "runs"
        if not os.path.exists(runs_dir):
            self.ui.print_info("No training runs found.")
            self.ui.wait_for_any_key()
            return
        
        # List training runs
        run_dirs = [d for d in os.listdir(runs_dir) 
                   if os.path.isdir(os.path.join(runs_dir, d))]
        
        if not run_dirs:
            self.ui.print_info("No training runs found.")
            self.ui.wait_for_any_key()
            return
        
        # Sort by modification time (newest first)
        run_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)), 
                     reverse=True)
        
        # Let user select a run
        self.ui.print_info("Available training runs:")
        for i, run_dir in enumerate(run_dirs, 1):
            # Get run info
            run_path = os.path.join(runs_dir, run_dir)
            mtime = os.path.getmtime(run_path)
            mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
            
            print(f"  {i}. {run_dir} (Last modified: {mtime_str})")
        
        print()
        choice = self.ui.prompt_int(
            "Select a training run to resume (0 to cancel)",
            default=0,
            min_value=0,
            max_value=len(run_dirs)
        )
        
        if choice == 0:
            return
        
        selected_run = os.path.join(runs_dir, run_dirs[choice - 1])
        
        # Check for checkpoints
        checkpoint_dir = os.path.join(selected_run, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            self.ui.print_info(f"No checkpoints found in {selected_run}.")
            self.ui.wait_for_any_key()
            return
        
        # List checkpoints
        checkpoints = [f for f in os.listdir(checkpoint_dir) 
                       if f.endswith(".pt") or f.endswith(".bin")]
        
        if not checkpoints:
            self.ui.print_info(f"No checkpoint files found in {checkpoint_dir}.")
            self.ui.wait_for_any_key()
            return
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)), 
                        reverse=True)
        
        # Let user select a checkpoint
        self.ui.print_info("Available checkpoints:")
        for i, ckpt in enumerate(checkpoints, 1):
            # Get checkpoint info
            ckpt_path = os.path.join(checkpoint_dir, ckpt)
            mtime = os.path.getmtime(ckpt_path)
            mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
            
            print(f"  {i}. {ckpt} (Last modified: {mtime_str})")
        
        print()
        choice = self.ui.prompt_int(
            "Select a checkpoint to resume from (0 to cancel)",
            default=0,
            min_value=0,
            max_value=len(checkpoints)
        )
        
        if choice == 0:
            return
        
        selected_checkpoint = os.path.join(checkpoint_dir, checkpoints[choice - 1])
        
        # Attempt to load configuration
        config_path = os.path.join(selected_run, "config.json")
        if os.path.exists(config_path):
            try:
                self.current_config = self.config_manager.load_config(config_path)
                self.ui.print_info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.ui.print_warning(f"Could not load configuration: {str(e)}")
                self.ui.print_info("Using default configuration.")
                self.current_config = self.config_manager.create_default_config()
        else:
            self.ui.print_warning(f"No configuration file found at {config_path}")
            self.ui.print_info("Using default configuration.")
            self.current_config = self.config_manager.create_default_config()
        
        # Resume training with progress display
        try:
            # Convert dict config to config classes
            config_classes = self.config_manager.to_config_classes(self.current_config)
            
            # Import TrainerFactory
            from src.training import TrainerFactory
            
            # Create appropriate trainer
            trainer_factory = TrainerFactory()
            
            # Clear screen and show initialization message
            self.ui.clear_screen()
            self.ui.print_header("Resuming Training")
            
            # Reset progress steps
            self._progress_steps = {
                "trainer": False,
                "model": False,
                "tokenizer": False,
                "dataloaders": False,
                "optimizer": False,
                "checkpoint": False
            }
            
            # Create a progress monitor thread
            stop_monitor = threading.Event()
            progress_thread = threading.Thread(
                target=self._monitor_progress,
                args=(stop_monitor, config_classes["data"].dataset_name)
            )
            progress_thread.daemon = True
            self._set_progress_message("Initializing")
            progress_thread.start()
            
            try:
                # Create trainer
                self._set_progress_message("Creating trainer")
                trainer = trainer_factory.create_trainer(
                    model_config=config_classes["model"],
                    training_config=config_classes["training"],
                    data_config=config_classes["data"],
                    output_dir=selected_run
                )
                self._progress_steps["trainer"] = True
                
                # Initialize components
                self._set_progress_message("Initializing model")
                trainer.initialize_model()
                self._progress_steps["model"] = True
                self.ui.print_success("Model initialized")
                
                self._set_progress_message("Initializing tokenizer")
                trainer.initialize_tokenizer()
                self._progress_steps["tokenizer"] = True
                self.ui.print_success("Tokenizer initialized")
                
                self._set_progress_message("Loading dataset")
                trainer.initialize_dataloaders()
                self._progress_steps["dataloaders"] = True
                self.ui.print_success("Data loaders initialized")
                
                self._set_progress_message("Initializing optimizer")
                trainer.initialize_optimizer()
                self._progress_steps["optimizer"] = True
                self.ui.print_success("Optimizer initialized")
                
                # Load checkpoint
                self._set_progress_message(f"Loading checkpoint")
                trainer.load_checkpoint(selected_checkpoint)
                self._progress_steps["checkpoint"] = True
                self.ui.print_success("Checkpoint loaded successfully")
                
                # Stop the progress monitor
                stop_monitor.set()
                progress_thread.join(timeout=1.0)
                
                # Get remaining epochs
                start_epoch = trainer.current_epoch + 1
                max_epochs = config_classes["training"].max_epochs
                remaining_epochs = max_epochs - start_epoch
                
                if remaining_epochs <= 0:
                    self.ui.print_warning("No epochs remaining based on configuration.")
                    self.ui.print_info("Increasing max epochs to allow for additional training.")
                    max_epochs = start_epoch + 3  # Train for 3 more epochs
                    remaining_epochs = 3
                    trainer.training_config.max_epochs = max_epochs
                
                # Train for remaining epochs
                self.ui.print_header(f"Resuming Training (Epochs {start_epoch+1}-{max_epochs})")
                
                total_batches = len(trainer.dataloaders["train"])
                self.ui.print_info(f"Training on {len(trainer.dataloaders['train'].dataset)} examples "
                                  f"({total_batches} batches per epoch)")
                
                for epoch in range(start_epoch, max_epochs):
                    trainer.current_epoch = epoch
                    self.ui.print_section(f"Epoch {epoch+1}/{max_epochs}")
                    
                    # Train epoch
                    start_time = time.time()
                    epoch_metrics = trainer.train_epoch()
                    elapsed_time = time.time() - start_time
                    
                    # Display epoch summary
                    self.ui.print_info(f"Epoch {epoch+1} completed in {elapsed_time:.1f}s:")
                    self.ui.print_info(f"  Train Loss: {epoch_metrics.get('train_loss', 'N/A')}")
                    
                    # Run validation
                    self.ui.print_info("Running validation...")
                    start_time = time.time()
                    val_metrics = trainer.evaluate()
                    elapsed_time = time.time() - start_time
                    
                    self.ui.print_info(f"  Validation Loss: {val_metrics.get('loss', 'N/A')}")
                    self.ui.print_info(f"  Validation Perplexity: {val_metrics.get('perplexity', 'N/A')}")
                    self.ui.print_info(f"  Validation completed in {elapsed_time:.1f}s")
                    
                    # Save checkpoint
                    if epoch == max_epochs - 1 or trainer.training_config.save_every_epoch:
                        checkpoint_path = os.path.join(trainer.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
                        trainer.save_checkpoint(checkpoint_path)
                        self.ui.print_success(f"Saved checkpoint to {checkpoint_path}")
                
                # Final message
                self.ui.print_success(f"Training completed after {remaining_epochs} additional epochs")
                self.ui.print_info(f"Model saved to {trainer.output_dir}")
                
            except Exception as e:
                # Stop the progress monitor
                stop_monitor.set()
                if progress_thread.is_alive():
                    progress_thread.join(timeout=1.0)
                raise e
                
        except Exception as e:
            import traceback
            self.ui.print_error(f"Error during training: {str(e)}")
            self.ui.print_info("Detailed error trace:")
            self.ui.print_info(traceback.format_exc())
        
        self.ui.wait_for_any_key("Press any key to return to the menu...")
    
    def handle_configure_extensions(self) -> None:
        """Handle configuring extensions."""
        self.ui.clear_screen()
        self.ui.print_header("Configure Extensions")
        
        # Create a new configuration if none exists
        if self.current_config is None:
            self.current_config = self.config_manager.create_default_config()
        
        # Initialize extensions configuration if needed
        if "model" not in self.current_config:
            self.current_config["model"] = {}
        if "extensions" not in self.current_config["model"]:
            self.current_config["model"]["extensions"] = {
                "extensions_enabled": False,
                "enable_memory": False,
                "enable_multimodal": False,
                "enable_quantum": False
            }
        
        extensions_config = self.current_config["model"]["extensions"]
        
        # Enable/disable all extensions
        extensions_enabled = self.ui.prompt_bool(
            "Enable extensions?", 
            default=extensions_config.get("extensions_enabled", False)
        )
        extensions_config["extensions_enabled"] = extensions_enabled
        
        if extensions_enabled:
            # Configure individual extensions
            memory_enabled = self.ui.prompt_bool(
                "Enable Memory extension?", 
                default=extensions_config.get("enable_memory", False)
            )
            extensions_config["enable_memory"] = memory_enabled
            
            multimodal_enabled = self.ui.prompt_bool(
                "Enable Multimodal extension?", 
                default=extensions_config.get("enable_multimodal", False)
            )
            extensions_config["enable_multimodal"] = multimodal_enabled
            
            quantum_enabled = self.ui.prompt_bool(
                "Enable Quantum extension?", 
                default=extensions_config.get("enable_quantum", False)
            )
            extensions_config["enable_quantum"] = quantum_enabled
            
            # Configure enabled extensions
            if memory_enabled and self.ui.prompt_bool("Configure Memory extension details?", default=False):
                self.handle_memory_extensions()
                
            if multimodal_enabled and self.ui.prompt_bool("Configure Multimodal extension details?", default=False):
                self.handle_multimodal_extensions()
                
            if quantum_enabled and self.ui.prompt_bool("Configure Quantum extension details?", default=False):
                self.handle_quantum_extensions()
        
        self.ui.print_success("Extensions configuration updated.")
        self.ui.wait_for_any_key()
    
    def handle_memory_extensions(self) -> None:
        """Handle memory extensions configuration."""
        self.ui.clear_screen()
        self.ui.print_header("Memory Extensions Configuration")
        
        # Initialize configuration
        if self.current_config is None:
            self.current_config = self.config_manager.create_default_config()
        
        if "model" not in self.current_config:
            self.current_config["model"] = {}
        if "extensions" not in self.current_config["model"]:
            self.current_config["model"]["extensions"] = {}
        if "memory_config" not in self.current_config["model"]["extensions"]:
            self.current_config["model"]["extensions"]["memory_config"] = {
                "memory_size": 1000,
                "entity_dim": 256,
                "relation_dim": 128,
                "max_entities": 10000,
                "max_relations": 50000
            }
        
        memory_config = self.current_config["model"]["extensions"]["memory_config"]
        
        # Configure memory settings
        memory_size = self.ui.prompt_int(
            "Memory size",
            default=memory_config.get("memory_size", 1000),
            min_value=100,
            max_value=100000
        )
        memory_config["memory_size"] = memory_size
        
        entity_dim = self.ui.prompt_int(
            "Entity dimension",
            default=memory_config.get("entity_dim", 256),
            min_value=32,
            max_value=1024
        )
        memory_config["entity_dim"] = entity_dim
        
        relation_dim = self.ui.prompt_int(
            "Relation dimension",
            default=memory_config.get("relation_dim", 128),
            min_value=32,
            max_value=1024
        )
        memory_config["relation_dim"] = relation_dim
        
        self.ui.print_success("Memory extension configuration updated.")
        self.ui.wait_for_any_key()
    
    def handle_multimodal_extensions(self) -> None:
        """Handle multimodal extensions configuration."""
        self.ui.clear_screen()
        self.ui.print_header("Multimodal Extensions Configuration")
        
        # Initialize configuration
        if self.current_config is None:
            self.current_config = self.config_manager.create_default_config()
        
        if "model" not in self.current_config:
            self.current_config["model"] = {}
        if "extensions" not in self.current_config["model"]:
            self.current_config["model"]["extensions"] = {}
        if "multimodal_config" not in self.current_config["model"]["extensions"]:
            self.current_config["model"]["extensions"]["multimodal_config"] = {
                "vision_model": "resnet50",
                "use_spatial_features": True,
                "fusion_type": "film",
                "vision_primes": [23, 29, 31, 37],
                "fusion_heads": 6
            }
        
        multimodal_config = self.current_config["model"]["extensions"]["multimodal_config"]
        
        # Configure multimodal settings
        vision_model_options = ["resnet50", "resnet101", "vit_base", "vit_large"]
        vision_model_idx = self.ui.prompt_choice(
            "Vision model",
            vision_model_options,
            default=vision_model_options.index(multimodal_config.get("vision_model", "resnet50")) if multimodal_config.get("vision_model", "resnet50") in vision_model_options else 0
        )
        multimodal_config["vision_model"] = vision_model_options[vision_model_idx]
        
        use_spatial = self.ui.prompt_bool(
            "Use spatial features?",
            default=multimodal_config.get("use_spatial_features", True)
        )
        multimodal_config["use_spatial_features"] = use_spatial
        
        fusion_type_options = ["film", "concat", "attention"]
        fusion_type_idx = self.ui.prompt_choice(
            "Fusion type",
            fusion_type_options,
            default=fusion_type_options.index(multimodal_config.get("fusion_type", "film")) if multimodal_config.get("fusion_type", "film") in fusion_type_options else 0
        )
        multimodal_config["fusion_type"] = fusion_type_options[fusion_type_idx]
        
        self.ui.print_success("Multimodal extension configuration updated.")
        self.ui.wait_for_any_key()
    
    def handle_quantum_extensions(self) -> None:
        """Handle quantum extensions configuration."""
        self.ui.clear_screen()
        self.ui.print_header("Quantum Extensions Configuration")
        
        # Initialize configuration
        if self.current_config is None:
            self.current_config = self.config_manager.create_default_config()
        
        if "model" not in self.current_config:
            self.current_config["model"] = {}
        if "extensions" not in self.current_config["model"]:
            self.current_config["model"]["extensions"] = {}
        if "quantum_config" not in self.current_config["model"]["extensions"]:
            self.current_config["model"]["extensions"]["quantum_config"] = {
                "pattern_type": "harmonic",
                "base_sparsity": 0.8,
                "mask_type": "binary"
            }
        
        quantum_config = self.current_config["model"]["extensions"]["quantum_config"]
        
        # Configure quantum settings
        pattern_type_options = ["harmonic", "prime", "fibonacci", "custom"]
        pattern_type_idx = self.ui.prompt_choice(
            "Pattern type",
            pattern_type_options,
            default=pattern_type_options.index(quantum_config.get("pattern_type", "harmonic")) if quantum_config.get("pattern_type", "harmonic") in pattern_type_options else 0
        )
        quantum_config["pattern_type"] = pattern_type_options[pattern_type_idx]
        
        base_sparsity = self.ui.prompt_float(
            "Base sparsity (0.0-1.0)",
            default=quantum_config.get("base_sparsity", 0.8),
            min_value=0.0,
            max_value=0.99
        )
        quantum_config["base_sparsity"] = base_sparsity
        
        mask_type_options = ["binary", "continuous", "adaptive"]
        mask_type_idx = self.ui.prompt_choice(
            "Mask type",
            mask_type_options,
            default=mask_type_options.index(quantum_config.get("mask_type", "binary")) if quantum_config.get("mask_type", "binary") in mask_type_options else 0
        )
        quantum_config["mask_type"] = mask_type_options[mask_type_idx]
        
        self.ui.print_success("Quantum extension configuration updated.")
        self.ui.wait_for_any_key()
    
    def handle_list_extensions(self) -> None:
        """List all enabled extensions."""
        self.ui.clear_screen()
        self.ui.print_header("Enabled Extensions")
        
        # Check if configuration exists
        if self.current_config is None or "model" not in self.current_config or "extensions" not in self.current_config["model"]:
            self.ui.print_info("No extensions configuration found.")
            self.ui.wait_for_any_key()
            return
        
        extensions_config = self.current_config["model"]["extensions"]
        
        # Check if extensions are enabled
        if not extensions_config.get("extensions_enabled", False):
            self.ui.print_info("Extensions are currently disabled.")
            self.ui.wait_for_any_key()
            return
        
        # List enabled extensions
        enabled_extensions = []
        
        if extensions_config.get("enable_memory", False):
            enabled_extensions.append("Memory")
            self.ui.print_section("Memory Extension")
            memory_config = extensions_config.get("memory_config", {})
            for key, value in memory_config.items():
                self.ui.print_info(f"  {key}: {value}")
        
        if extensions_config.get("enable_multimodal", False):
            enabled_extensions.append("Multimodal")
            self.ui.print_section("Multimodal Extension")
            multimodal_config = extensions_config.get("multimodal_config", {})
            for key, value in multimodal_config.items():
                self.ui.print_info(f"  {key}: {value}")
        
        if extensions_config.get("enable_quantum", False):
            enabled_extensions.append("Quantum")
            self.ui.print_section("Quantum Extension")
            quantum_config = extensions_config.get("quantum_config", {})
            for key, value in quantum_config.items():
                self.ui.print_info(f"  {key}: {value}")
        
        if not enabled_extensions:
            self.ui.print_info("No extensions are currently enabled.")
        
        self.ui.wait_for_any_key()
    
    # Placeholder implementations for the other handlers
    
    def handle_configure_evaluation(self) -> None:
        """Handle configuring evaluation parameters."""
        self.ui.print_info("Evaluation configuration not yet implemented.")
        self.ui.wait_for_any_key()
    
    def handle_load_model(self) -> None:
        """Handle loading a trained model."""
        self.ui.print_info("Model loading not yet implemented.")
        self.ui.wait_for_any_key()
    
    def handle_run_evaluation(self) -> None:
        """Handle running evaluation on a model."""
        self.ui.print_info("Evaluation not yet implemented.")
        self.ui.wait_for_any_key()
    
    def handle_visualize_results(self) -> None:
        """Handle visualizing evaluation results."""
        self.ui.print_info("Results visualization not yet implemented.")
        self.ui.wait_for_any_key()
    
    def handle_configure_generation(self) -> None:
        """Handle configuring generation parameters."""
        self.ui.print_info("Generation configuration not yet implemented.")
        self.ui.wait_for_any_key()
    
    def handle_interactive_generation(self) -> None:
        """Handle interactive text generation."""
        self.ui.print_info("Interactive generation not yet implemented.")
        self.ui.wait_for_any_key()
    
    def handle_batch_generation(self) -> None:
        """Handle batch text generation."""
        self.ui.print_info("Batch generation not yet implemented.")
        self.ui.wait_for_any_key()
    
    def handle_configure_compression(self) -> None:
        """Handle configuring compression parameters."""
        self.ui.print_info("Compression configuration not yet implemented.")
        self.ui.wait_for_any_key()
    
    def handle_compress_model(self) -> None:
        """Handle compressing a model."""
        self.ui.print_info("Model compression not yet implemented.")
        self.ui.wait_for_any_key()
    
    def handle_compare_compression(self) -> None:
        """Handle comparing model before and after compression."""
        self.ui.print_info("Compression comparison not yet implemented.")
        self.ui.wait_for_any_key()