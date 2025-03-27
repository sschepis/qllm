#!/usr/bin/env python3
"""
Menu-driven CLI for the Quantum Resonance Language Model.

This script provides a user-friendly interface for configuring, training,
evaluating, and using QLLM through an interactive menu system with
arrow key navigation and rich visual feedback.
"""

import os
import sys
import argparse
import logging
import random
import time
from typing import Dict, Any, Optional

from src.cli.menu_system import MenuSystem
from src.cli.user_interface import TerminalUI
from src.config.config_manager import ConfigManager
from src.utils.logging import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="QLLM - Quantum Resonance Language Model CLI"
    )
    
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="Directory containing configuration files"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/quantum_resonance",
        help="Output directory for model and logs"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["menu", "train", "eval", "generate", "compress"],
        default="menu",
        help="Operation mode (default: menu for interactive mode)"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level"
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Set up the environment."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, "qllm_cli.log")
    logger = setup_logger(
        name="qllm_cli",
        log_file=log_file,
        log_level=getattr(logging, args.log_level.upper())
    )
    
    return logger


def show_welcome_screen(ui):
    """Show a colorful welcome screen with helpful information."""
    ui.clear_screen()
    ui.print_banner()
    
    # Display welcome text
    welcome_text = [
        "Welcome to the Quantum Resonance Language Model CLI",
        "",
        "This interactive tool allows you to configure, train, evaluate,",
        "and use the QLLM through an intuitive menu system.",
        "",
        "• Navigate using the arrow keys ↑ ↓",
        "• Select options with Enter",
        "• Exit any screen with Ctrl+C",
        "",
        "Start exploring the menus to get started!"
    ]
    
    ui.print_box(welcome_text, "WELCOME", color="BRIGHT_CYAN")
    
    # Display a random quantum fact
    quantum_facts = [
        "Quantum entanglement allows particles to maintain instantaneous correlations across any distance.",
        "Quantum superposition allows particles to exist in multiple states simultaneously.",
        "In quantum mechanics, particles can tunnel through barriers that should be impenetrable.",
        "Quantum resonance uses harmonic frequencies to enable efficient information transfer.",
        "The quantum observer effect suggests that the act of observation changes quantum states.",
        "Quantum prime patterns create efficient mathematical representations of complex data."
    ]
    
    ui.print_section("Did You Know?")
    ui.animate_text(random.choice(quantum_facts), delay=0.01, color="BRIGHT_MAGENTA")
    
    # Prompt user to continue
    ui.print_info("\nUse arrow keys to navigate and Enter to select options.")
    key = ui.wait_for_any_key("Press any key to begin...")
    
    # Easter egg if user presses 'q'
    if key.lower() == 'q':
        ui.clear_screen()
        ui.print_header("Quantum Superposition Activated!")
        ui.animate_text(
            "You've discovered the quantum Easter egg! The CLI now exists in multiple states simultaneously.",
            delay=0.01,
            color="BRIGHT_GREEN"
        )
        ui.wait_for_any_key()


def run_menu_mode(args, logger):
    """Run in interactive menu mode."""
    logger.info("Starting QLLM CLI in menu mode")
    
    # Initialize components
    ui = TerminalUI()
    
    # Show welcome screen with smooth animation
    ui.show_loading_screen("Initializing Quantum Resonance Language Model")
    show_welcome_screen(ui)
    
    config_manager = ConfigManager()
    
    # Import MenuHandler through the getter function to avoid circular imports
    from src.cli import get_menu_handler
    menu_handler = get_menu_handler()
    
    # Build main menu
    main_menu = menu_handler.build_main_menu()
    
    # Create menu system
    menu_system = MenuSystem(main_menu)
    
    # Run menu system
    try:
        menu_system.run()
        
        # Show exit message
        ui.clear_screen()
        ui.celebrate("Thank you for using QLLM CLI!")
        ui.print_info("Quantum computing principles, in your hands.")
        
    except KeyboardInterrupt:
        logger.info("QLLM CLI interrupted by user")
        ui.clear_screen()
        ui.print_box(["Exiting QLLM CLI"], "GOODBYE", color="BRIGHT_MAGENTA")
        ui.print_info("Quantum computing principles, in your hands.")
    except Exception as e:
        logger.error(f"Error running QLLM CLI: {str(e)}", exc_info=True)
        ui.print_error(f"An error occurred: {str(e)}")
        ui.wait_for_any_key()
    
    return 0


def run_direct_mode(args, logger):
    """Run in direct mode using command-line arguments."""
    mode = args.mode
    logger.info(f"Starting QLLM CLI in {mode} mode")
    
    # Initialize UI
    ui = TerminalUI()
    
    # For now, we'll just print a message since we're focusing on the menu-driven interface
    ui.clear_screen()
    ui.print_banner()
    ui.print_box(
        [f"Direct mode '{mode}' is available but the menu-driven interface",
         "offers a more user-friendly experience with arrow key navigation.",
         "",
         "We recommend using the menu-driven interface:",
         "",
         "./cli.py"],
        "DIRECT MODE",
        color="BRIGHT_YELLOW"
    )
    ui.wait_for_any_key()
    
    return 1


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Set up environment
    logger = setup_environment(args)
    
    # Run in appropriate mode
    if args.mode == "menu":
        return run_menu_mode(args, logger)
    else:
        return run_direct_mode(args, logger)


if __name__ == "__main__":
    sys.exit(main())