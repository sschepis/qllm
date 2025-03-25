#!/usr/bin/env python3
"""
Main entry point for the Quantum Resonance Language Model.

This script provides a unified interface for training, evaluating,
compressing, and generating text with the model.
"""

import sys
import argparse
import logging
from typing import Dict, Any

from src.cli.arg_parsing import parse_args, create_parser_for_mode, infer_mode_from_args
from src.cli.commands import train_command, evaluate_command, generate_command, compress_command
from src.utils.logging import setup_logger


def main():
    """Main entry point."""
    # Determine mode from arguments
    mode = infer_mode_from_args()
    
    # Parse arguments for the specified mode
    args = parse_args(mode)
    
    # Set up logging
    log_level = args.log_level if hasattr(args, 'log_level') else "info"
    logger = setup_logger(
        name="quantum_resonance",
        log_level=getattr(logging, log_level.upper())
    )
    
    # Convert args to dictionary
    args_dict = vars(args)
    
    # Execute command based on mode
    try:
        if mode == "train":
            return train_command(args_dict)
        elif mode == "eval":
            return evaluate_command(args_dict)
        elif mode == "generate":
            return generate_command(args_dict)
        elif mode == "compress":
            return compress_command(args_dict)
        else:
            logger.error(f"Unknown mode: {mode}")
            return 1
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 130  # Standard UNIX exit code for SIGINT
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())