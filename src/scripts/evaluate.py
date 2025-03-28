#!/usr/bin/env python3
"""
Unified Evaluation Script for QLLM Models.

This script provides a single interface for evaluating QLLM models,
consolidating functionality from previous evaluation scripts and using
only components that are available in the codebase.
"""

import os
import sys
import argparse
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

# Import only components that actually exist
from src.evaluation.core.config import EvaluationConfig
from src.evaluation.core.suite import EvaluationSuite
from src.evaluation.comprehensive_suite import ComprehensiveSuite


def setup_directories(output_dir: str) -> Dict[str, str]:
    """
    Set up directories for evaluation outputs.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Dictionary of directory paths
    """
    # Create timestamp for unique directory names
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory structure
    result_dir = os.path.join(output_dir, f"evaluation_{timestamp}")
    vis_dir = os.path.join(result_dir, "visualizations")
    
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    return {
        "result_dir": result_dir,
        "vis_dir": vis_dir
    }


def run_evaluation(config_path: str, output_dir: str, eval_type: str = "basic") -> Dict[str, Any]:
    """
    Run the evaluation using the selected evaluation type.
    
    Args:
        config_path: Path to the evaluation configuration file
        output_dir: Directory to store evaluation results
        eval_type: Type of evaluation to run ('basic' or 'comprehensive')
        
    Returns:
        Dictionary with evaluation results
    """
    # Set up directories
    dirs = setup_directories(output_dir)
    result_dir = dirs["result_dir"]
    
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Update configuration with output directory
    config_dict["output_dir"] = result_dir
    
    # Write the configuration used for this run
    config_output_path = os.path.join(result_dir, "evaluation_config.json")
    with open(config_output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n=== Starting QLLM {eval_type.capitalize()} Evaluation ===")
    print(f"Configuration: {config_path}")
    print(f"Results directory: {result_dir}")
    print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)
    
    try:
        # Create evaluation config from dict
        eval_config = EvaluationConfig.from_dict(config_dict)
        
        # Select the suite based on eval_type
        if eval_type == "comprehensive":
            # Use ComprehensiveSuite for comprehensive evaluation
            suite = ComprehensiveSuite(config=eval_config)
        else:
            # Use the base EvaluationSuite for basic evaluation
            suite = EvaluationSuite(config=eval_config)
        
        # Run evaluation
        results = suite.run_evaluation()
        
        # Save results
        result_path = suite.save_results()
        
        # Save results summary
        summary_path = os.path.join(result_dir, "evaluation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nEvaluation complete! Results saved to {result_path}")
        
        return results
    
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Unified QLLM Evaluation Suite")
    
    # Main arguments
    parser.add_argument(
        "--type", "-t",
        type=str,
        choices=["basic", "comprehensive"],
        default="basic",
        help="Type of evaluation to run (basic or comprehensive)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="examples/evaluation_config.json",
        help="Path to evaluation configuration file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="evaluation_results",
        help="Directory to store evaluation results"
    )
    
    # Parse args and run
    args = parser.parse_args()
    
    # Check for config file compatibility based on evaluation type
    if args.type == "comprehensive" and args.config == "examples/evaluation_config.json":
        # For comprehensive evaluation, look for the complete config file
        complete_config_path = "examples/complete_evaluation_config.json"
        if os.path.exists(complete_config_path):
            print(f"Notice: For comprehensive evaluation, using '{complete_config_path}' instead of '{args.config}'")
            args.config = complete_config_path
    
    # Ensure config exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)
    
    # Run the evaluation
    run_evaluation(
        config_path=args.config,
        output_dir=args.output_dir,
        eval_type=args.type
    )


if __name__ == "__main__":
    main()