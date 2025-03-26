#!/usr/bin/env python
"""
Run Comprehensive Evaluation Suite for QLLM Extensions.

This script provides a command-line interface for running the comprehensive
evaluation suite on QLLM extensions and visualizing the results.
"""

import os
import sys
import json
import argparse
import datetime
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.comprehensive_suite import run_evaluation_suite
from src.evaluation.visualize_results import create_summary_dashboard


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


def run_evaluation(config_path: str, output_dir: str, skip_visualization: bool = False) -> None:
    """
    Run the evaluation suite and generate visualizations.
    
    Args:
        config_path: Path to the evaluation configuration file
        output_dir: Directory to store evaluation results
        skip_visualization: Whether to skip result visualization
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set up directories
    dirs = setup_directories(output_dir)
    result_dir = dirs["result_dir"]
    vis_dir = dirs["vis_dir"]
    
    # Update configuration with output directory
    config["output_dir"] = result_dir
    
    # Write the configuration used for this run
    config_output_path = os.path.join(result_dir, "evaluation_config.json")
    with open(config_output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n=== Starting QLLM Evaluation ===")
    print(f"Configuration: {config_path}")
    print(f"Results directory: {result_dir}")
    print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)
    
    # Run evaluation
    try:
        results = run_evaluation_suite(config_path)
        
        # Save results summary
        summary_path = os.path.join(result_dir, "evaluation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nEvaluation complete! Results saved to {result_dir}")
        
        # Generate visualizations
        if not skip_visualization:
            print("\nGenerating visualizations...")
            
            # Find the latest results file
            results_files = [f for f in os.listdir(result_dir) if f.startswith("evaluation_results_") and f.endswith(".json")]
            if results_files:
                # Sort by timestamp (part of filename)
                latest_results_file = sorted(results_files)[-1]
                results_path = os.path.join(result_dir, latest_results_file)
                
                # Load complete results (not just summary)
                with open(results_path, 'r') as f:
                    full_results = json.load(f)
                
                # Create visualizations
                create_summary_dashboard(full_results, vis_dir)
                print(f"Visualizations saved to {vis_dir}")
            else:
                print("No results files found for visualization")
    
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run QLLM Comprehensive Evaluation Suite")
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
    parser.add_argument(
        "--skip-visualization", "-s",
        action="store_true",
        help="Skip generating visualizations"
    )
    
    args = parser.parse_args()
    
    run_evaluation(args.config, args.output_dir, args.skip_visualization)


if __name__ == "__main__":
    main()