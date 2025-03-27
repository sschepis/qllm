#!/usr/bin/env python3
"""
Enhanced Evaluation Script for QLLM Models.

This script demonstrates how to use the enhanced evaluation capabilities
including new metrics and visualizations for comprehensive model assessment.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation import run_enhanced_evaluation
from src.evaluation.metrics import METRICS
from src.evaluation.visualization import VISUALIZATIONS
from src.evaluation.metrics.compositional import compositional_entailment_score
from src.evaluation.metrics.emergent import emergent_knowledge_capabilities
from src.evaluation.metrics.resonance import resonance_stability_evaluation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run enhanced QLLM evaluation")
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="examples/enhanced_evaluation_config.json",
        help="Path to enhanced evaluation configuration file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to store evaluation results (default: auto-generated)"
    )
    
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Generate all visualizations"
    )
    
    parser.add_argument(
        "--focus", "-f",
        type=str,
        choices=["compositional", "emergent", "resonance", "all"],
        default="all",
        help="Focus on specific advanced metric group"
    )
    
    return parser.parse_args()


def show_available_metrics_and_visualizations():
    """Display available metrics and visualizations."""
    print("\n=== Available Metrics ===")
    for category, metrics in {
        "General": ["perplexity", "parameter_efficiency", "memory_usage", "inference_speed", "generation_diversity"],
        "Compositional": ["compositional_entailment_score", "systematic_generalization", "cross_domain_transfer"],
        "Emergent": ["fact_retrieval_accuracy", "emergent_reasoning", "knowledge_integration"],
        "Resonance": ["entropy_collapse_efficiency", "prime_resonance_metrics", "mask_evolution_stability"],
        "Multimodal": ["multimodal_accuracy", "image_captioning_quality", "cross_modal_retrieval"],
        "Memory": ["knowledge_graph_retrieval", "memory_consistency", "memory_capacity"],
        "Quantum": ["quantum_efficiency_gain", "pattern_effectiveness", "sparsity_accuracy_tradeoff"]
    }.items():
        print(f"\n{category} Metrics:")
        for metric in metrics:
            print(f"  - {metric}")
    
    print("\n=== Available Visualizations ===")
    for category, visualizations in {
        "Standard": ["metric_comparison", "parameter_breakdown", "metric_heatmap"],
        "Quantum": ["mask_evolution_heatmap", "sparsity_evolution", "mask_stability_metrics", "mask_pattern_comparison"],
        "Memory": ["knowledge_graph", "traversal_path", "complex_structure", "counterfactual_comparison"],
        "Multimodal": ["cross_modal_attention", "attention_regions", "multimodal_fusion", "modal_entanglement"],
        "Advanced": ["ablation_study", "extension_comparison", "efficiency_metrics", "extension_metrics"],
        "Dashboards": ["summary_dashboard", "extension_dashboard"]
    }.items():
        print(f"\n{category} Visualizations:")
        for viz in visualizations:
            print(f"  - {viz}")


def customize_config_for_focus(config_path, focus):
    """Customize the config file to focus on specific advanced metrics."""
    # Load the config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Basic metrics that should always be included
    basic_metrics = [
        "perplexity", 
        "parameter_efficiency",
        "inference_speed"
    ]
    
    # Define focus-specific metrics
    focus_metrics = {
        "compositional": [
            "compositional_entailment_score",
            "systematic_generalization",
            "cross_domain_transfer"
        ],
        "emergent": [
            "fact_retrieval_accuracy",
            "emergent_reasoning",
            "knowledge_integration",
            "emergent_knowledge_capabilities"
        ],
        "resonance": [
            "entropy_collapse_efficiency",
            "prime_resonance_metrics",
            "mask_evolution_stability",
            "resonance_stability_evaluation"
        ]
    }
    
    # Update metrics based on focus
    if focus == "all":
        # Use all metrics (config is already complete)
        pass
    else:
        # Use basic metrics plus the focused ones
        config["evaluation_config"]["metrics"] = basic_metrics + focus_metrics[focus]
    
    # Define focus-specific visualizations
    focus_visualizations = {
        "compositional": [
            "metric_comparison",
            "extension_metrics"
        ],
        "emergent": [
            "knowledge_graph",
            "traversal_path",
            "complex_structure",
            "inductive_reasoning"
        ],
        "resonance": [
            "mask_evolution_heatmap",
            "sparsity_evolution",
            "mask_stability_metrics",
            "mask_pattern_comparison"
        ]
    }
    
    # Update visualizations based on focus
    if focus == "all":
        # Keep all visualizations
        pass
    else:
        config["visualize_metrics"] = focus_visualizations[focus]
    
    # Save the modified config to a temporary file
    temp_config_path = "temp_focus_config.json"
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return temp_config_path


def demo_focused_metrics(args):
    """Demonstrate the focused metrics with sample inputs."""
    if args.focus == "compositional" or args.focus == "all":
        print("\n=== Demonstrating Compositional Generalization Metrics ===")
        test_examples = [
            {
                "premise": "Neural networks process information through layers of neurons.",
                "hypothesis": "Information flows through multiple transformation steps in neural networks.",
                "relation": "entailment"
            },
            {
                "premise": "Quantum computing uses qubits for computation.",
                "hypothesis": "Quantum computers can only solve specialized problems.",
                "relation": "contradiction"
            }
        ]
        print("Example compositional_entailment_score input:")
        print(json.dumps(test_examples, indent=2))
        print("This metric evaluates how well models understand logical relationships between statements.")
    
    if args.focus == "emergent" or args.focus == "all":
        print("\n=== Demonstrating Emergent Knowledge Metrics ===")
        eval_data = {
            "fact_questions": [
                {"question": "What is quantum resonance?", "answer": "A phenomenon where quantum systems oscillate in sync."}
            ],
            "reasoning_problems": [
                {
                    "context": "Semantic resonance uses entropy-driven collapse to reduce uncertainty.",
                    "question": "How does this relate to quantum measurement?",
                    "answer": "Both involve uncertainty reduction through measurement-like operations.",
                    "steps": ["Identify key properties of semantic resonance", "Compare to quantum measurement"]
                }
            ]
        }
        print("Example emergent_knowledge_capabilities input:")
        print(json.dumps(eval_data, indent=2))
        print("This metric evaluates the model's ability to integrate knowledge and perform multi-step reasoning.")
    
    if args.focus == "resonance" or args.focus == "all":
        print("\n=== Demonstrating Resonance Stability Metrics ===")
        resonance_data = {
            "collapse_test_inputs": [
                "Simple factual query with low uncertainty",
                "Complex reasoning task with multiple steps"
            ],
            "expected_iterations": [1, 5],
            "prime_test_patterns": [
                "Pattern with regular structure",
                "Pattern with irregular structure"
            ]
        }
        print("Example resonance_stability_evaluation input:")
        print(json.dumps(resonance_data, indent=2))
        print("This metric evaluates the stability and efficiency of quantum-inspired resonance mechanisms.")


def main():
    """Main function."""
    args = parse_args()
    
    # Show available metrics and visualizations
    show_available_metrics_and_visualizations()
    
    # Demonstrate focused metrics
    if args.focus != "all":
        print(f"\nFocusing on {args.focus.upper()} metrics and visualizations")
        demo_focused_metrics(args)
    
    # Customize config based on focus
    if args.focus != "all":
        config_path = customize_config_for_focus(args.config, args.focus)
        print(f"\nCustomized configuration for {args.focus} metrics")
    else:
        config_path = args.config
    
    # Run the enhanced evaluation
    print(f"\nRunning enhanced evaluation with configuration from {config_path}")
    results = run_enhanced_evaluation(config_path, args.output_dir)
    
    # Clean up temporary config if created
    if args.focus != "all":
        try:
            os.remove("temp_focus_config.json")
        except:
            pass
    
    print("\nEvaluation complete!")
    print(f"Results summary: {json.dumps(results, indent=2)[:300]}...")


if __name__ == "__main__":
    main()