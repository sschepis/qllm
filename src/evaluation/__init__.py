"""
Evaluation package for QLLM models.

This package provides comprehensive evaluation tools for Quantum Language 
Learning Models (QLLM), including metrics for performance, integration tests,
and visualizations for understanding model behavior.
"""

from src.evaluation.metrics import (
    # General metrics
    perplexity,
    parameter_efficiency,
    memory_usage,
    inference_speed,
    generation_diversity,
    
    # Multimodal metrics
    multimodal_accuracy,
    image_captioning_quality,
    cross_modal_retrieval,
    evaluate_multimodal_extension,
    
    # Memory metrics
    knowledge_graph_retrieval,
    memory_consistency,
    memory_capacity,
    evaluate_memory_extension,
    
    # Quantum metrics
    quantum_efficiency_gain,
    pattern_effectiveness,
    sparsity_accuracy_tradeoff,
    evaluate_quantum_extension,
    
    # Compositional generalization metrics
    compositional_entailment_score,
    systematic_generalization,
    cross_domain_transfer,
    
    # Emergent knowledge metrics
    fact_retrieval_accuracy,
    emergent_reasoning,
    knowledge_integration,
    emergent_knowledge_capabilities,
    
    # Resonance stability metrics
    entropy_collapse_efficiency,
    prime_resonance_metrics,
    mask_evolution_stability,
    resonance_stability_evaluation,
    
    # Dictionary of all metrics
    METRICS
)

from src.evaluation.visualization import (
    # Basic metric plots
    plot_metric_comparison,
    plot_parameter_breakdown,
    plot_time_series,
    plot_metric_heatmap,
    
    # Comparison plots
    plot_ablation_study,
    plot_extension_comparison,
    plot_efficiency_metrics,
    plot_extension_metrics,
    
    # Dashboards
    create_summary_dashboard,
    create_interactive_dashboard,
    
    # Mask evolution visualizations
    plot_mask_evolution_heatmap,
    create_mask_evolution_animation,
    plot_sparsity_evolution,
    plot_mask_stability_metrics,
    visualize_mask_pattern_comparison,
    
    # Knowledge graph visualizations
    visualize_knowledge_graph,
    visualize_traversal_path,
    visualize_complex_structure,
    visualize_counterfactual_comparison,
    visualize_inductive_reasoning,
    
    # Multimodal attention visualizations
    visualize_cross_modal_attention,
    visualize_attention_regions,
    visualize_multimodal_fusion,
    visualize_modal_entanglement,
    
    # Results loading utilities
    load_evaluation_results,
    find_latest_results,
    extract_metric_data,
    extract_ablation_data,
    convert_to_dataframe,
    get_extension_metrics,
    
    # Dictionary of all visualizations
    VISUALIZATIONS
)

from src.evaluation.comprehensive_suite import (
    run_evaluation_suite,
    EvaluationSuite
)

# Define a mapping of metric groups for the comprehensive suite
METRIC_GROUPS = {
    "basic": [
        "perplexity",
        "parameter_efficiency", 
        "memory_usage",
        "inference_speed",
        "generation_diversity"
    ],
    "compositional": [
        "compositional_entailment_score",
        "systematic_generalization",
        "cross_domain_transfer"
    ],
    "emergent": [
        "fact_retrieval_accuracy",
        "emergent_reasoning",
        "knowledge_integration"
    ],
    "resonance": [
        "entropy_collapse_efficiency",
        "prime_resonance_metrics",
        "mask_evolution_stability"
    ],
    "multimodal": [
        "multimodal_accuracy",
        "image_captioning_quality",
        "cross_modal_retrieval"
    ],
    "memory": [
        "knowledge_graph_retrieval",
        "memory_consistency",
        "memory_capacity"
    ],
    "quantum": [
        "quantum_efficiency_gain",
        "pattern_effectiveness",
        "sparsity_accuracy_tradeoff"
    ]
}

# Define a mapping of visualization groups
VISUALIZATION_GROUPS = {
    "standard": [
        "metric_comparison",
        "parameter_breakdown",
        "metric_heatmap"
    ],
    "quantum_visualizations": [
        "mask_evolution_heatmap",
        "sparsity_evolution",
        "mask_stability_metrics",
        "mask_pattern_comparison"
    ],
    "memory_visualizations": [
        "knowledge_graph",
        "traversal_path",
        "complex_structure",
        "counterfactual_comparison",
        "inductive_reasoning"
    ],
    "multimodal_visualizations": [
        "cross_modal_attention",
        "attention_regions",
        "multimodal_fusion",
        "modal_entanglement"
    ],
    "advanced_comparisons": [
        "ablation_study",
        "extension_comparison",
        "efficiency_metrics",
        "extension_metrics"
    ],
    "dashboards": [
        "summary_dashboard",
        "extension_dashboard"
    ]
}

# Run a complete evaluation with all metrics and visualizations
def run_comprehensive_evaluation(config_path, output_dir=None):
    """
    Run a comprehensive evaluation with all available metrics and visualizations.
    
    This function provides a simplified interface to run a full evaluation
    suite with all the advanced metrics and visualizations.
    
    Args:
        config_path (str): Path to the evaluation configuration
        output_dir (str, optional): Directory to save results
        
    Returns:
        dict: Evaluation results summary
    """
    import json
    import os
    from datetime import datetime
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set default output directory if not provided
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("evaluation_results", f"evaluation_{timestamp}")
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure config has all necessary sections
    if "evaluation_config" not in config:
        config["evaluation_config"] = {}
    
    # Add all metric groups to the configuration
    all_metrics = []
    for group in METRIC_GROUPS.values():
        all_metrics.extend(group)
    
    # Remove duplicates while preserving order
    config["evaluation_config"]["metrics"] = list(dict.fromkeys(all_metrics))
    
    # Add all visualization types
    all_visualizations = []
    for group in VISUALIZATION_GROUPS.values():
        all_visualizations.extend(group)
    
    # Remove duplicates while preserving order
    config["visualize_metrics"] = list(dict.fromkeys(all_visualizations))
    
    # Update output directory
    config["output_dir"] = output_dir
    
    # Save the comprehensive configuration
    comprehensive_config_path = os.path.join(output_dir, "comprehensive_config.json")
    with open(comprehensive_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run evaluation suite
    print(f"Running comprehensive evaluation with configuration from {config_path}")
    print(f"Results will be saved to {output_dir}")
    
    # Run the evaluation
    results = run_evaluation_suite(comprehensive_config_path)
    
    print(f"Comprehensive evaluation complete. Results saved to {output_dir}")
    
    return results

# Version information
__version__ = "1.0.0"