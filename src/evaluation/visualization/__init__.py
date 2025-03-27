"""
Visualization package for QLLM model evaluation results.

This package provides a comprehensive set of visualization tools for analyzing
and presenting evaluation results for QLLM models, including performance metrics,
knowledge graphs, quantum mask patterns, and multimodal attention visualizations.
"""

# Basic metric plots
from .metric_plots import (
    plot_metric_comparison,
    plot_parameter_breakdown,
    plot_time_series,
    plot_metric_heatmap
)

# Comparison plots
from .comparison_plots import (
    plot_ablation_study,
    plot_extension_comparison,
    plot_efficiency_metrics,
    plot_extension_metrics
)

# Dashboard generation
from .dashboard import (
    create_summary_dashboard,
    create_summary_markdown,
    create_interactive_dashboard
)

# Results loading utilities
from .results_loader import (
    load_evaluation_results,
    find_latest_results,
    extract_metric_data,
    extract_ablation_data,
    convert_to_dataframe,
    get_extension_metrics
)

# Mask evolution visualization
from .mask_evolution import (
    plot_mask_evolution_heatmap,
    create_mask_evolution_animation,
    plot_sparsity_evolution,
    plot_mask_stability_metrics,
    visualize_mask_pattern_comparison
)

# Knowledge graph visualization
from .knowledge_graph import (
    visualize_knowledge_graph,
    visualize_traversal_path,
    visualize_complex_structure,
    visualize_counterfactual_comparison,
    visualize_inductive_reasoning
)

# Multimodal attention visualization
from .multimodal_attention import (
    visualize_cross_modal_attention,
    visualize_attention_regions,
    visualize_multimodal_fusion,
    visualize_modal_entanglement
)

# Dictionary mapping visualization names to functions for easier dynamic usage
VISUALIZATIONS = {
    # Basic metric plots
    "metric_comparison": plot_metric_comparison,
    "parameter_breakdown": plot_parameter_breakdown,
    "time_series": plot_time_series,
    "metric_heatmap": plot_metric_heatmap,
    
    # Comparison plots
    "ablation_study": plot_ablation_study,
    "extension_comparison": plot_extension_comparison,
    "efficiency_metrics": plot_efficiency_metrics,
    "extension_metrics": plot_extension_metrics,
    
    # Dashboards
    "summary_dashboard": create_summary_dashboard,
    "interactive_dashboard": create_interactive_dashboard,
    
    # Mask evolution visualizations
    "mask_evolution_heatmap": plot_mask_evolution_heatmap,
    "mask_evolution_animation": create_mask_evolution_animation,
    "sparsity_evolution": plot_sparsity_evolution,
    "mask_stability_metrics": plot_mask_stability_metrics,
    "mask_pattern_comparison": visualize_mask_pattern_comparison,
    
    # Knowledge graph visualizations
    "knowledge_graph": visualize_knowledge_graph,
    "traversal_path": visualize_traversal_path,
    "complex_structure": visualize_complex_structure,
    "counterfactual_comparison": visualize_counterfactual_comparison,
    "inductive_reasoning": visualize_inductive_reasoning,
    
    # Multimodal attention visualizations
    "cross_modal_attention": visualize_cross_modal_attention,
    "attention_regions": visualize_attention_regions,
    "multimodal_fusion": visualize_multimodal_fusion,
    "modal_entanglement": visualize_modal_entanglement
}