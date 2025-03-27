"""
Evaluation metrics package for QLLM models.

This package provides a comprehensive set of metrics for evaluating
QLLM models, including general performance, multimodal capabilities,
memory extension features, quantum-inspired components, compositional
generalization, emergent knowledge, and resonance stability.
"""

# General metrics
from .general import (
    perplexity,
    parameter_efficiency,
    memory_usage,
    inference_speed,
    generation_diversity
)

# Multimodal metrics
from .multimodal import (
    multimodal_accuracy,
    image_captioning_quality,
    cross_modal_retrieval,
    evaluate_multimodal_extension
)

# Memory metrics
from .memory import (
    knowledge_graph_retrieval,
    memory_consistency,
    memory_capacity,
    evaluate_memory_extension
)

# Quantum metrics
from .quantum import (
    quantum_efficiency_gain,
    pattern_effectiveness,
    sparsity_accuracy_tradeoff,
    evaluate_quantum_extension
)

# Compositional generalization metrics
from .compositional import (
    compositional_entailment_score,
    systematic_generalization,
    cross_domain_transfer
)

# Emergent knowledge metrics
from .emergent import (
    fact_retrieval_accuracy,
    emergent_reasoning,
    knowledge_integration,
    emergent_knowledge_capabilities
)

# Resonance stability metrics
from .resonance import (
    entropy_collapse_efficiency,
    prime_resonance_metrics,
    mask_evolution_stability,
    resonance_stability_evaluation
)

# Dictionary mapping metric names to functions for easier dynamic usage
METRICS = {
    # General metrics
    "perplexity": perplexity,
    "parameter_efficiency": parameter_efficiency,
    "memory_usage": memory_usage,
    "inference_speed": inference_speed,
    "generation_diversity": generation_diversity,
    
    # Multimodal metrics
    "multimodal_accuracy": multimodal_accuracy,
    "image_captioning_quality": image_captioning_quality,
    "cross_modal_retrieval": cross_modal_retrieval,
    "evaluate_multimodal_extension": evaluate_multimodal_extension,
    
    # Memory metrics
    "knowledge_graph_retrieval": knowledge_graph_retrieval,
    "memory_consistency": memory_consistency,
    "memory_capacity": memory_capacity,
    "evaluate_memory_extension": evaluate_memory_extension,
    
    # Quantum metrics
    "quantum_efficiency_gain": quantum_efficiency_gain,
    "pattern_effectiveness": pattern_effectiveness,
    "sparsity_accuracy_tradeoff": sparsity_accuracy_tradeoff,
    "evaluate_quantum_extension": evaluate_quantum_extension,
    
    # Compositional generalization metrics
    "compositional_entailment_score": compositional_entailment_score,
    "systematic_generalization": systematic_generalization,
    "cross_domain_transfer": cross_domain_transfer,
    
    # Emergent knowledge metrics
    "fact_retrieval_accuracy": fact_retrieval_accuracy,
    "emergent_reasoning": emergent_reasoning,
    "knowledge_integration": knowledge_integration,
    "emergent_knowledge_capabilities": emergent_knowledge_capabilities,
    
    # Resonance stability metrics
    "entropy_collapse_efficiency": entropy_collapse_efficiency,
    "prime_resonance_metrics": prime_resonance_metrics,
    "mask_evolution_stability": mask_evolution_stability,
    "resonance_stability_evaluation": resonance_stability_evaluation
}