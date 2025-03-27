"""
Compositional generalization metrics for QLLM models.

This module provides metrics to evaluate how well models generalize
to novel combinations of concepts they've seen during training.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union


def compositional_entailment_score(
    model: torch.nn.Module,
    test_examples: List[Dict[str, str]],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate compositional generalization through entailment classification.
    
    Args:
        model: The model to evaluate
        test_examples: List of premise-hypothesis pairs with relation labels
            Each example should be a dict with 'premise', 'hypothesis', and 'relation'
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary of compositional generalization metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not test_examples:
        return {"error": "No test examples provided"}
    
    # Tokenizer
    tokenizer = model.tokenizer
    
    # Classification results
    correct = 0
    total = 0
    relation_results = {"entailment": [], "contradiction": [], "neutral": []}
    
    for example in test_examples:
        premise = example.get("premise", "")
        hypothesis = example.get("hypothesis", "")
        true_relation = example.get("relation", "neutral")
        
        # Skip invalid examples
        if not premise or not hypothesis:
            continue
            
        # Format input for entailment task
        task_input = f"Premise: {premise}\nHypothesis: {hypothesis}\nRelation:"
        
        with torch.no_grad():
            # Generate relation classification
            output = model.generate(task_input, max_length=20)
            
            # Parse output to get predicted relation
            prediction = output.lower()
            
            if "entailment" in prediction:
                predicted_relation = "entailment"
            elif "contradiction" in prediction:
                predicted_relation = "contradiction"
            else:
                predicted_relation = "neutral"
            
            # Check if prediction is correct
            if predicted_relation == true_relation:
                correct += 1
                
            # Track results by relation type
            relation_results[true_relation].append(predicted_relation == true_relation)
            
            total += 1
    
    if total == 0:
        return {"error": "No valid examples processed"}
    
    # Calculate accuracy by relation type
    relation_accuracy = {}
    for relation, results in relation_results.items():
        if results:
            relation_accuracy[relation] = sum(results) / len(results)
    
    # Calculate overall accuracy
    accuracy = correct / total
    
    return {
        "compositional_accuracy": accuracy,
        "relation_accuracy": relation_accuracy,
        "total_examples": total,
        "correct_examples": correct
    }


def systematic_generalization(
    model: torch.nn.Module,
    base_patterns: List[str],
    novel_combinations: List[str]
) -> Dict[str, Any]:
    """
    Evaluate systematic generalization to novel combinations of known patterns.
    
    Args:
        model: The model to evaluate
        base_patterns: List of base patterns the model should know
        novel_combinations: List of novel combinations of base patterns
        
    Returns:
        Dictionary of systematic generalization metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not base_patterns or not novel_combinations:
        return {"error": "Base patterns or novel combinations not provided"}
    
    # Evaluate base pattern perplexity
    base_perplexities = []
    for pattern in base_patterns:
        try:
            ppl = perplexity_for_text(model, pattern)
            base_perplexities.append(ppl)
        except Exception as e:
            if len(base_perplexities) == 0:
                return {"error": f"Failed to calculate base perplexities: {str(e)}"}
    
    # Evaluate novel combination perplexity
    novel_perplexities = []
    for combo in novel_combinations:
        try:
            ppl = perplexity_for_text(model, combo)
            novel_perplexities.append(ppl)
        except Exception as e:
            if len(novel_perplexities) == 0:
                return {"error": f"Failed to calculate novel perplexities: {str(e)}"}
    
    # Compare perplexities
    mean_base_ppl = sum(base_perplexities) / len(base_perplexities)
    mean_novel_ppl = sum(novel_perplexities) / len(novel_perplexities)
    
    # Calculate generalization gap
    generalization_gap = mean_novel_ppl - mean_base_ppl
    
    # Calculate generalization ratio
    generalization_ratio = mean_novel_ppl / mean_base_ppl if mean_base_ppl > 0 else float('inf')
    
    return {
        "base_perplexity": mean_base_ppl,
        "novel_perplexity": mean_novel_ppl,
        "generalization_gap": generalization_gap,
        "generalization_ratio": generalization_ratio,
        "base_samples": len(base_perplexities),
        "novel_samples": len(novel_perplexities)
    }


def cross_domain_transfer(
    model: torch.nn.Module,
    source_domain_examples: List[Dict[str, str]],
    target_domain_examples: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Evaluate transfer of knowledge across domains.
    
    Args:
        model: The model to evaluate
        source_domain_examples: List of examples from source domain
        target_domain_examples: List of examples from target domain
        
    Returns:
        Dictionary of cross-domain transfer metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not source_domain_examples or not target_domain_examples:
        return {"error": "Source or target domain examples not provided"}
    
    # Evaluate on source domain
    source_scores = compositional_entailment_score(model, source_domain_examples)
    
    # Evaluate on target domain
    target_scores = compositional_entailment_score(model, target_domain_examples)
    
    # Calculate transfer metrics
    source_accuracy = source_scores.get("compositional_accuracy", 0)
    target_accuracy = target_scores.get("compositional_accuracy", 0)
    
    # Transfer ratio (how well performance transfers)
    transfer_ratio = target_accuracy / source_accuracy if source_accuracy > 0 else 0
    
    # Transfer gap (absolute performance difference)
    transfer_gap = source_accuracy - target_accuracy
    
    return {
        "source_accuracy": source_accuracy,
        "target_accuracy": target_accuracy,
        "transfer_ratio": transfer_ratio,
        "transfer_gap": transfer_gap,
        "source_examples": source_scores.get("total_examples", 0),
        "target_examples": target_scores.get("total_examples", 0)
    }


def perplexity_for_text(model: torch.nn.Module, text: str) -> float:
    """Helper function to calculate perplexity for a single text"""
    tokenizer = model.tokenizer
    encodings = tokenizer(text, return_tensors="pt")
    
    input_ids = encodings["input_ids"].to(next(model.parameters()).device)
    target_ids = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        
        # Shift logits and targets for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        
        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Calculate perplexity
        return torch.exp(loss).item()