"""
Emergent knowledge metrics for QLLM models.

This module provides metrics to evaluate emergent knowledge capabilities
that arise from the integration of information across different components.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import re


def fact_retrieval_accuracy(
    model: torch.nn.Module,
    fact_questions: List[Dict[str, str]],
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Evaluate accuracy of retrieving factual knowledge.
    
    Args:
        model: The model to evaluate
        fact_questions: List of question-answer pairs
            Each pair should be a dict with 'question' and 'answer' keys
        threshold: Similarity threshold for answer matching
        
    Returns:
        Dictionary of fact retrieval metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not fact_questions:
        return {"error": "No fact questions provided"}
    
    # Track results
    correct = 0
    confidence_scores = []
    
    for qa_pair in fact_questions:
        question = qa_pair.get("question", "")
        expected_answer = qa_pair.get("answer", "")
        
        # Skip invalid pairs
        if not question or not expected_answer:
            continue
            
        # Generate answer
        with torch.no_grad():
            prediction = model.generate(question, max_length=100)
            
        # Calculate similarity score
        similarity = answer_similarity(prediction, expected_answer)
        confidence_scores.append(similarity)
        
        if similarity >= threshold:
            correct += 1
    
    # Calculate metrics
    total = len(confidence_scores)
    accuracy = correct / total if total > 0 else 0
    mean_confidence = sum(confidence_scores) / total if total > 0 else 0
    
    return {
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "correct": correct,
        "total": total
    }


def emergent_reasoning(
    model: torch.nn.Module,
    reasoning_problems: List[Dict[str, Any]],
    verbosity: int = 0
) -> Dict[str, Any]:
    """
    Evaluate multi-step reasoning capabilities that require emergent understanding.
    
    Args:
        model: The model to evaluate
        reasoning_problems: List of reasoning problems
            Each should have 'context', 'question', 'answer', and 'steps' keys
        verbosity: Level of detail in returned results (0-2)
        
    Returns:
        Dictionary of reasoning metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not reasoning_problems:
        return {"error": "No reasoning problems provided"}
    
    results = {
        "correct": 0,
        "partial_correct": 0,
        "incorrect": 0,
        "step_accuracy": [],
        "final_accuracy": 0,
        "details": []
    }
    
    for problem in reasoning_problems:
        context = problem.get("context", "")
        question = problem.get("question", "")
        expected_answer = problem.get("answer", "")
        expected_steps = problem.get("steps", [])
        
        # Skip invalid problems
        if not context or not question or not expected_answer:
            continue
            
        # Format prompt to encourage step-by-step reasoning
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nLet's solve this step-by-step:\n"
        
        # Generate reasoning trace
        with torch.no_grad():
            reasoning = model.generate(prompt, max_length=500)
        
        # Analyze reasoning trace
        problem_result = analyze_reasoning_trace(
            reasoning,
            expected_steps,
            expected_answer
        )
        
        # Update aggregate metrics
        if problem_result["answer_correct"]:
            results["correct"] += 1
        elif problem_result["step_correctness"] > 0.5:
            results["partial_correct"] += 1
        else:
            results["incorrect"] += 1
            
        results["step_accuracy"].append(problem_result["step_correctness"])
        
        # Store detailed results if requested
        if verbosity > 0:
            detail = {
                "question": question,
                "model_reasoning": reasoning,
                "answer_correct": problem_result["answer_correct"],
                "step_correctness": problem_result["step_correctness"]
            }
            
            if verbosity > 1:
                detail["expected_steps"] = expected_steps
                detail["expected_answer"] = expected_answer
                detail["step_analysis"] = problem_result["step_analysis"]
                
            results["details"].append(detail)
    
    # Calculate final metrics
    total = results["correct"] + results["partial_correct"] + results["incorrect"]
    results["final_accuracy"] = results["correct"] / total if total > 0 else 0
    results["mean_step_accuracy"] = sum(results["step_accuracy"]) / len(results["step_accuracy"]) if results["step_accuracy"] else 0
    results["total_problems"] = total
    
    return results


def knowledge_integration(
    model: torch.nn.Module,
    integration_tests: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluate integration of knowledge across domains or modalities.
    
    Args:
        model: The model to evaluate
        integration_tests: List of integration tests
            Each should have 'query', 'domains', and 'expected_answer' keys
        
    Returns:
        Dictionary of knowledge integration metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not integration_tests:
        return {"error": "No integration tests provided"}
    
    results = {
        "correct": 0,
        "total": 0,
        "domain_combinations": {}
    }
    
    for test in integration_tests:
        query = test.get("query", "")
        domains = test.get("domains", [])
        expected_answer = test.get("expected_answer", "")
        
        # Skip invalid tests
        if not query or not domains or not expected_answer:
            continue
            
        # Format domain information
        domain_str = " + ".join(domains)
        if domain_str not in results["domain_combinations"]:
            results["domain_combinations"][domain_str] = {"correct": 0, "total": 0}
        
        # Generate answer
        with torch.no_grad():
            prediction = model.generate(query, max_length=200)
        
        # Check if answer is correct
        similarity = answer_similarity(prediction, expected_answer)
        correct = similarity >= 0.7
        
        if correct:
            results["correct"] += 1
            results["domain_combinations"][domain_str]["correct"] += 1
            
        results["total"] += 1
        results["domain_combinations"][domain_str]["total"] += 1
    
    # Calculate accuracy
    if results["total"] > 0:
        results["accuracy"] = results["correct"] / results["total"]
    else:
        results["accuracy"] = 0
    
    # Calculate domain combination accuracies
    domain_accuracies = {}
    for domain, counts in results["domain_combinations"].items():
        if counts["total"] > 0:
            domain_accuracies[domain] = counts["correct"] / counts["total"]
        else:
            domain_accuracies[domain] = 0
    
    results["domain_accuracies"] = domain_accuracies
    
    return results


def emergent_knowledge_capabilities(
    model: torch.nn.Module,
    evaluation_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of emergent knowledge capabilities.
    
    Args:
        model: The model to evaluate
        evaluation_data: Dictionary containing evaluation data
            - "fact_questions": List of factual QA pairs
            - "reasoning_problems": List of multi-step reasoning problems
            - "integration_tests": List of cross-domain integration tests
            
    Returns:
        Dictionary of emergent knowledge metrics
    """
    results = {}
    
    # Evaluate fact retrieval
    if "fact_questions" in evaluation_data:
        results["fact_retrieval"] = fact_retrieval_accuracy(
            model,
            evaluation_data["fact_questions"]
        )
    
    # Evaluate emergent reasoning
    if "reasoning_problems" in evaluation_data:
        results["reasoning"] = emergent_reasoning(
            model,
            evaluation_data["reasoning_problems"],
            verbosity=1
        )
    
    # Evaluate knowledge integration
    if "integration_tests" in evaluation_data:
        results["integration"] = knowledge_integration(
            model,
            evaluation_data["integration_tests"]
        )
    
    # Calculate aggregated emergent knowledge score
    scores = []
    if "fact_retrieval" in results and "accuracy" in results["fact_retrieval"]:
        scores.append(results["fact_retrieval"]["accuracy"])
    if "reasoning" in results and "final_accuracy" in results["reasoning"]:
        scores.append(results["reasoning"]["final_accuracy"])
    if "integration" in results and "accuracy" in results["integration"]:
        scores.append(results["integration"]["accuracy"])
    
    if scores:
        results["emergent_knowledge_score"] = sum(scores) / len(scores)
    else:
        results["emergent_knowledge_score"] = 0
    
    return results


# Helper functions

def answer_similarity(prediction: str, expected: str) -> float:
    """Calculate similarity between predicted and expected answers"""
    # Simple case-insensitive partial match
    prediction = prediction.lower()
    expected = expected.lower()
    
    # Check for exact match
    if expected in prediction:
        return 1.0
    
    # Check for partial match
    words_expected = set(re.findall(r'\b\w+\b', expected))
    words_predicted = set(re.findall(r'\b\w+\b', prediction))
    
    if not words_expected:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(words_expected.intersection(words_predicted))
    union = len(words_expected.union(words_predicted))
    
    return intersection / union if union > 0 else 0.0


def analyze_reasoning_trace(
    reasoning: str,
    expected_steps: List[str],
    expected_answer: str
) -> Dict[str, Any]:
    """Analyze a reasoning trace to evaluate correctness of steps and answer"""
    # Extract final answer
    answer_pattern = r"(?:answer|result|conclusion|therefore)[:\s]+([^\n]+)"
    answer_match = re.search(answer_pattern, reasoning.lower())
    final_answer = answer_match.group(1) if answer_match else reasoning.split("\n")[-1]
    
    # Check if final answer is correct
    answer_correct = answer_similarity(final_answer, expected_answer) >= 0.7
    
    # Analyze reasoning steps
    step_analysis = []
    step_correctness = 0.0
    
    # Extract steps from reasoning
    reasoning_lines = reasoning.split("\n")
    reasoning_steps = [line for line in reasoning_lines if line.strip() and not line.startswith("Context:") and not line.startswith("Question:")]
    
    if not expected_steps or not reasoning_steps:
        return {
            "answer_correct": answer_correct,
            "step_correctness": 0.0,
            "step_analysis": []
        }
    
    # Compare expected steps with reasoning steps
    for i, expected in enumerate(expected_steps):
        best_match = 0.0
        best_idx = -1
        
        for j, actual in enumerate(reasoning_steps):
            similarity = answer_similarity(actual, expected)
            if similarity > best_match:
                best_match = similarity
                best_idx = j
        
        step_analysis.append({
            "expected": expected,
            "actual": reasoning_steps[best_idx] if best_idx >= 0 else "",
            "similarity": best_match
        })
    
    # Calculate overall step correctness
    if step_analysis:
        step_correctness = sum(step["similarity"] for step in step_analysis) / len(step_analysis)
    
    return {
        "answer_correct": answer_correct,
        "step_correctness": step_correctness,
        "step_analysis": step_analysis
    }