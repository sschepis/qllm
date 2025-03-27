"""
Memory extension evaluation metrics for QLLM models.

This module provides metrics specifically for evaluating knowledge graph
and memory capabilities of QLLM models.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union


def knowledge_graph_retrieval(
    model: torch.nn.Module,
    queries: List[Dict[str, Any]],
    expected_results: List[Any]
) -> Dict[str, float]:
    """
    Measure accuracy of knowledge graph retrieval.
    
    Args:
        model: The model to evaluate
        queries: List of knowledge graph queries
        expected_results: List of expected results
        
    Returns:
        Dictionary of retrieval metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not hasattr(model, "memory_extension") or model.memory_extension is None:
        return {"error": "Memory extension not enabled"}
    
    correct_count = 0
    partial_matches = 0
    
    for i, query in enumerate(queries):
        query_type = query.get("type", "entity")
        query_params = query.get("params", {})
        
        # Execute query
        if query_type == "entity":
            result = model.memory_extension.retrieve_entity(**query_params)
        elif query_type == "relation":
            result = model.memory_extension.retrieve_relations(**query_params)
        else:
            continue
        
        # Compare with expected result
        expected = expected_results[i]
        
        # Exact match
        if result == expected:
            correct_count += 1
        # Partial match for lists of entities/relations
        elif isinstance(result, list) and isinstance(expected, list):
            # Calculate overlap
            if expected:
                overlap = len(set(result) & set(expected)) / len(expected)
                if overlap >= 0.5:  # At least 50% match
                    partial_matches += 1
    
    # Calculate metrics
    total_queries = len(queries)
    exact_accuracy = correct_count / total_queries if total_queries else 0
    partial_accuracy = (correct_count + partial_matches) / total_queries if total_queries else 0
    
    return {
        "exact_accuracy": exact_accuracy,
        "partial_accuracy": partial_accuracy,
        "correct_count": correct_count,
        "partial_matches": partial_matches,
        "total_queries": total_queries
    }


def memory_consistency(
    model: torch.nn.Module,
    entity_relations: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Measure consistency of memory operations.
    
    Args:
        model: The model to evaluate
        entity_relations: List of entity and relation data to test
        
    Returns:
        Dictionary of consistency metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not hasattr(model, "memory_extension") or model.memory_extension is None:
        return {"error": "Memory extension not enabled"}
    
    # Reset memory for clean state
    model.memory_extension.reset()
    
    # Add entities and relations to memory
    entity_ids = {}
    relation_success = []
    
    # First add all entities
    for item in entity_relations:
        entity_data = item.get("entity", {})
        if entity_data:
            entity_id = model.memory_extension.add_entity(**entity_data)
            entity_ids[entity_data.get("name", "")] = entity_id
    
    # Then add relations
    for item in entity_relations:
        relation_data = item.get("relation", {})
        if relation_data and "source" in relation_data and "target" in relation_data:
            # Map source and target names to IDs
            source_name = relation_data.get("source", "")
            target_name = relation_data.get("target", "")
            
            if source_name in entity_ids and target_name in entity_ids:
                relation_data["source_id"] = entity_ids[source_name]
                relation_data["target_id"] = entity_ids[target_name]
                del relation_data["source"]
                del relation_data["target"]
                
                success = model.memory_extension.add_relation(**relation_data)
                relation_success.append(success)
    
    # Retrieve and verify
    verification_success = []
    
    for item in entity_relations:
        entity_data = item.get("entity", {})
        if entity_data and "name" in entity_data:
            # Try to retrieve entity
            entity_name = entity_data["name"]
            retrieved = model.memory_extension.retrieve_entity(name=entity_name)
            
            if retrieved and retrieved.get("name") == entity_name:
                verification_success.append(1)
            else:
                verification_success.append(0)
    
    # Calculate metrics
    consistency_score = sum(verification_success) / len(verification_success) if verification_success else 0
    relation_success_rate = sum(relation_success) / len(relation_success) if relation_success else 0
    
    return {
        "consistency_score": consistency_score,
        "relation_success_rate": relation_success_rate,
        "entities_added": len(entity_ids),
        "relations_added": len(relation_success)
    }


def memory_capacity(
    model: torch.nn.Module,
    batch_size: int = 100,
    max_entities: int = 1000
) -> Dict[str, float]:
    """
    Measure memory capacity and scaling.
    
    Args:
        model: The model to evaluate
        batch_size: Number of entities to add in each batch
        max_entities: Maximum number of entities to test
        
    Returns:
        Dictionary of capacity metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not hasattr(model, "memory_extension") or model.memory_extension is None:
        return {"error": "Memory extension not enabled"}
    
    # Reset memory for clean state
    model.memory_extension.reset()
    
    # Generate test entities
    entity_batches = []
    for i in range(0, max_entities, batch_size):
        batch = []
        for j in range(min(batch_size, max_entities - i)):
            entity_id = i + j
            batch.append({
                "name": f"entity_{entity_id}",
                "type": entity_id % 5,  # Randomly assign one of 5 types
                "attributes": {
                    "value": entity_id,
                    "category": ["test", f"category_{entity_id % 10}"]
                }
            })
        entity_batches.append(batch)
    
    # Measure insertion times
    insertion_times = []
    success_rates = []
    
    import time
    for batch in entity_batches:
        successful = 0
        start_time = time.time()
        
        for entity_data in batch:
            entity_id = model.memory_extension.add_entity(**entity_data)
            if entity_id:
                successful += 1
        
        batch_time = time.time() - start_time
        insertion_times.append(batch_time)
        success_rates.append(successful / len(batch))
    
    # Measure retrieval performance
    retrieval_times = []
    retrieval_success = []
    
    # Sample entities to test
    sample_indices = np.random.choice(max_entities, min(100, max_entities), replace=False)
    
    for idx in sample_indices:
        entity_name = f"entity_{idx}"
        
        start_time = time.time()
        retrieved = model.memory_extension.retrieve_entity(name=entity_name)
        retrieval_time = time.time() - start_time
        
        retrieval_times.append(retrieval_time)
        retrieval_success.append(1 if retrieved and retrieved.get("name") == entity_name else 0)
    
    # Get memory statistics
    memory_stats = model.memory_extension.get_statistics() if hasattr(model.memory_extension, "get_statistics") else {}
    
    return {
        "entities_stored": memory_stats.get("total_entries", 0),
        "avg_insertion_time": sum(insertion_times) / len(insertion_times) if insertion_times else 0,
        "avg_retrieval_time": sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0,
        "insertion_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0,
        "retrieval_success_rate": sum(retrieval_success) / len(retrieval_success) if retrieval_success else 0,
        "memory_stats": memory_stats
    }


def evaluate_memory_extension(
    model: torch.nn.Module,
    datasets: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of memory extension.
    
    Args:
        model: The model to evaluate
        datasets: Dictionary containing evaluation datasets
            - "kg_queries": Knowledge graph query dataset
            - "memory_test": Memory consistency test dataset
        
    Returns:
        Dictionary of evaluation results
    """
    results = {}
    
    # Check if extension is enabled
    if not hasattr(model, "memory_extension") or model.memory_extension is None:
        return {"error": "Memory extension not enabled"}
    
    # Reset memory to start with clean state
    if hasattr(model.memory_extension, "reset"):
        model.memory_extension.reset()
    
    # Evaluate knowledge graph retrieval if dataset provided
    if "kg_queries" in datasets:
        kg_data = datasets["kg_queries"]
        results["retrieval_accuracy"] = knowledge_graph_retrieval(
            model,
            kg_data.get("queries", []),
            kg_data.get("expected_results", [])
        )
    
    # Evaluate memory consistency if dataset provided
    if "memory_test" in datasets:
        memory_data = datasets["memory_test"]
        results["memory_consistency"] = memory_consistency(
            model,
            memory_data.get("entity_relations", [])
        )
    
    # Evaluate memory capacity with default parameters
    results["memory_capacity"] = memory_capacity(
        model,
        batch_size=datasets.get("batch_size", 100),
        max_entities=datasets.get("max_entities", 1000)
    )
    
    # Extension-specific metrics
    if hasattr(model.memory_extension, "get_statistics"):
        results["memory_statistics"] = model.memory_extension.get_statistics()
    
    results["extension_info"] = {
        "type": model.memory_extension.type if hasattr(model.memory_extension, "type") else "memory",
        "name": model.memory_extension.name if hasattr(model.memory_extension, "name") else "unknown"
    }
    
    return results