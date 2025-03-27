"""
Multimodal extension evaluation metrics for QLLM models.

This module provides metrics specifically for evaluating multimodal
capabilities and performance of QLLM models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union


def multimodal_accuracy(
    model: torch.nn.Module,
    image_text_pairs: List[Dict[str, Any]],
    options: List[str],
    reference_answers: List[str]
) -> Dict[str, float]:
    """
    Measure multimodal accuracy on a visual question answering task.
    
    Args:
        model: The model to evaluate
        image_text_pairs: List of image and question pairs
        options: List of possible answers for each question
        reference_answers: List of correct answers
        
    Returns:
        Dictionary of accuracy metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not hasattr(model, "multimodal_extension") or model.multimodal_extension is None:
        return {"error": "Multimodal extension not enabled"}
    
    correct_count = 0
    
    for i, pair in enumerate(image_text_pairs):
        image = pair["image"]
        question = pair["text"]
        
        # Process image
        vision_features = model.multimodal_extension.process_images([image])[0]
        
        # Compute likelihood for each option
        option_scores = []
        
        for option in options[i]:
            input_text = f"{question} {option}"
            
            # Get model output
            with torch.no_grad():
                output = model(
                    input_text,
                    vision_features=vision_features
                )
            
            # Compute probability of the option given the context
            if isinstance(output, dict) and "logits" in output:
                logits = output["logits"]
            else:
                logits = output
            
            # Simplistic scoring based on final token prediction
            final_logits = logits[0, -1, :]
            option_score = F.softmax(final_logits, dim=0).max().item()
            option_scores.append(option_score)
        
        # Select highest scoring option
        predicted_idx = option_scores.index(max(option_scores))
        predicted_answer = options[i][predicted_idx]
        
        # Check if correct
        if predicted_answer == reference_answers[i]:
            correct_count += 1
    
    # Calculate accuracy
    accuracy = correct_count / len(image_text_pairs) if image_text_pairs else 0
    
    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": len(image_text_pairs)
    }


def image_captioning_quality(
    model: torch.nn.Module,
    images: List[Any],
    reference_captions: List[str],
    max_length: int = 50
) -> Dict[str, float]:
    """
    Evaluate image captioning quality.
    
    Args:
        model: The model to evaluate
        images: List of images to caption
        reference_captions: List of ground truth captions
        max_length: Maximum length of generated captions
        
    Returns:
        Dictionary of captioning quality metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not hasattr(model, "multimodal_extension") or model.multimodal_extension is None:
        return {"error": "Multimodal extension not enabled"}
    
    # Generate captions for each image
    generated_captions = []
    
    for image in images:
        # Process image
        vision_features = model.multimodal_extension.process_images([image])[0]
        
        # Generate caption
        with torch.no_grad():
            caption = model.generate(
                "Describe this image:",
                vision_features=vision_features,
                max_length=max_length
            )
        
        generated_captions.append(caption)
    
    # Compare with reference captions
    # Import here to avoid dependency if not used
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize
        
        bleu_scores = []
        
        for gen_caption, ref_caption in zip(generated_captions, reference_captions):
            gen_tokens = word_tokenize(gen_caption)
            ref_tokens = [word_tokenize(ref_caption)]
            
            bleu = sentence_bleu(ref_tokens, gen_tokens)
            bleu_scores.append(bleu)
        
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        
        return {
            "avg_bleu": avg_bleu,
            "generated_captions": generated_captions
        }
    except ImportError:
        return {
            "error": "NLTK not available for BLEU calculation",
            "generated_captions": generated_captions
        }


def cross_modal_retrieval(
    model: torch.nn.Module,
    images: List[Any],
    texts: List[str]
) -> Dict[str, Any]:
    """
    Evaluate cross-modal retrieval performance.
    
    Args:
        model: The model to evaluate
        images: List of images
        texts: List of texts
        
    Returns:
        Dictionary of retrieval metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not hasattr(model, "multimodal_extension") or model.multimodal_extension is None:
        return {"error": "Multimodal extension not enabled"}
    
    # Process all images
    vision_features = [
        model.multimodal_extension.process_images([image])[0]
        for image in images
    ]
    
    # Process all texts
    text_features = []
    
    for text in texts:
        with torch.no_grad():
            output = model(text)
            hidden_states = output.get("hidden_states", None)
            
            if hidden_states is not None:
                # Use the mean of hidden states as the text feature
                text_feature = hidden_states.mean(dim=1)
                text_features.append(text_feature)
    
    # Calculate similarity matrix
    similarity_matrix = np.zeros((len(vision_features), len(text_features)))
    
    for i, v_feat in enumerate(vision_features):
        for j, t_feat in enumerate(text_features):
            # Simple cosine similarity
            v_feat_flat = v_feat.reshape(-1)
            t_feat_flat = t_feat.reshape(-1)
            
            # Normalize
            v_norm = torch.norm(v_feat_flat)
            t_norm = torch.norm(t_feat_flat)
            
            if v_norm > 0 and t_norm > 0:
                similarity = torch.dot(v_feat_flat, t_feat_flat) / (v_norm * t_norm)
                similarity_matrix[i, j] = similarity.item()
    
    # Calculate retrieval metrics
    # Image to text retrieval
    image_to_text_ranks = []
    for i in range(similarity_matrix.shape[0]):
        # Sort similarities for this image
        sorted_indices = np.argsort(-similarity_matrix[i])
        # Find where the matching text is (assuming i-th text matches i-th image)
        rank = np.where(sorted_indices == i)[0][0] + 1
        image_to_text_ranks.append(rank)
    
    # Text to image retrieval
    text_to_image_ranks = []
    for j in range(similarity_matrix.shape[1]):
        # Sort similarities for this text
        sorted_indices = np.argsort(-similarity_matrix[:, j])
        # Find where the matching image is
        rank = np.where(sorted_indices == j)[0][0] + 1
        text_to_image_ranks.append(rank)
    
    # Calculate recall@k
    r_at_1_i2t = sum(1 for rank in image_to_text_ranks if rank <= 1) / len(image_to_text_ranks)
    r_at_5_i2t = sum(1 for rank in image_to_text_ranks if rank <= 5) / len(image_to_text_ranks)
    
    r_at_1_t2i = sum(1 for rank in text_to_image_ranks if rank <= 1) / len(text_to_image_ranks)
    r_at_5_t2i = sum(1 for rank in text_to_image_ranks if rank <= 5) / len(text_to_image_ranks)
    
    return {
        "r@1_i2t": r_at_1_i2t,
        "r@5_i2t": r_at_5_i2t,
        "r@1_t2i": r_at_1_t2i,
        "r@5_t2i": r_at_5_t2i,
        "median_rank_i2t": np.median(image_to_text_ranks),
        "median_rank_t2i": np.median(text_to_image_ranks)
    }


def evaluate_multimodal_extension(
    model: torch.nn.Module,
    datasets: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of multimodal extension.
    
    Args:
        model: The model to evaluate
        datasets: Dictionary containing evaluation datasets
            - "vqa": Visual Question Answering dataset
            - "captioning": Image Captioning dataset
            - "retrieval": Cross-modal Retrieval dataset
        
    Returns:
        Dictionary of evaluation results
    """
    results = {}
    
    # Check if extension is enabled
    if not hasattr(model, "multimodal_extension") or model.multimodal_extension is None:
        return {"error": "Multimodal extension not enabled"}
    
    # Evaluate VQA if dataset provided
    if "vqa" in datasets:
        vqa_data = datasets["vqa"]
        results["vqa_accuracy"] = multimodal_accuracy(
            model,
            vqa_data.get("image_text_pairs", []),
            vqa_data.get("options", []),
            vqa_data.get("reference_answers", [])
        )
    
    # Evaluate image captioning if dataset provided
    if "captioning" in datasets:
        captioning_data = datasets["captioning"]
        results["captioning_quality"] = image_captioning_quality(
            model,
            captioning_data.get("images", []),
            captioning_data.get("reference_captions", [])
        )
    
    # Evaluate cross-modal retrieval if dataset provided
    if "retrieval" in datasets:
        retrieval_data = datasets["retrieval"]
        results["retrieval_performance"] = cross_modal_retrieval(
            model,
            retrieval_data.get("images", []),
            retrieval_data.get("texts", [])
        )
    
    # Extension-specific metrics
    results["extension_info"] = {
        "type": model.multimodal_extension.type if hasattr(model.multimodal_extension, "type") else "multimodal",
        "name": model.multimodal_extension.name if hasattr(model.multimodal_extension, "name") else "unknown"
    }
    
    return results