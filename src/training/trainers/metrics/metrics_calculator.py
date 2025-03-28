"""
Metrics calculation for the enhanced training system.

This module provides utilities for calculating various metrics for
model evaluation, including perplexity, accuracy, and specialized
metrics for dialogue and multimodal models.
"""

import math
import logging
from typing import Dict, List, Optional, Union, Any, Callable, Tuple

import torch
import torch.nn.functional as F
import numpy as np


# Get logger
logger = logging.getLogger("quantum_resonance")


def compute_perplexity(
    loss: Union[float, torch.Tensor],
    base: float = math.e
) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss
        base: Logarithm base (default: e for natural log)
        
    Returns:
        Perplexity
    """
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    
    return float(base ** loss)


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute accuracy for classification.
    
    Args:
        predictions: Model predictions (logits or class indices)
        labels: Ground truth labels
        ignore_index: Index to ignore in the accuracy calculation
        
    Returns:
        Accuracy
    """
    if predictions.dim() > 1 and predictions.size(-1) > 1:
        # Convert logits to predictions
        predictions = predictions.argmax(dim=-1)
    
    # Create mask for valid indices
    mask = (labels != ignore_index)
    
    # Calculate accuracy
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item()


def compute_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute token-level accuracy for language modeling.
    
    Args:
        logits: Model output logits (batch_size, seq_len, vocab_size)
        labels: Ground truth labels (batch_size, seq_len)
        ignore_index: Index to ignore in the accuracy calculation
        
    Returns:
        Token accuracy
    """
    predictions = logits.argmax(dim=-1)
    
    # Create mask for valid indices
    mask = (labels != ignore_index)
    
    # Calculate accuracy
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item()


def compute_top_k_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
    ignore_index: int = -100
) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        logits: Model output logits (batch_size, ..., vocab_size)
        labels: Ground truth labels (batch_size, ...)
        k: Number of top predictions to consider
        ignore_index: Index to ignore in the accuracy calculation
        
    Returns:
        Top-k accuracy
    """
    # Get top-k predictions
    _, top_k_indices = torch.topk(logits, k, dim=-1)
    
    # Reshape labels to match top_k_indices
    labels_expanded = labels.unsqueeze(-1).expand_as(top_k_indices)
    
    # Check if label is in top-k predictions
    correct = (top_k_indices == labels_expanded)
    
    # Create mask for valid indices
    mask = (labels != ignore_index).unsqueeze(-1).expand_as(top_k_indices)
    
    # Calculate accuracy
    top_k_acc = (correct & mask).any(dim=-1).sum().float() / mask[:, :, 0].sum().float()
    
    return top_k_acc.item()


def compute_f1_score(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    average: str = "micro",
    ignore_index: Optional[int] = None
) -> float:
    """
    Compute F1 score.
    
    Args:
        predictions: Model predictions (class indices)
        labels: Ground truth labels
        average: Averaging method ('micro', 'macro', 'weighted')
        ignore_index: Index to ignore in the F1 calculation
        
    Returns:
        F1 score
    """
    # Move tensors to CPU and convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Create mask for valid indices
    if ignore_index is not None:
        mask = (labels != ignore_index)
        predictions = predictions[mask]
        labels = labels[mask]
    
    # Get unique classes
    classes = np.unique(np.concatenate([predictions, labels]))
    
    # Compute metrics for each class
    f1_scores = []
    
    for cls in classes:
        tp = np.sum((predictions == cls) & (labels == cls))
        fp = np.sum((predictions == cls) & (labels != cls))
        fn = np.sum((predictions != cls) & (labels == cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    # Average F1 scores
    if average == "micro":
        # Compute global precision and recall
        tp_total = np.sum([(predictions == cls) & (labels == cls) for cls in classes])
        fp_total = np.sum([(predictions == cls) & (labels != cls) for cls in classes])
        fn_total = np.sum([(predictions != cls) & (labels == cls) for cls in classes])
        
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return float(f1)
    
    elif average == "macro":
        # Simple average of class F1 scores
        return float(np.mean(f1_scores))
    
    elif average == "weighted":
        # Weighted average by support
        weights = np.array([np.sum(labels == cls) for cls in classes])
        weights = weights / weights.sum()
        return float(np.sum(weights * np.array(f1_scores)))
    
    else:
        raise ValueError(f"Unsupported average method: {average}")


def compute_bleu_score(
    predictions: List[str],
    references: List[List[str]],
    max_n: int = 4,
    weights: Optional[List[float]] = None
) -> float:
    """
    Compute BLEU score for language generation.
    
    Args:
        predictions: Model predictions (list of strings)
        references: Ground truth references (list of list of strings)
        max_n: Maximum n-gram order
        weights: Weights for different n-gram orders
        
    Returns:
        BLEU score
    """
    try:
        # Try to use nltk's implementation
        from nltk.translate.bleu_score import corpus_bleu
        from nltk.tokenize import word_tokenize
        
        # Tokenize predictions and references
        tokenized_preds = [word_tokenize(pred.lower()) for pred in predictions]
        tokenized_refs = [[word_tokenize(ref.lower()) for ref in refs] for refs in references]
        
        # Compute BLEU score
        bleu = corpus_bleu(tokenized_refs, tokenized_preds, weights=weights)
        return float(bleu)
    
    except ImportError:
        logger.warning("NLTK not installed, falling back to simple BLEU implementation")
        
        # Simple implementation of BLEU
        def count_ngrams(tokens, n):
            ngrams = {}
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                ngrams[ngram] = ngrams.get(ngram, 0) + 1
            return ngrams
        
        def modified_precision(pred_tokens, ref_tokens_list, n):
            pred_ngrams = count_ngrams(pred_tokens, n)
            
            if not pred_ngrams:
                return 0
            
            # Count maximum reference ngrams
            max_ref_counts = {}
            for ref_tokens in ref_tokens_list:
                ref_ngrams = count_ngrams(ref_tokens, n)
                for ngram, count in ref_ngrams.items():
                    max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)
            
            # Compute clipped counts
            clipped_counts = {
                ngram: min(count, max_ref_counts.get(ngram, 0))
                for ngram, count in pred_ngrams.items()
            }
            
            numerator = sum(clipped_counts.values())
            denominator = sum(pred_ngrams.values())
            
            return numerator / denominator if denominator > 0 else 0
        
        # Tokenize (simple space-based tokenization)
        tokenized_preds = [pred.lower().split() for pred in predictions]
        tokenized_refs = [[ref.lower().split() for ref in refs] for refs in references]
        
        # Set default weights if not provided
        if weights is None:
            weights = [1/max_n] * max_n
        
        # Compute scores for each n-gram order
        precisions = []
        for n in range(1, max_n + 1):
            # Skip n-grams larger than prediction length
            if n > min(len(tokens) for tokens in tokenized_preds):
                precisions.append(0.0)
                continue
            
            # Compute precision for each sample
            p_n = [
                modified_precision(pred, refs, n)
                for pred, refs in zip(tokenized_preds, tokenized_refs)
            ]
            precisions.append(np.mean(p_n))
        
        # Apply brevity penalty
        ref_lengths = [min(len(ref) for ref in refs) for refs in tokenized_refs]
        pred_lengths = [len(pred) for pred in tokenized_preds]
        
        bp = np.exp(min(0, 1 - np.sum(ref_lengths) / np.sum(pred_lengths)))
        
        # Compute weighted geometric mean of precisions
        weighted_p = 0
        for p, w in zip(precisions, weights):
            if p > 0:
                weighted_p += w * np.log(p)
        
        bleu = bp * np.exp(weighted_p)
        return float(bleu)


def compute_rouge_score(
    predictions: List[str],
    references: List[str],
    rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"]
) -> Dict[str, float]:
    """
    Compute ROUGE score for text summarization.
    
    Args:
        predictions: Model predictions (list of strings)
        references: Ground truth references (list of strings)
        rouge_types: Types of ROUGE metrics to compute
        
    Returns:
        Dictionary of ROUGE scores
    """
    try:
        # Try to use rouge package
        from rouge import Rouge
        
        # Initialize Rouge
        rouge = Rouge(metrics=rouge_types)
        
        # Compute ROUGE scores
        scores = rouge.get_scores(predictions, references, avg=True)
        
        # Extract F1 scores
        results = {}
        for rouge_type in rouge_types:
            results[rouge_type] = scores[rouge_type]["f"]
        
        return results
    
    except ImportError:
        logger.warning("Rouge package not installed, falling back to simple ROUGE implementation")
        
        # Simple implementation of ROUGE
        def tokenize(text):
            return text.lower().split()
        
        def count_ngrams(tokens, n):
            ngrams = {}
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                ngrams[ngram] = ngrams.get(ngram, 0) + 1
            return ngrams
        
        def rouge_n(pred_tokens, ref_tokens, n):
            pred_ngrams = count_ngrams(pred_tokens, n)
            ref_ngrams = count_ngrams(ref_tokens, n)
            
            # Count matching ngrams
            matches = 0
            for ngram, count in pred_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
            
            # Compute precision and recall
            precision = matches / sum(pred_ngrams.values()) if pred_ngrams else 0
            recall = matches / sum(ref_ngrams.values()) if ref_ngrams else 0
            
            # Compute F1 score
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                "precision": precision,
                "recall": recall,
                "f": f1
            }
        
        def lcs(a, b):
            """Longest common subsequence length."""
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        def rouge_l(pred_tokens, ref_tokens):
            lcs_length = lcs(pred_tokens, ref_tokens)
            
            # Compute precision and recall
            precision = lcs_length / len(pred_tokens) if pred_tokens else 0
            recall = lcs_length / len(ref_tokens) if ref_tokens else 0
            
            # Compute F1 score
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                "precision": precision,
                "recall": recall,
                "f": f1
            }
        
        # Tokenize inputs
        tokenized_preds = [tokenize(pred) for pred in predictions]
        tokenized_refs = [tokenize(ref) for ref in references]
        
        # Compute scores
        results = {}
        for rouge_type in rouge_types:
            if rouge_type == "rouge1":
                scores = [rouge_n(pred, ref, 1) for pred, ref in zip(tokenized_preds, tokenized_refs)]
            elif rouge_type == "rouge2":
                scores = [rouge_n(pred, ref, 2) for pred, ref in zip(tokenized_preds, tokenized_refs)]
            elif rouge_type == "rougeL":
                scores = [rouge_l(pred, ref) for pred, ref in zip(tokenized_preds, tokenized_refs)]
            else:
                logger.warning(f"Unsupported ROUGE type: {rouge_type}")
                continue
            
            # Average scores
            results[rouge_type] = np.mean([score["f"] for score in scores])
        
        return results


def compute_image_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics for image generation or reconstruction.
    
    Args:
        predictions: Predicted images (batch_size, channels, height, width)
        targets: Target images (batch_size, channels, height, width)
        
    Returns:
        Dictionary of image metrics
    """
    # Ensure tensors are on CPU
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu()
    
    # Compute Mean Squared Error
    mse = F.mse_loss(predictions, targets).item()
    
    # Compute Peak Signal to Noise Ratio
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else 100.0
    
    # Compute Structural Similarity Index (simplified)
    def compute_ssim(pred, target):
        # Constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Compute means
        mu_pred = torch.mean(pred)
        mu_target = torch.mean(target)
        
        # Compute variances and covariance
        sigma_pred = torch.var(pred)
        sigma_target = torch.var(target)
        sigma_pred_target = torch.mean((pred - mu_pred) * (target - mu_target))
        
        # Compute SSIM
        ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
        
        return ssim.item()
    
    # Compute SSIM for each image and channel
    ssim_values = []
    for i in range(predictions.size(0)):
        for c in range(predictions.size(1)):
            ssim_values.append(compute_ssim(predictions[i, c], targets[i, c]))
    
    ssim = np.mean(ssim_values)
    
    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim
    }


class MetricsCalculator:
    """
    Calculator for various training and evaluation metrics.
    
    This class provides methods for computing metrics for different model types,
    including language models, dialogue models, and multimodal models.
    """
    
    def __init__(
        self,
        model_type: str = "standard",
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize the metrics calculator.
        
        Args:
            model_type: Type of model ("standard", "dialogue", "multimodal")
            tokenizer: Tokenizer for text processing
        """
        self.model_type = model_type.lower()
        self.tokenizer = tokenizer
        
        # Set up metric functions based on model type
        self.metrics_functions = {
            "perplexity": compute_perplexity,
            "accuracy": compute_token_accuracy,
            "top_k_accuracy": compute_top_k_accuracy
        }
        
        # Add model-specific metrics
        if self.model_type == "dialogue":
            self.metrics_functions.update({
                "bleu": compute_bleu_score,
                "rouge": compute_rouge_score,
                "f1": compute_f1_score
            })
        
        elif self.model_type == "multimodal":
            self.metrics_functions.update({
                "image_metrics": compute_image_metrics
            })
    
    def compute_metrics(
        self,
        outputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        batch: Dict[str, torch.Tensor],
        additional_metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute metrics for model outputs.
        
        Args:
            outputs: Model outputs
            batch: Input batch
            additional_metrics: Additional metrics to compute
            
        Returns:
            Dictionary of computed metrics
        """
        # Extract outputs and labels
        logits = None
        labels = None
        
        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("predictions"))
            
            # Handle different output formats
            if logits is None:
                for key in ["logits", "predictions", "probs", "class_logits"]:
                    if key in outputs:
                        logits = outputs[key]
                        break
        else:
            logits = outputs
        
        # Check if logits are available
        if logits is None:
            logger.warning("No logits found in outputs, metrics computation might be incomplete")
            return {"loss": outputs["loss"].item() if isinstance(outputs, dict) and "loss" in outputs else 0.0}
        
        # Get labels from batch
        if "labels" in batch:
            labels = batch["labels"]
        elif "input_ids" in batch:
            # For causal language modeling, shifted input_ids can be used as labels
            labels = batch["input_ids"][:, 1:] if batch["input_ids"].dim() > 1 else None
        
        # Check if labels are available
        if labels is None:
            logger.warning("No labels found in batch, metrics computation might be incomplete")
            return {"loss": outputs["loss"].item() if isinstance(outputs, dict) and "loss" in outputs else 0.0}
        
        # Start with default metrics
        metrics = {}
        
        # Add loss if available
        if isinstance(outputs, dict) and "loss" in outputs:
            metrics["loss"] = outputs["loss"].item()
            metrics["perplexity"] = compute_perplexity(outputs["loss"])
        
        # Add accuracy if logits and labels are available
        if logits is not None and labels is not None:
            # Ensure logits and labels have compatible shapes
            if logits.dim() - labels.dim() == 1:
                # For token classification, we need to match dimensions
                if self.model_type == "standard" and logits.dim() > 2:
                    logits = logits.view(-1, logits.size(-1))
                    labels = labels.view(-1)
            
            # Compute accuracy
            metrics["accuracy"] = compute_token_accuracy(logits, labels)
            
            # Compute top-5 accuracy if vocabulary is large enough
            if logits.size(-1) > 5:
                metrics["top_5_accuracy"] = compute_top_k_accuracy(logits, labels, k=5)
        
        # Compute model-specific or additional metrics
        if additional_metrics:
            for metric_name in additional_metrics:
                if metric_name in self.metrics_functions:
                    # Get the metric function
                    metric_fn = self.metrics_functions[metric_name]
                    
                    try:
                        # Compute the metric
                        result = metric_fn(logits, labels)
                        
                        # Add to metrics dictionary
                        if isinstance(result, dict):
                            metrics.update(result)
                        else:
                            metrics[metric_name] = result
                    except Exception as e:
                        logger.warning(f"Error computing metric {metric_name}: {e}")
        
        return metrics
    
    def decode_predictions(
        self,
        logits: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode token predictions to text.
        
        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size)
            skip_special_tokens: Whether to skip special tokens in the output
            
        Returns:
            List of decoded predictions
        """
        if self.tokenizer is None:
            logger.warning("No tokenizer available for decoding predictions")
            return []
        
        # Convert logits to token IDs
        predictions = logits.argmax(dim=-1).detach().cpu().numpy()
        
        # Decode predictions
        decoded = []
        for pred in predictions:
            try:
                text = self.tokenizer.decode(pred, skip_special_tokens=skip_special_tokens)
                decoded.append(text)
            except Exception as e:
                logger.warning(f"Error decoding prediction: {e}")
                decoded.append("")
        
        return decoded
    
    def decode_batch(
        self,
        batch: Dict[str, torch.Tensor],
        key: str = "input_ids",
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode batch tensor to text.
        
        Args:
            batch: Input batch
            key: Key for the tensor to decode
            skip_special_tokens: Whether to skip special tokens in the output
            
        Returns:
            List of decoded texts
        """
        if self.tokenizer is None:
            logger.warning("No tokenizer available for decoding batch")
            return []
        
        if key not in batch:
            logger.warning(f"Key {key} not found in batch")
            return []
        
        # Get tensor to decode
        tensor = batch[key].detach().cpu().numpy()
        
        # Decode tensor
        decoded = []
        for ids in tensor:
            try:
                text = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
                decoded.append(text)
            except Exception as e:
                logger.warning(f"Error decoding batch: {e}")
                decoded.append("")
        
        return decoded