"""
Pre-Manifest Resonance Layer Module.

This module implements the final specialized block that refines outputs in superposition,
then collapses to a final distribution, as described in the Semantic Resonance Language Model paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreManifestResonanceLayer(nn.Module):
    """
    Pre-Manifest Resonance Layer that attends over vocabulary embeddings and refines
    the output representation before final distribution.
    
    This layer ensures the model "thinks twice" about the final token, similar to
    a quantum wavefunction being measured. If uncertain, it resonates through more
    iterations until entropy is minimized.
    """
    
    def __init__(self, hidden_dim, vocab_size, embedding_weight=None, 
                 max_iterations=5, epsilon=0.05, dropout=0.1):
        """
        Initialize the Pre-Manifest Resonance Layer.
        
        Args:
            hidden_dim (int): Size of the hidden dimension
            vocab_size (int): Size of the vocabulary
            embedding_weight (torch.Tensor, optional): Weight matrix from embeddings to share parameters
            max_iterations (int): Maximum number of refinement iterations
            epsilon (float): Entropy threshold for halting
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        
        # If shared weights are provided, use them, otherwise create new weights
        if embedding_weight is not None:
            self.embedding_weight = embedding_weight
        else:
            self.embedding_weight = nn.Parameter(torch.randn(vocab_size, hidden_dim))
            nn.init.normal_(self.embedding_weight, mean=0.0, std=0.02)
        
        # Query projection for attention
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def compute_entropy(self, probs):
        """
        Compute Shannon entropy of token distribution.
        
        Args:
            probs (torch.Tensor): Token probabilities of shape [batch_size, seq_len, vocab_size]
            
        Returns:
            torch.Tensor: Entropy values of shape [batch_size, seq_len]
        """
        # Add small epsilon to avoid log(0)
        probs = probs + 1e-10
        
        # Compute entropy: -∑ p_i * log(p_i)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)  # [batch_size, seq_len]
        
        return entropy
    
    def forward(self, hidden_states, attention_mask=None, return_iterations=False):
        """
        Forward pass with iterative refinement until entropy is minimized.
        
        Args:
            hidden_states (torch.Tensor): Hidden states from previous layer [batch_size, seq_len, hidden_dim]
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]
            return_iterations (bool): Whether to return iteration count
            
        Returns:
            torch.Tensor: Logits of shape [batch_size, seq_len, vocab_size]
            dict: Metadata including entropy and iterations
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Apply layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Initialize metadata
        metadata = {
            "entropy": torch.zeros(batch_size, seq_len, device=hidden_states.device),
            "iterations": torch.zeros(batch_size, seq_len, dtype=torch.long, device=hidden_states.device)
        }
        
        # Initial query projection
        query = self.query_proj(hidden_states)  # [batch_size, seq_len, hidden_dim]
        
        # Iterative refinement
        for t in range(self.max_iterations):
            # Compute attention scores with vocabulary embeddings
            # [batch_size, seq_len, hidden_dim] @ [vocab_size, hidden_dim]^T -> [batch_size, seq_len, vocab_size]
            attn_scores = torch.matmul(query, self.embedding_weight.transpose(0, 1))
            
            # Add bias
            attn_scores = attn_scores + self.bias
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Convert mask to additive mask (0 for tokens to keep, large negative for tokens to mask)
                additive_mask = (1.0 - attention_mask.unsqueeze(-1)) * -10000.0
                attn_scores = attn_scores + additive_mask
            
            # Compute token probabilities
            token_probs = F.softmax(attn_scores, dim=-1)  # [batch_size, seq_len, vocab_size]
            
            # Compute entropy of token distribution
            entropy = self.compute_entropy(token_probs)  # [batch_size, seq_len]
            metadata["entropy"] = entropy
            
            # Update iteration count
            metadata["iterations"] = torch.maximum(
                metadata["iterations"],
                torch.full_like(metadata["iterations"], t + 1)
            )
            
            # Check entropy-based halting condition
            if (entropy < self.epsilon).all():
                break
            
            # If not halted, compute weighted embeddings for next iteration
            # [batch_size, seq_len, vocab_size] @ [vocab_size, hidden_dim] -> [batch_size, seq_len, hidden_dim]
            weighted_embeddings = torch.matmul(token_probs, self.embedding_weight)
            
            # Update query for next iteration
            query = query + self.dropout(self.output_proj(weighted_embeddings))
        
        # Final attention scores for logits
        logits = torch.matmul(query, self.embedding_weight.transpose(0, 1)) + self.bias
        
        return logits, metadata