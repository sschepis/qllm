

"""
Resonance Attention Module.

This module implements multi-head attention with iterative refinement and 
entropy-based halting as described in the Semantic Resonance Language Model paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResonanceAttention(nn.Module):
    """
    Resonance Attention with iterative refinement and entropy-based halting.
    
    The attention is iterated until the entropy of the attention distribution
    falls below a threshold or the maximum iterations are reached.
    
    Mathematically, for each iteration t:
    - Calculate attention weights
    - Compute entropy H(p_t)
    - If H(p_t) < ε, stop iteration
    - Otherwise, continue refinement
    """
    
    def __init__(self, hidden_dim, num_heads, max_iterations=10, epsilon=3.0, dropout=0.1,
                 momentum=0.2, entropy_penalty=0.05):
        """
        Initialize the Resonance Attention module with improved convergence mechanisms.
        
        Args:
            hidden_dim (int): Size of the hidden dimension
            num_heads (int): Number of attention heads
            max_iterations (int): Maximum number of refinement iterations
            epsilon (float): Entropy threshold for halting (increased from 0.1 to 3.0)
            dropout (float): Dropout probability
            momentum (float): Momentum factor for attention weight updates (0.0-1.0)
                              Higher values give more weight to previous iterations
            entropy_penalty (float): Penalty factor for entropy increases between iterations
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.momentum = momentum
        self.entropy_penalty = entropy_penalty
        
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        self.head_dim = hidden_dim // num_heads
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Resonance bias (learned or fixed)
        self.register_parameter(
            "resonance_bias", 
            nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        )
        
        self.dropout = nn.Dropout(dropout)
    def compute_entropy(self, attention_weights, chunk_size=128):
        """
        Compute Shannon entropy of attention distributions with memory-efficient chunking.
        
        Args:
            attention_weights (torch.Tensor): Attention weights of shape
                [batch_size, num_heads, seq_len, seq_len]
            chunk_size (int): Size of chunks to process at once to reduce memory usage
            
        Returns:
            torch.Tensor: Entropy values of shape [batch_size, num_heads]
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Process in chunks to avoid OOM errors for large batch sizes or sequences
        entropy_chunks = []
        for i in range(0, seq_len, chunk_size):
            # Process a chunk of the sequence dimension
            end_idx = min(i + chunk_size, seq_len)
            # Get chunk of attention weights
            chunk = attention_weights[:, :, i:end_idx, :]
            
            # Add small epsilon to avoid log(0)
            probs = chunk + 1e-10
            
            # Compute partial entropy sum: -∑ p_i * log(p_i) for this chunk
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Use lower precision for log computation to save memory
                chunk_entropy = -torch.sum(probs * torch.log(probs), dim=-1)  # [batch_size, num_heads, chunk_size]
            
            entropy_chunks.append(chunk_entropy)
        
        # Concatenate chunks along sequence dimension
        if len(entropy_chunks) > 1:
            entropy = torch.cat(entropy_chunks, dim=2)
        else:
            entropy = entropy_chunks[0]
        
        # Average over sequence length to get entropy per head
        entropy = entropy.mean(dim=-1)  # [batch_size, num_heads]
        
        return entropy
        return entropy
    
    def forward(self, x, attention_mask=None, return_attn_weights=False):
        """
        Forward pass with iterative refinement and entropy-based halting.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim]
            attention_mask (torch.Tensor, optional): Attention mask of shape 
                [batch_size, 1, 1, seq_len]
            return_attn_weights (bool): Whether to return attention weights
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_dim]
            dict: Metadata including entropy, iterations used, and optionally attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Initial projections
        q = self.query(x)  # [batch_size, seq_len, hidden_dim]
        k = self.key(x)    # [batch_size, seq_len, hidden_dim]
        v = self.value(x)  # [batch_size, seq_len, hidden_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Initialize outputs and metadata
        output = torch.zeros_like(x)
        metadata = {
            "iterations": torch.zeros(batch_size, dtype=torch.long, device=x.device),
            "entropy": torch.zeros(batch_size, self.num_heads, device=x.device),
            "entropy_history": [], # Track entropy across iterations
            "entropy_threshold": self.epsilon,
            "convergence_gap": torch.zeros(batch_size, device=x.device), # How far from convergence
            "entropy_penalties": [] # Track any penalties applied for entropy increases
        }
        
        # Track previous attention weights for momentum
        prev_attn_weights = None
        prev_entropy = None
        
        # Iterative attention refinement with memory-efficient chunking
        for t in range(self.max_iterations):
            # Compute attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Add resonance bias
            attn_scores = attn_scores + self.resonance_bias
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            
            # Compute attention weights
            current_attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
            
            # Apply momentum if this isn't the first iteration
            if prev_attn_weights is not None and self.momentum > 0:
                # Mix current weights with previous weights based on momentum factor
                attn_weights = (1 - self.momentum) * current_attn_weights + self.momentum * prev_attn_weights
                # Re-normalize to ensure proper probability distribution
                attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
            else:
                attn_weights = current_attn_weights
                
            # Store for next iteration's momentum
            prev_attn_weights = attn_weights.detach()
            
            # Apply dropout
            attn_weights = self.dropout(attn_weights)
            
            # Calculate entropy of attention weights (with memory optimization for large batches)
            # We could implement chunking here for very large seq_len
            entropy = self.compute_entropy(attn_weights)  # [batch_size, num_heads]
            
            # Apply entropy penalty if entropy increased from previous iteration
            entropy_penalty_applied = False
            if prev_entropy is not None and self.entropy_penalty > 0:
                # Check if entropy increased for any batch element
                entropy_increases = entropy.mean(dim=1) > prev_entropy.mean(dim=1)
                if entropy_increases.any():
                    # Apply penalty by scaling back increases
                    penalty_factor = self.entropy_penalty
                    penalty_mask = entropy_increases.unsqueeze(1).expand_as(entropy)
                    entropy = torch.where(
                        penalty_mask,
                        entropy * (1 - penalty_factor) + prev_entropy * penalty_factor,
                        entropy
                    )
                    entropy_penalty_applied = True
                    metadata["entropy_penalties"].append({
                        "iteration": t + 1,
                        "num_penalized": entropy_increases.sum().item()
                    })
            
            # Store current entropy for the next iteration
            prev_entropy = entropy.detach()
            
            # Average entropy across heads for halting decision
            mean_entropy = entropy.mean(dim=1)  # [batch_size]
            
            # Store current entropy
            metadata["entropy"] = entropy
            
            # Store detailed entropy history for analysis
            metadata["entropy_history"].append({
                "iteration": t + 1,
                "entropy_per_head": entropy.detach().cpu(),
                "mean_entropy": mean_entropy.detach().cpu(),
                "penalty_applied": entropy_penalty_applied
            })
            
            # Compute weighted values
            attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
            
            # Reshape attention output
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
            
            # Apply output projection
            output = self.output_projection(attn_output)
            
            # Calculate convergence gap (how far we are from threshold)
            convergence_gap = mean_entropy - self.epsilon
            metadata["convergence_gap"] = convergence_gap
            
            # Update iteration count for each sample
            metadata["iterations"] = torch.maximum(
                metadata["iterations"],
                torch.full_like(metadata["iterations"], t + 1)
            )
            
            # Check if all samples are below threshold
            if (mean_entropy < self.epsilon).all():
                break
        
        # If requested, include attention weights in metadata
        if return_attn_weights:
            metadata["attention_weights"] = attn_weights
        
        return output, metadata