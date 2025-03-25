

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
    Advanced Resonance Attention with iterative refinement and multi-criteria convergence.
    
    This enhanced implementation includes:
    - Iterative phase modulation to shift attention perspective across iterations
    - Temperature scheduling to progressively sharpen attention distribution
    - Cosine similarity convergence detection for more reliable halting
    - Momentum-based attention updates to stabilize convergence
    
    Mathematically, for each iteration t:
    1. Apply phase modulation to Q/K: Q^(t) = Q + Δ(t)
    2. Compute attention with temperature: α^(t) = softmax(β(t) * QK^T)
    3. Check cosine similarity: cos(α^(t), α^(t-1))
    4. If similarity > threshold or entropy < ε, stop iteration
    5. Otherwise, continue refinement
    """
    
    def __init__(self, hidden_dim, num_heads, max_iterations=10, epsilon=3.0, dropout=0.1,
                 momentum=0.2, entropy_penalty=0.05,
                 use_phase_modulation=True, phase_factor=0.1, freq_factor=2.0, phase_offset=0.0,
                 use_temperature_scheduling=True, beta_0=0.8, beta_delta=0.2,
                 use_cosine_convergence=True, convergence_threshold=0.95):
        """
        Initialize the Advanced Resonance Attention module with improved convergence mechanisms.
        
        Args:
            hidden_dim (int): Size of the hidden dimension
            num_heads (int): Number of attention heads
            max_iterations (int): Maximum number of refinement iterations
            epsilon (float): Entropy threshold for halting
            dropout (float): Dropout probability
            momentum (float): Momentum factor for attention weight updates (0.0-1.0)
                              Higher values give more weight to previous iterations
            entropy_penalty (float): Penalty factor for entropy increases between iterations
            
            # Phase modulation parameters
            use_phase_modulation (bool): Whether to use phase modulation
            phase_factor (float): Scale of the phase modulation
            freq_factor (float): Frequency of the sinusoidal modulation
            phase_offset (float): Offset of the phase modulation
            
            # Temperature scheduling parameters
            use_temperature_scheduling (bool): Whether to use temperature scheduling
            beta_0 (float): Initial inverse temperature (higher = sharper)
            beta_delta (float): Rate of temperature decrease per iteration
            
            # Cosine convergence parameters
            use_cosine_convergence (bool): Whether to use cosine similarity for convergence
            convergence_threshold (float): Threshold for cosine similarity convergence
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
        
        # Phase modulation parameters
        self.use_phase_modulation = use_phase_modulation
        self.phase_factor = phase_factor
        self.freq_factor = freq_factor
        self.phase_offset = phase_offset
        
        # Temperature scheduling parameters
        self.use_temperature_scheduling = use_temperature_scheduling
        self.beta_0 = beta_0
        self.beta_delta = beta_delta
        
        # Cosine convergence parameters
        self.use_cosine_convergence = use_cosine_convergence
        self.convergence_threshold = convergence_threshold
        
        # Learnable phase parameters
        if self.use_phase_modulation:
            # Phase modulation factors specific to each head
            self.register_parameter(
                "phase_factors",
                nn.Parameter(torch.ones(1, num_heads, 1, 1) * phase_factor)
            )
        
        self.dropout = nn.Dropout(dropout)
        
        # Log initialization parameters
        print(f"Initialized ResonanceAttention with:")
        print(f"  - max_iterations: {max_iterations}")
        print(f"  - epsilon: {epsilon}")
        print(f"  - momentum: {momentum}")
        print(f"  - use_phase_modulation: {use_phase_modulation} (factor={phase_factor})")
        print(f"  - use_temperature_scheduling: {use_temperature_scheduling} (beta_0={beta_0}, delta={beta_delta})")
        print(f"  - use_cosine_convergence: {use_cosine_convergence} (threshold={convergence_threshold})")
    def compute_entropy(self, attention_weights, chunk_size=128):
        """
        Compute Shannon entropy of attention distributions with memory-efficient chunking.
        
        Args:
            attention_weights (torch.Tensor): Attention weights tensor
                Supports multiple shapes including 5D tensors from multi-block attention
            chunk_size (int): Size of chunks to process at once to reduce memory usage
            
        Returns:
            torch.Tensor: Entropy values of shape [batch_size, num_heads]
        """

        # Handle various tensor shapes more robustly
        shape = attention_weights.shape
        
        if len(shape) == 5:
            # Handle 5D case: [batch_size, blocks, heads_per_block, seq_len, seq_len]
            batch_size, blocks, heads_per_block, seq_len, _ = shape
            
            # Reshape to 4D by combining blocks and heads dimensions
            reshaped_attn = attention_weights.reshape(batch_size, blocks * heads_per_block, seq_len, seq_len)

            # Use chunking for memory efficiency
            entropy_chunks = []
            for i in range(0, seq_len, chunk_size):
                end_idx = min(i + chunk_size, seq_len)
                chunk = reshaped_attn[:, :, i:end_idx, :]
                
                # Add small epsilon to avoid log(0)
                probs = chunk + 1e-10
                
                # Compute entropy for this chunk
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    chunk_entropy = -torch.sum(probs * torch.log(probs), dim=-1)
                
                entropy_chunks.append(chunk_entropy)
            
            # Combine chunks
            if len(entropy_chunks) > 1:
                entropy = torch.cat(entropy_chunks, dim=2)
            else:
                entropy = entropy_chunks[0]
            
            # Average over sequence dimension
            entropy = entropy.mean(dim=-1)  # [batch_size, blocks*heads_per_block]
            
            # Reshape back to [batch_size, blocks, heads_per_block]
            entropy = entropy.reshape(batch_size, blocks, heads_per_block)
            
            # Average across blocks to get [batch_size, heads_per_block]
            entropy = entropy.mean(dim=1)

        elif len(shape) == 4:
            batch_size, num_heads, seq_len, _ = shape
            
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
            
        elif len(shape) == 3:
            # Handle case where attention is already flattened (batch_size, num_heads, seq_len*seq_len)
            batch_size, num_heads, flattened_len = shape
            
            # Add small epsilon to avoid log(0)
            probs = attention_weights + 1e-10
            
            # Compute entropy directly
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                entropy = -torch.sum(probs * torch.log(probs), dim=-1) / flattened_len
                
        elif len(shape) == 2:
            # Handle case where batch dimension is flattened (batch_size*num_heads, seq_len*seq_len)
            flattened_batch, flattened_len = shape
            
            # Assume first dimension is batch_size * num_heads
            # We need the original num_heads value to reshape correctly
            if hasattr(self, 'num_heads'):
                num_heads = self.num_heads
                batch_size = flattened_batch // num_heads
                
                # Add small epsilon to avoid log(0)
                probs = attention_weights + 1e-10
                
                # Compute entropy directly
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    entropy_flat = -torch.sum(probs * torch.log(probs), dim=-1) / flattened_len
                    entropy = entropy_flat.view(batch_size, num_heads)
            else:
                # Fallback if we can't reshape
                probs = attention_weights + 1e-10
                entropy = -torch.sum(probs * torch.log(probs), dim=-1)
                print(f"WARNING: Couldn't reshape entropy to [batch_size, num_heads]")
        else:
            raise ValueError(f"Unexpected attention_weights shape: {shape}, needs to be 2D, 3D, 4D, or 5D")
        
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
            "entropy_penalties": [], # Track any penalties applied for entropy increases
            "cosine_similarities": [], # Track cosine similarities between iterations
            "temperatures": [], # Track temperatures used in each iteration
            "converged_samples": 0, # Count of samples that converged early
            "phase_modulations": []  # Track phase modulations applied
        }
        
        # Track previous attention weights for momentum and cosine similarity
        prev_attn_weights = None
        prev_entropy = None
        
        # Initialize cosine similarity tracking
        cos_sim = None
        
        # Iterative attention refinement with memory-efficient chunking
        for t in range(self.max_iterations):
            # Apply phase modulation if enabled
            if self.use_phase_modulation:
                # Compute phase modulation factors based on iteration
                t_norm = t / self.max_iterations  # normalized iteration progress [0-1]
                
                # Convert scalar values to tensors with appropriate device
                phase_angle = torch.tensor(
                    self.freq_factor * t_norm * math.pi + self.phase_offset,
                    device=q.device
                )
                
                # Apply sin to create the modulation
                if hasattr(self, "phase_factors"):
                    # Use the learned per-head factors
                    phase_mod = self.phase_factors * torch.sin(phase_angle)
                    # Record phase modulation applied for debugging
                    metadata["phase_modulations"].append(phase_mod.mean().item())
                else:
                    # Use the global factor
                    phase_scalar = torch.tensor(self.phase_factor, device=q.device)
                    phase_mod = phase_scalar * torch.sin(phase_angle)
                    metadata["phase_modulations"].append(phase_scalar.item() * torch.sin(phase_angle).item())
                
                # Create modulation that matches the tensor shape (expand across batch and sequence dimensions)
                phase_mod = phase_mod.unsqueeze(-1)  # [1, num_heads, 1, 1]
                
                # Apply the phase modulation to both query and key
                q_t = q + phase_mod  # Broadcasting will handle dimension expansion
                k_t = k + phase_mod
            else:
                q_t = q
                k_t = k
            
            # Compute attention scores
            attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Add resonance bias
            attn_scores = attn_scores + self.resonance_bias
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            
            # Apply temperature scheduling if enabled
            if self.use_temperature_scheduling:
                # Compute temperature factor that sharpens distribution with each iteration
                beta_t = self.beta_0 + t * self.beta_delta
                metadata["temperatures"].append(beta_t)
                # Scale logits by inverse temperature (higher beta = sharper distribution)
                attn_scores = attn_scores * beta_t
            
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
            
            # Compute cosine similarity with previous iteration if cosine convergence is enabled
            if self.use_cosine_convergence and prev_attn_weights is not None:
                # Flatten seq_len dimensions to compute similarity between attention distributions
                flat_current = attn_weights.flatten(2)  # [batch_size, num_heads, seq_len*seq_len]
                flat_prev = prev_attn_weights.flatten(2)  # [batch_size, num_heads, seq_len*seq_len]
                
                # Compute cosine similarity (1.0 = identical distributions)
                cos_sim = F.cosine_similarity(flat_current, flat_prev, dim=2)  # [batch_size, num_heads]
                
                # Store for analysis
                metadata["cosine_similarities"].append(cos_sim.detach().cpu().mean().item())
                
                # Check for convergence based on cosine similarity (per sample)
                # High similarity means attention distribution has stabilized
                converged_by_similarity = (cos_sim > self.convergence_threshold).all(dim=1)  # [batch_size]
            else:
                cos_sim = None
                converged_by_similarity = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
            
            # Average entropy across heads for halting decision
            mean_entropy = entropy.mean(dim=1)  # [batch_size]
            
            # Store current entropy
            metadata["entropy"] = entropy
            
            # Store detailed entropy history for analysis
            entry = {
                "iteration": t + 1,
                "entropy_per_head": entropy.detach().cpu(),
                "mean_entropy": mean_entropy.detach().cpu(),
                "penalty_applied": entropy_penalty_applied
            }
            
            # Add cosine similarity to history if available
            if cos_sim is not None:
                entry["cosine_similarity"] = cos_sim.detach().cpu()
                
            # Add temperature to history if temperature scheduling is enabled
            if self.use_temperature_scheduling:
                entry["temperature"] = self.beta_0 + t * self.beta_delta
                
            metadata["entropy_history"].append(entry)
            
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
            
            # Check for convergence by either entropy threshold or cosine similarity
            converged_by_entropy = (mean_entropy < self.epsilon)
            
            # Combine convergence criteria (OR operation)
            converged = converged_by_entropy | converged_by_similarity
            
            # Count samples that converged
            newly_converged = converged.sum().item()
            if newly_converged > 0:
                metadata["converged_samples"] += newly_converged
                # Log which convergence mechanism triggered
                entropy_triggers = converged_by_entropy.sum().item()
                similarity_triggers = (converged_by_similarity & ~converged_by_entropy).sum().item()
                print(f"Iteration {t+1}: {entropy_triggers} samples converged by entropy, " +
                      f"{similarity_triggers} by similarity")
            
            # Break if all samples have converged
            if converged.all():
                break
        
        # If requested, include attention weights in metadata
        if return_attn_weights:
            metadata["attention_weights"] = attn_weights
        
        return output, metadata