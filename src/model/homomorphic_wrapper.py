"""
Homomorphic Computational Wrapper (HCW) Module.

This module implements the self-evolving memory component described in the 
Semantic Resonance Language Model paper, allowing the model to update knowledge 
on the fly without catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HomomorphicComputationalWrapper(nn.Module):
    """
    Homomorphic Computational Wrapper (HCW) for self-evolving memory.
    
    The HCW functions as a contextual weight generator that maps new data or context
    to parameter deltas, allowing the model to continuously update its knowledge
    while maintaining stability in the base weights.
    
    Formally, it implements:
        W_eff = W_0 + Δ_Φ(C)
    where:
        - W_0 are the base weights
        - Δ_Φ is the contextual weight generator
        - C is the current context or input
    """
    
    def __init__(self, hidden_dim, memory_size=1000, key_dim=128, delta_factor=0.1, dropout=0.1):
        """
        Initialize the Homomorphic Computational Wrapper.
        
        Args:
            hidden_dim (int): Size of the hidden dimension
            memory_size (int): Size of the episodic memory
            key_dim (int): Dimension of the memory keys
            delta_factor (float): Factor controlling the magnitude of weight deltas
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.delta_factor = delta_factor
        
        # Memory key generation network
        self.key_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, key_dim)
        )
        
        # Initialize episodic memory
        # Memory keys have shape [memory_size, key_dim]
        self.memory_keys = nn.Parameter(torch.randn(memory_size, key_dim))
        # Memory values have shape [memory_size, hidden_dim]
        self.memory_values = nn.Parameter(torch.randn(memory_size, hidden_dim))
        
        # Initialize memory using Kaiming initialization
        nn.init.kaiming_normal_(self.memory_keys)
        nn.init.kaiming_normal_(self.memory_values)
        
        # Adapter network for delta generation
        self.adapter_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Normalization for memory lookups
        self.key_norm = nn.LayerNorm(key_dim)
    
    def update_memory(self, keys, values, update_indices=None):
        """
        Update the episodic memory with new key-value pairs.
        
        Args:
            keys (torch.Tensor): New memory keys of shape [batch_size, key_dim]
            values (torch.Tensor): New memory values of shape [batch_size, hidden_dim]
            update_indices (torch.Tensor, optional): Indices in memory to update.
                If None, uses least recently used or random positions.
                
        Returns:
            bool: Whether the memory was successfully updated
        """
        batch_size = keys.shape[0]
        
        # If no update indices provided, find least accessed or random positions
        if update_indices is None:
            # Simple strategy: randomly select positions to update
            update_indices = torch.randint(
                0, self.memory_size, (batch_size,), device=keys.device
            )
        
        # Update memory at specified indices
        self.memory_keys.data[update_indices] = keys
        self.memory_values.data[update_indices] = values
        
        return True
    
    def memory_lookup(self, query_keys):
        """
        Perform memory lookup using attention mechanism.
        
        Args:
            query_keys (torch.Tensor): Query keys of shape [batch_size, seq_len, key_dim]
            
        Returns:
            torch.Tensor: Retrieved memory values of shape [batch_size, seq_len, hidden_dim]
            torch.Tensor: Attention weights of shape [batch_size, seq_len, memory_size]
        """
        batch_size, seq_len, _ = query_keys.shape
        
        # Normalize keys for better attention
        query_keys = self.key_norm(query_keys)
        memory_keys = F.normalize(self.memory_keys, dim=-1)
        
        # Reshape for batch matrix multiplication
        # [batch_size, seq_len, key_dim] @ [memory_size, key_dim]^T
        # -> [batch_size, seq_len, memory_size]
        attn_scores = torch.matmul(
            query_keys, 
            memory_keys.transpose(0, 1)
        )
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores / (self.key_dim ** 0.5), dim=-1)
        
        # Retrieve memory values using attention weights
        # [batch_size, seq_len, memory_size] @ [memory_size, hidden_dim]
        # -> [batch_size, seq_len, hidden_dim]
        retrieved_values = torch.matmul(attn_weights, self.memory_values)
        
        return retrieved_values, attn_weights
    
    def generate_delta(self, x):
        """
        Generate parameter deltas based on input context.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim]
            
        Returns:
            torch.Tensor: Parameter deltas of shape [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Generate memory keys from input
        query_keys = self.key_net(x)  # [batch_size, seq_len, key_dim]
        
        # Lookup values from memory
        memory_values, attn_weights = self.memory_lookup(query_keys)
        
        # Generate parameter deltas using adapter network
        deltas = self.adapter_net(memory_values)  # [batch_size, seq_len, hidden_dim]
        
        # Scale the deltas to control update magnitude
        deltas = deltas * self.delta_factor
        
        return deltas, attn_weights
    
    def forward(self, x, params=None):
        """
        Forward pass of the HCW, generating and applying parameter deltas.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim]
            params (dict, optional): Original parameters to modify. If None, returns only deltas.
            
        Returns:
            torch.Tensor: Modified representation of shape [batch_size, seq_len, hidden_dim]
            dict: Generated deltas if params is None, otherwise the modified parameters
        """
        # Generate deltas based on input context
        deltas, attn_weights = self.generate_delta(x)
        
        # If no params provided, just return the deltas and modified input
        if params is None:
            return x + deltas, {"deltas": deltas, "attn_weights": attn_weights}
        
        # Otherwise, apply deltas to the provided parameters
        modified_params = {}
        for name, param in params.items():
            # This is a simplified update - in practice, you'd need to handle
            # different parameter shapes and specific update strategies
            if len(param.shape) == 2:  # Only update matrices for simplicity
                param_delta = deltas.mean(dim=(0, 1))  # Average across batch and sequence
                # Reshape if needed
                if param_delta.shape[0] != param.shape[0]:
                    param_delta = param_delta[:param.shape[0]]
                modified_params[name] = param + param_delta.view_as(param)
            else:
                modified_params[name] = param
        
        return x + deltas, modified_params