"""
Quantum-Inspired Modal Entanglement Module.

This module implements quantum-inspired techniques to model and leverage
entangled relationships between different modalities, enabling deeper
cross-modal interactions and understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union
import math
import numpy as np


class QuantumStateEncoder(nn.Module):
    """
    Encodes modality features into quantum-inspired state representations.
    
    This module maps classical neural representations to quantum-inspired
    state representations that can exhibit entanglement properties.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        quantum_dim: int = 64,
        num_qubits: int = 6,
        use_complex: bool = True
    ):
        """
        Initialize the quantum state encoder.
        
        Args:
            embedding_dim: Dimension of input embeddings
            quantum_dim: Dimension of quantum state representation
            num_qubits: Number of virtual qubits to simulate
            use_complex: Whether to use complex-valued representations
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.quantum_dim = quantum_dim
        self.num_qubits = num_qubits
        self.use_complex = use_complex
        
        # Calculate required state dimension (2^num_qubits)
        self.state_dim = 2 ** num_qubits
        
        # Projection to quantum amplitude space
        self.real_projection = nn.Sequential(
            nn.Linear(embedding_dim, self.state_dim),
            nn.Tanh()  # Use tanh to constrain to [-1, 1]
        )
        
        if use_complex:
            self.imag_projection = nn.Sequential(
                nn.Linear(embedding_dim, self.state_dim),
                nn.Tanh()  # Use tanh to constrain to [-1, 1]
            )
        
        # Phase encoding (for quantum phases)
        self.phase_encoding = nn.Sequential(
            nn.Linear(embedding_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, self.state_dim),
            nn.Tanh()
        )
        
        # Amplitude normalization factor (learned)
        self.normalization = nn.Parameter(torch.ones(1))
        
        # Quantum dimension reduction (if needed)
        if self.state_dim != quantum_dim:
            self.quantum_projection = nn.Linear(self.state_dim, quantum_dim)
    
    def normalize_state(
        self,
        real_amplitudes: torch.Tensor,
        imag_amplitudes: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Normalize quantum state to ensure unit norm.
        
        Args:
            real_amplitudes: Real part of quantum amplitudes
            imag_amplitudes: Imaginary part of quantum amplitudes
            
        Returns:
            Normalized real and imaginary amplitudes
        """
        if imag_amplitudes is not None:
            # For complex representation: |ψ|² = ∑(real² + imag²) = 1
            squared_norm = torch.sum(
                real_amplitudes ** 2 + imag_amplitudes ** 2,
                dim=-1, keepdim=True
            )
        else:
            # For real representation: ∑|ψ|² = 1
            squared_norm = torch.sum(real_amplitudes ** 2, dim=-1, keepdim=True)
        
        # Add small epsilon to avoid division by zero
        norm = torch.sqrt(squared_norm + 1e-12)
        
        # Normalize
        real_normalized = real_amplitudes / norm
        
        if imag_amplitudes is not None:
            imag_normalized = imag_amplitudes / norm
            return real_normalized, imag_normalized
        
        return real_normalized, None
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode classical features into quantum state representation.
        
        Args:
            features: Input features [batch_size, seq_len, embedding_dim]
            
        Returns:
            Dictionary with quantum state representations
        """
        batch_size = features.shape[0]
        seq_len = features.shape[1] if features.dim() > 2 else 1
        
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        # Project to quantum amplitude space
        real_amplitudes = self.real_projection(features)
        
        if self.use_complex:
            imag_amplitudes = self.imag_projection(features)
        else:
            imag_amplitudes = None
        
        # Get quantum phases
        phases = self.phase_encoding(features) * math.pi  # Scale to [-π, π]
        
        # Apply phases to amplitudes
        phase_factor_real = torch.cos(phases)
        phase_factor_imag = torch.sin(phases)
        
        if self.use_complex:
            # Apply phases to complex amplitudes
            # (a+bi)(cos(θ)+sin(θ)i) = a*cos(θ) - b*sin(θ) + (a*sin(θ) + b*cos(θ))i
            real_with_phase = real_amplitudes * phase_factor_real - imag_amplitudes * phase_factor_imag
            imag_with_phase = real_amplitudes * phase_factor_imag + imag_amplitudes * phase_factor_real
        else:
            # Apply phases to real amplitudes
            real_with_phase = real_amplitudes * phase_factor_real
            imag_with_phase = real_amplitudes * phase_factor_imag
        
        # Normalize the quantum state
        real_normalized, imag_normalized = self.normalize_state(real_with_phase, imag_with_phase)
        
        # Apply quantum dimension reduction if needed
        if hasattr(self, 'quantum_projection'):
            real_normalized = self.quantum_projection(real_normalized)
            if imag_normalized is not None:
                imag_normalized = self.quantum_projection(imag_normalized)
        
        # Prepare output
        results = {
            "real": real_normalized,
            "phases": phases
        }
        
        if imag_normalized is not None:
            results["imag"] = imag_normalized
        
        return results


class ModalEntanglementLayer(nn.Module):
    """
    Creates and leverages entanglement between modalities.
    
    This layer implements quantum-inspired operations to create entangled
    representations across different modalities and leverage them for
    enhanced cross-modal understanding.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        quantum_dim: int = 64,
        num_qubits: int = 6,
        num_unitary_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize the modal entanglement layer.
        
        Args:
            embedding_dim: Dimension of input embeddings
            quantum_dim: Dimension of quantum state representation
            num_qubits: Number of virtual qubits to simulate
            num_unitary_layers: Number of unitary transformation layers
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.quantum_dim = quantum_dim
        self.num_qubits = num_qubits
        
        # Quantum state encoders for each modality
        self.state_encoders = nn.ModuleDict({
            "vision": QuantumStateEncoder(embedding_dim, quantum_dim, num_qubits),
            "text": QuantumStateEncoder(embedding_dim, quantum_dim, num_qubits),
            "audio": QuantumStateEncoder(embedding_dim, quantum_dim, num_qubits)
        })
        
        # "Unitary" transformations (approximated with neural networks)
        self.unitary_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(quantum_dim * 2, quantum_dim * 2),
                nn.LayerNorm(quantum_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(quantum_dim * 2, quantum_dim * 2)
            ) for _ in range(num_unitary_layers)
        ])
        
        # Measurement and back-projection to embedding space
        self.measurement_projections = nn.ModuleDict({
            "vision": nn.Linear(quantum_dim, embedding_dim),
            "text": nn.Linear(quantum_dim, embedding_dim),
            "audio": nn.Linear(quantum_dim, embedding_dim)
        })
        
        # Entanglement measurement operators
        self.entanglement_operators = nn.ParameterDict({
            f"{m1}_{m2}": nn.Parameter(torch.randn(quantum_dim, quantum_dim))
            for m1 in ["vision", "text", "audio"]
            for m2 in ["vision", "text", "audio"]
            if m1 != m2
        })
        
        # Initialize entanglement operators as approximately Hermitian
        for name, param in self.entanglement_operators.items():
            nn.init.xavier_normal_(param)
            with torch.no_grad():
                # Make approximately Hermitian: (M + M^T) / 2
                param.copy_((param + param.transpose(-1, -2)) / 2.0)
    
    def apply_entanglement(
        self,
        quantum_states: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Apply entangling operations to quantum states.
        
        Args:
            quantum_states: Dictionary of quantum states per modality
                {modality: {"real": tensor, "imag": tensor, "phases": tensor}}
                
        Returns:
            Dictionary of entangled quantum states
        """
        entangled_states = {}
        
        # Create pairs of modalities for entanglement
        modalities = list(quantum_states.keys())
        if len(modalities) < 2:
            return quantum_states  # No entanglement possible with single modality
        
        # Group states by pairs
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities[i+1:], i+1):
                # Get states for each modality
                state1_real = quantum_states[mod1]["real"]
                state2_real = quantum_states[mod2]["real"]
                
                # Check dimensions and shapes
                batch_size = state1_real.shape[0]
                seq_len1 = state1_real.shape[1]
                seq_len2 = state2_real.shape[1]
                
                # Reshape for sequence-wise processing if needed
                if seq_len1 > 1 or seq_len2 > 1:
                    # For simplicity, we'll process each position in sequence
                    # In practice, you might want to handle this differently
                    flat_state1 = state1_real.view(-1, self.quantum_dim)
                    flat_state2 = state2_real.view(-1, self.quantum_dim)
                    
                    # Combine states for the unitary operations
                    # For simplicity, we'll just combine corresponding positions
                    # In a real implementation, you might use attention or other mechanisms
                    min_seq_len = min(seq_len1, seq_len2)
                    combined_states = torch.cat([
                        flat_state1[:batch_size*min_seq_len],
                        flat_state2[:batch_size*min_seq_len]
                    ], dim=-1)
                else:
                    # Simple case: single representation per modality
                    combined_states = torch.cat([state1_real, state2_real], dim=-1)
                
                # Apply unitary operations
                entangled_combined = combined_states
                for layer in self.unitary_layers:
                    entangled_combined = combined_states + layer(combined_states)  # Residual connection
                
                # Split back to individual modalities
                entangled_state1, entangled_state2 = torch.chunk(entangled_combined, 2, dim=-1)
                
                # Reshape back if needed
                if seq_len1 > 1 or seq_len2 > 1:
                    entangled_state1 = entangled_state1.view(batch_size, min_seq_len, self.quantum_dim)
                    entangled_state2 = entangled_state2.view(batch_size, min_seq_len, self.quantum_dim)
                
                # Store entangled states
                if mod1 not in entangled_states:
                    entangled_states[mod1] = {"real": entangled_state1}
                if mod2 not in entangled_states:
                    entangled_states[mod2] = {"real": entangled_state2}
        
        return entangled_states
    
    def measure_entanglement(
        self,
        quantum_states: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Measure the degree of entanglement between modalities.
        
        Args:
            quantum_states: Dictionary of quantum states per modality
                
        Returns:
            Dictionary of entanglement measurements between modality pairs
        """
        entanglement_measures = {}
        
        # Measure entanglement between each pair of modalities
        modalities = list(quantum_states.keys())
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if mod1 == mod2:
                    continue
                
                key = f"{mod1}_{mod2}"
                if key not in self.entanglement_operators:
                    continue
                
                # Get states
                state1 = quantum_states[mod1]["real"]
                state2 = quantum_states[mod2]["real"]
                
                # Shape adaptations
                batch_size = state1.shape[0]
                
                # Apply entanglement operator
                entanglement_op = self.entanglement_operators[key]
                
                # Measure entanglement using density matrix approach
                # ⟨ψ1|M|ψ2⟩, where M is the entanglement operator
                if state1.dim() > 2 and state2.dim() > 2:
                    # Sequence case
                    seq_len1, seq_len2 = state1.shape[1], state2.shape[1]
                    
                    # Reshape for batch matrix multiplication
                    flat_state1 = state1.reshape(-1, self.quantum_dim, 1)
                    flat_state2 = state2.reshape(-1, 1, self.quantum_dim)
                    
                    # Broadcast operator to match batch dimensions
                    op_expanded = entanglement_op.unsqueeze(0).expand(
                        flat_state1.shape[0], self.quantum_dim, self.quantum_dim
                    )
                    
                    # Compute entanglement measure
                    # |⟨ψ1|M|ψ2⟩|
                    entanglement = torch.abs(
                        torch.matmul(
                            torch.matmul(flat_state1.transpose(-2, -1), op_expanded),
                            flat_state2
                        )
                    ).squeeze(-1).squeeze(-1)
                    
                    # Reshape to batch and sequence dimensions
                    entanglement = entanglement.reshape(batch_size, -1)
                    # Average across sequence dimension for now
                    entanglement = entanglement.mean(dim=1)
                else:
                    # Simple case: single representation per modality
                    entanglement = torch.abs(
                        torch.matmul(
                            torch.matmul(state1, entanglement_op),
                            state2.transpose(-2, -1)
                        )
                    ).squeeze(-1)
                
                entanglement_measures[key] = entanglement
        
        return entanglement_measures
    
    def forward(
        self,
        features_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Apply quantum-inspired entanglement to features.
        
        Args:
            features_dict: Dictionary of features for each modality
                {modality_name: features_tensor}
                
        Returns:
            Tuple containing:
                - Dictionary of enhanced features
                - Dictionary of metadata including entanglement measures
        """
        # Encode features to quantum states
        quantum_states = {}
        for modality, features in features_dict.items():
            if modality in self.state_encoders:
                quantum_states[modality] = self.state_encoders[modality](features)
        
        # Apply entanglement operations
        entangled_states = self.apply_entanglement(quantum_states)
        
        # Measure entanglement
        entanglement_measures = self.measure_entanglement(entangled_states)
        
        # Project back to embedding space
        enhanced_features = {}
        for modality, state in entangled_states.items():
            if modality in self.measurement_projections:
                enhanced = self.measurement_projections[modality](state["real"])
                enhanced_features[modality] = enhanced + features_dict[modality]  # Residual connection
        
        # Prepare metadata
        metadata = {
            "entanglement_measures": entanglement_measures,
            "quantum_states": {
                modality: {k: v.detach() for k, v in state.items()}
                for modality, state in quantum_states.items()
            }
        }
        
        return enhanced_features, metadata


class QuantumEntanglementGate(nn.Module):
    """
    Implements a parameterized quantum gate for modality entanglement.
    
    This module provides parameterized entanglement operations inspired by
    quantum computing gates that can create and manipulate cross-modal
    entanglement.
    """
    
    def __init__(
        self,
        quantum_dim: int,
        gate_type: str = "cnot",
        num_params: int = 3
    ):
        """
        Initialize the quantum entanglement gate.
        
        Args:
            quantum_dim: Dimension of quantum state representation
            gate_type: Type of quantum gate to implement
                Options: "cnot", "hadamard", "phase", "custom"
            num_params: Number of learnable parameters for the gate
        """
        super().__init__()
        self.quantum_dim = quantum_dim
        self.gate_type = gate_type
        self.num_params = num_params
        
        # Parameters for the gate
        self.params = nn.Parameter(torch.randn(num_params))
        
        # Create gate matrices based on type
        if gate_type == "custom":
            # For custom gates, use fully learnable matrices
            self.gate_matrix_real = nn.Parameter(torch.randn(quantum_dim, quantum_dim))
            self.gate_matrix_imag = nn.Parameter(torch.randn(quantum_dim, quantum_dim))
            
            # Initialize as unitary-like
            nn.init.orthogonal_(self.gate_matrix_real)
            nn.init.orthogonal_(self.gate_matrix_imag)
        else:
            # For predefined gates, parameterize them
            self.register_buffer("gate_matrix_real", torch.eye(quantum_dim))
            self.register_buffer("gate_matrix_imag", torch.zeros(quantum_dim, quantum_dim))
            
            # Will be filled in the forward pass based on parameters
    
    def get_gate_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the current gate matrices based on parameters.
        
        Returns:
            Tuple of real and imaginary gate matrices
        """
        if self.gate_type == "custom":
            # Already have learnable matrices
            return self.gate_matrix_real, self.gate_matrix_imag
        
        # Get parameterized versions of standard gates
        if self.gate_type == "hadamard":
            # Hadamard-inspired gate with learnable weights
            theta = torch.sigmoid(self.params[0]) * math.pi  # [0, π]
            phi = self.params[1] * math.pi  # Arbitrary phase
            
            # Create rotation matrix
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)
            
            # Real and imaginary parts
            real_part = cos_theta * cos_phi
            imag_part = sin_theta * sin_phi
            
            # Update gate matrices (simplified version)
            gate_real = self.gate_matrix_real.clone()
            gate_imag = self.gate_matrix_imag.clone()
            
            # Apply to diagonal elements
            diag_size = min(self.quantum_dim, self.quantum_dim)
            diag_indices = torch.arange(diag_size)
            
            gate_real[diag_indices, diag_indices] = real_part
            gate_imag[diag_indices, diag_indices] = imag_part
            
            return gate_real, gate_imag
            
        elif self.gate_type == "phase":
            # Phase gate with learnable phase
            phase = self.params[0] * math.pi * 2  # [0, 2π]
            
            # Create phase matrix
            cos_phase = torch.cos(phase)
            sin_phase = torch.sin(phase)
            
            # Update gate matrices
            gate_real = torch.eye(self.quantum_dim, device=self.params.device) * cos_phase
            gate_imag = torch.eye(self.quantum_dim, device=self.params.device) * sin_phase
            
            return gate_real, gate_imag
            
        elif self.gate_type == "cnot":
            # CNOT-inspired gate (more complex to implement with continuous params)
            # Simplified implementation for demonstration
            gate_real = self.gate_matrix_real.clone()
            gate_imag = self.gate_matrix_imag.clone()
            
            # Parameterize the "flip" operation
            flip_strength = torch.sigmoid(self.params[0])  # [0, 1]
            
            # Apply to off-diagonal elements (simplified)
            half_dim = self.quantum_dim // 2
            for i in range(half_dim):
                j = i + half_dim
                gate_real[i, j] = flip_strength
                gate_real[j, i] = flip_strength
            
            return gate_real, gate_imag
        
        # Default fallback
        return self.gate_matrix_real, self.gate_matrix_imag
    
    def forward(
        self,
        state1_real: torch.Tensor,
        state1_imag: Optional[torch.Tensor] = None,
        state2_real: Optional[torch.Tensor] = None,
        state2_imag: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply the quantum gate to input states.
        
        Args:
            state1_real: Real part of first quantum state
            state1_imag: Imaginary part of first quantum state
            state2_real: Real part of second quantum state (for two-qubit gates)
            state2_imag: Imaginary part of second quantum state (for two-qubit gates)
            
        Returns:
            Tuple of real and imaginary parts of the output state
        """
        # Get current gate matrices
        gate_real, gate_imag = self.get_gate_matrices()
        
        # Apply gate to input state
        if state1_imag is None:
            # Real input state
            output_real = torch.matmul(state1_real, gate_real)
            output_imag = torch.matmul(state1_real, gate_imag)
        else:
            # Complex input state
            # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            output_real = torch.matmul(state1_real, gate_real) - torch.matmul(state1_imag, gate_imag)
            output_imag = torch.matmul(state1_real, gate_imag) + torch.matmul(state1_imag, gate_real)
        
        # For two-qubit gates, incorporate the second state
        if state2_real is not None:
            # Simplified implementation for demonstration
            # In a full implementation, you would use tensor products
            # and more sophisticated entanglement operations
            
            # Simple mixing of states
            mix_factor = torch.sigmoid(self.params[-1])  # [0, 1]
            
            if state2_imag is None:
                output_real = output_real * (1 - mix_factor) + state2_real * mix_factor
                output_imag = output_imag * (1 - mix_factor) if output_imag is not None else None
            else:
                output_real = output_real * (1 - mix_factor) + state2_real * mix_factor
                output_imag = output_imag * (1 - mix_factor) + state2_imag * mix_factor
        
        return output_real, output_imag


class ModalEntanglementModule(nn.Module):
    """
    Module for quantum-inspired modal entanglement.
    
    This module enables quantum-inspired entanglement between different modalities,
    allowing for deeper cross-modal integration and understanding.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        modalities: List[str] = ["vision", "text", "audio"],
        quantum_dim: int = 64,
        num_qubits: int = 6,
        num_entanglement_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize the modal entanglement module.
        
        Args:
            embedding_dim: Dimension of input embeddings
            modalities: List of supported modalities
            quantum_dim: Dimension of quantum state representation
            num_qubits: Number of virtual qubits to simulate
            num_entanglement_layers: Number of entanglement layers
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.modalities = modalities
        self.quantum_dim = quantum_dim
        
        # Primary entanglement layer
        self.entanglement_layer = ModalEntanglementLayer(
            embedding_dim=embedding_dim,
            quantum_dim=quantum_dim,
            num_qubits=num_qubits,
            num_unitary_layers=3,
            dropout=dropout
        )
        
        # Additional entanglement processing layers
        self.entanglement_processors = nn.ModuleList([
            ModalEntanglementLayer(
                embedding_dim=embedding_dim,
                quantum_dim=quantum_dim,
                num_qubits=num_qubits,
                num_unitary_layers=2,
                dropout=dropout
            ) for _ in range(num_entanglement_layers - 1)
        ]) if num_entanglement_layers > 1 else nn.ModuleList([])
        
        # Quantum gates for entanglement operations
        self.quantum_gates = nn.ModuleDict({
            f"{m1}_{m2}": QuantumEntanglementGate(
                quantum_dim=quantum_dim,
                gate_type="custom" if i % 3 == 0 else 
                           "hadamard" if i % 3 == 1 else "phase",
                num_params=3
            )
            for i, (m1, m2) in enumerate([
                (m1, m2) for m1 in modalities for m2 in modalities if m1 != m2
            ])
        })
        
        # Final integration layer
        self.final_integration = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(embedding_dim + quantum_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim)
            ) for modality in modalities
        })
    
    def forward(
        self,
        features_dict: Dict[str, torch.Tensor],
        attention_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Apply quantum-inspired modal entanglement.
        
        Args:
            features_dict: Dictionary of features for each modality
                {modality_name: features_tensor}
            attention_masks: Optional dictionary of attention masks
                {modality_name: attention_mask}
                
        Returns:
            Tuple containing:
                - Dictionary of enhanced features
                - Dictionary of metadata including entanglement measures
        """
        # Apply primary entanglement layer
        entangled_features, metadata = self.entanglement_layer(features_dict)
        
        # Apply additional entanglement processing layers
        current_features = entangled_features
        all_entanglement_measures = [metadata["entanglement_measures"]]
        
        for processor in self.entanglement_processors:
            processed_features, layer_metadata = processor(current_features)
            current_features = processed_features
            all_entanglement_measures.append(layer_metadata["entanglement_measures"])
        
        # Apply quantum gates between modality pairs
        quantum_gate_outputs = {}
        
        for modality in self.modalities:
            if modality not in current_features:
                continue
                
            quantum_gate_outputs[modality] = current_features[modality]
        
        for gate_key, gate in self.quantum_gates.items():
            mod1, mod2 = gate_key.split("_")
            
            if mod1 not in quantum_gate_outputs or mod2 not in quantum_gate_outputs:
                continue
            
            # Get quantum states from metadata
            if "quantum_states" in metadata and mod1 in metadata["quantum_states"]:
                state1_real = metadata["quantum_states"][mod1]["real"]
                state1_imag = metadata["quantum_states"][mod1].get("imag")
            else:
                # Fallback
                state1_real = quantum_gate_outputs[mod1]
                state1_imag = None
            
            if "quantum_states" in metadata and mod2 in metadata["quantum_states"]:
                state2_real = metadata["quantum_states"][mod2]["real"]
                state2_imag = metadata["quantum_states"][mod2].get("imag")
            else:
                # Fallback
                state2_real = quantum_gate_outputs[mod2]
                state2_imag = None
            
            # Apply quantum gate
            gate_out_real, gate_out_imag = gate(state1_real, state1_imag, state2_real, state2_imag)
            
            # Update states (simplified, in reality you'd handle this more carefully)
            quantum_gate_outputs[mod1] = gate_out_real
        
        # Final integration with original features
        integrated_features = {}
        
        for modality, features in features_dict.items():
            if modality not in self.final_integration:
                continue
            
            # Combine original features with quantum-enhanced features
            if modality in quantum_gate_outputs:
                quantum_features = quantum_gate_outputs[modality]
                
                # Handle sequence dimension
                if features.dim() > 2 and quantum_features.dim() > 2:
                    # Already have sequence dimensions
                    seq_len = min(features.shape[1], quantum_features.shape[1])
                    combined = torch.cat([
                        features[:, :seq_len, :], 
                        quantum_features[:, :seq_len, :]
                    ], dim=-1)
                else:
                    # Add sequence dimension if needed
                    if features.dim() == 2:
                        features = features.unsqueeze(1)
                    if quantum_features.dim() == 2:
                        quantum_features = quantum_features.unsqueeze(1)
                    
                    combined = torch.cat([features, quantum_features], dim=-1)
                
                # Apply final integration
                integrated = self.final_integration[modality](combined)
                
                # Add residual connection
                if integrated.shape == features.shape:
                    integrated = integrated + features
                
                integrated_features[modality] = integrated
            else:
                integrated_features[modality] = features
        
        # Combine metadata
        combined_metadata = {
            "entanglement_measures": all_entanglement_measures[-1],
            "all_entanglement_measures": all_entanglement_measures,
            "quantum_states": metadata.get("quantum_states", {})
        }
        
        return integrated_features, combined_metadata