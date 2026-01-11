"""
TTT-Linear layer implementation.

This layer replaces standard MLP in transformer blocks. The hidden weights W_h
are updated during the forward pass when learning=True, enabling test-time training.
"""

import torch
import torch.nn as nn


class TTTLinear(nn.Module):
    """
    TTT-Linear layer that updates hidden weights during forward pass.
    Replaces standard MLP in transformer blocks.
    
    Architecture:
        x → W_in → hidden → activation → W_h → W_out → output
        
    During learning (learning=True):
        - Compute self-supervised loss (next-token prediction on hidden states)
        - Update W_h with gradient descent using inner_lr
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        inner_lr: float = 0.01
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., 768 for TinyLlama)
            hidden_dim: Hidden state dimension - W_h shape is (hidden_dim, input_dim)
            output_dim: Output dimension (typically same as input_dim)
            inner_lr: Learning rate for inner loop weight updates
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.inner_lr = inner_lr
        
        # Input projection: input_dim → hidden_dim
        self.W_in = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # Learnable hidden weights - this is what TTT updates
        # Shape: (hidden_dim, input_dim) per spec test requirement
        self.W_h = nn.Parameter(torch.empty(hidden_dim, input_dim))
        nn.init.kaiming_uniform_(self.W_h)
        
        # Output projection: hidden_dim → output_dim
        self.W_out = nn.Linear(hidden_dim, output_dim, bias=False)
        
        # Activation
        self.activation = nn.SiLU()  # SwiGLU-style, common in modern LLMs
