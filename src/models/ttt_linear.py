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
        
        # Store initial weights for reset (Step 3.4)
        # register_buffer keeps it on same device but not as a parameter
        self.register_buffer('_W_h_initial', self.W_h.data.clone())
        
        # Output projection: hidden_dim → output_dim
        self.W_out = nn.Linear(hidden_dim, output_dim, bias=False)
        
        # Activation
        self.activation = nn.SiLU()  # SwiGLU-style, common in modern LLMs
    
    def forward(self, x: torch.Tensor, learning: bool = False) -> torch.Tensor:
        """
        Forward pass with optional weight update.
        
        Args:
            x: Input tensor [batch, seq, dim]
            learning: If True, compute loss and update W_h
            
        Returns:
            Output tensor [batch, seq, dim]
        """
        # Step 1: Input projection
        hidden = self.W_in(x)  # [batch, seq, hidden_dim]
        
        # Step 2: Apply activation
        hidden = self.activation(hidden)
        
        # Step 3: Apply W_h transformation
        # hidden: [batch, seq, hidden_dim], W_h: [hidden_dim, input_dim]
        # Result: [batch, seq, input_dim]
        hidden = torch.matmul(hidden, self.W_h)
        
        # Step 4: Output projection
        output = self.W_out(hidden)  # [batch, seq, output_dim]
        
        # Learning mode: update W_h (implemented in Step 3.5)
        if learning:
            self._update_weights(x, hidden)
        
        return output
    
    def _update_weights(self, x: torch.Tensor, hidden: torch.Tensor) -> None:
        """
        Update W_h using self-supervised learning.
        
        Self-supervised objective: predict next token's hidden state from current.
        Loss = MSE(W_h @ h_t, h_{t+1})
        """
        # Placeholder - will be implemented in Step 3.5
        pass
