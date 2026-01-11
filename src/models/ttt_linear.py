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
        x → (W_h) → hidden → activation → W_out → output
        
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

        # Learnable hidden weights - this is what TTT updates
        # Shape: (hidden_dim, input_dim) per spec test requirement.
        # Forward uses x @ W_h.T to produce hidden_dim features.
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
        # Step 1: project into hidden_dim using W_h
        # x: [batch, seq, input_dim], W_h.T: [input_dim, hidden_dim]
        hidden = torch.matmul(x, self.W_h.t())  # [batch, seq, hidden_dim]

        # Step 2: activation in hidden space
        hidden = self.activation(hidden)

        # Step 3: output projection
        output = self.W_out(hidden)  # [batch, seq, output_dim]
        
        # Learning mode: update W_h (implemented in Step 3.5)
        if learning:
            self._update_weights(x, hidden)
        
        return output
    
    def _update_weights(self, x: torch.Tensor, hidden: torch.Tensor) -> None:
        """
        Update W_h using self-supervised learning.
        
        Self-supervised objective: predict next token's hidden state from current.
        We use a simple reconstruction loss on the input sequence.
        """
        # Use input x for self-supervised learning
        # Predict: given x[t], predict x[t+1] via W_h
        # This is a simplified TTT objective
        
        seq_len = x.shape[1]
        if seq_len < 2:
            return  # Need at least 2 tokens for next-token prediction
        
        # Current and next tokens
        x_curr = x[:, :-1, :]  # [batch, seq-1, input_dim]
        x_next = x[:, 1:, :]   # [batch, seq-1, input_dim]
        
        # Predict next token: project through W_h and back
        # h = x_curr @ W_h.T  -> [batch, seq-1, hidden_dim]
        # pred = h @ W_h      -> [batch, seq-1, input_dim]
        h_curr = torch.matmul(x_curr, self.W_h.t())
        pred = torch.matmul(h_curr, self.W_h)
        
        # MSE loss
        loss = torch.mean((pred - x_next) ** 2)
        
        # Compute gradient w.r.t. W_h
        grad = torch.autograd.grad(loss, self.W_h, retain_graph=False)[0]
        
        # Update W_h with gradient descent (inner loop)
        with torch.no_grad():
            self.W_h.data -= self.inner_lr * grad
    
    def reset_weights(self) -> None:
        """Reset W_h to initial state (for new document)."""
        with torch.no_grad():
            self.W_h.data.copy_(self._W_h_initial)
    
    def get_weight_delta(self) -> float:
        """Return L2 norm of weight change from initial."""
        with torch.no_grad():
            delta = self.W_h - self._W_h_initial
            return torch.norm(delta).item()
