"""
TTT-Linear layer: A SwiGLU MLP layer where the Gate (W_h) is trainable at test time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TTTLinear(nn.Module):
    """
    SwiGLU layer where W_h (Gate) acts as the 'Fast Weight' memory.
    
    Architecture:
        gate = W_h(x)      <-- Learned during inference
        up   = W_up(x)     <-- Frozen
        hidden = SiLU(gate) * up
        output = W_out(hidden) <-- Frozen
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        # W_h: The "Hidden State" weights we update
        self.W_h = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # W_up & W_out: The static knowledge weights
        self.W_up = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, output_dim, bias=False)
        
        # Buffer to store initial state for reset
        self.register_buffer('_W_h_initial', torch.empty(hidden_dim, input_dim))
        self._initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard SwiGLU forward pass."""
        # 1. Capture original dtype (likely float16)
        original_dtype = x.dtype
        
        # 2. Cast input to layer's dtype (float32) for stable training
        target_dtype = self.W_h.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        # Save initial weights on first run if not done
        if not self._initialized:
            with torch.no_grad():
                self._W_h_initial = self.W_h.weight.data.clone()
                self._initialized = True

        # 3. Perform calculations in Float32
        gate = self.W_h(x)
        up = self.W_up(x)
        hidden = F.silu(gate) * up
        output = self.W_out(hidden)
        
        # 4. Cast output back to original dtype (float16) to satisfy next layers
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
            
        return output

    def reset_weights(self) -> None:
        """Restore W_h to pre-trained state."""
        if self._initialized:
            with torch.no_grad():
                self.W_h.weight.data.copy_(self._W_h_initial)

    def get_weight_delta(self) -> float:
        """Measure how much we've learned."""
        if not self._initialized: return 0.0
        with torch.no_grad():
            delta = self.W_h.weight - self._W_h_initial
            return torch.norm(delta).item()