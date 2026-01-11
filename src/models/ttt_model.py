"""
TTTModel - TinyLlama with TTT-Linear layers replacing MLPs.

This module wraps a pretrained TinyLlama model and replaces its MLP layers
with TTTLinear layers that can learn from documents at inference time.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .ttt_linear import TTTLinear


class TTTModel(nn.Module):
    """
    TinyLlama with TTT-Linear layers replacing MLPs.
    
    Usage:
        model = TTTModel.from_pretrained()
        model.learn_from_chunks(chunks)  # Updates W_h in all TTT layers
        model.clear_context()            # Clears KV cache
        output = model.generate(prompt)  # Generate with learned weights
        model.reset_learning()           # Reset W_h for new document
    """
    
    def __init__(self, base_model: nn.Module, tokenizer, ttt_layers: List[TTTLinear]):
        """
        Args:
            base_model: The underlying transformer model (TinyLlama)
            tokenizer: The tokenizer for the model
            ttt_layers: List of TTTLinear layers that replaced MLPs
        """
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.ttt_layers = ttt_layers
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ttt_layer_indices: Optional[List[int]] = None,
        device: str = "cuda"
    ) -> "TTTModel":
        """
        Load base model and replace specified MLP layers with TTT-Linear.
        
        Args:
            model_name: HuggingFace model identifier
            ttt_layer_indices: Which transformer layers get TTT (default: all)
            device: Device to load model on
            
        Returns:
            TTTModel instance with TTT-Linear layers
        """
        # Step 4.2: Load base model
        # Step 4.4-4.6: Replace MLP layers
        # Placeholder - will be implemented in subsequent steps
        raise NotImplementedError("Step 4.2")
    
    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs) -> str:
        """Generate text from prompt using current (possibly learned) weights."""
        # Placeholder - will be implemented
        raise NotImplementedError("generate not implemented")
    
    def reset_learning(self) -> None:
        """Reset all TTT-Linear layers to initial weights."""
        # Step 4.7
        raise NotImplementedError("Step 4.7")
    
    def clear_context(self) -> None:
        """Clear KV cache (for demo: prove no context used)."""
        # Step 4.8
        raise NotImplementedError("Step 4.8")
    
    def get_total_weight_delta(self) -> float:
        """Sum of weight deltas across all TTT layers."""
        return sum(layer.get_weight_delta() for layer in self.ttt_layers)
