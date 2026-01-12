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
        # Step 4.2: Load base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use FP16 for memory efficiency
            device_map=device
        )
        
        # Ensure pad token exists (TinyLlama uses eos as pad)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Step 4.4-4.6: Replace MLP layers with TTTLinear
        ttt_layers = cls._replace_mlp_layers(base_model, ttt_layer_indices)
        
        return cls(base_model, tokenizer, ttt_layers)
    
    @classmethod
    def _replace_mlp_layers(
        cls, 
        model: nn.Module, 
        layer_indices: Optional[List[int]] = None
    ) -> List[TTTLinear]:
        """
        Replace MLP layers with TTTLinear.
        
        Args:
            model: The transformer model
            layer_indices: Which layers to replace (None = all)
            
        Returns:
            List of TTTLinear layers that were installed
        """
        ttt_layers = []
        
        # Access transformer layers (TinyLlama structure: model.model.layers)
        transformer_layers = model.model.layers
        num_layers = len(transformer_layers)
        
        # Default: replace all layers
        if layer_indices is None:
            layer_indices = list(range(num_layers))
        
        for idx in layer_indices:
            if idx >= num_layers:
                continue
                
            layer = transformer_layers[idx]
            mlp = layer.mlp
            
            # Get dimensions from existing MLP
            # TinyLlama MLP: gate_proj (hidden->intermediate), up_proj, down_proj (intermediate->hidden)
            hidden_size = mlp.gate_proj.in_features      # 2048 for TinyLlama
            intermediate_size = mlp.gate_proj.out_features  # 5632 for TinyLlama
            
            # Create TTTLinear to replace MLP
            # Input: hidden_size, Hidden: intermediate_size, Output: hidden_size
            ttt_layer = TTTLinear(
                input_dim=hidden_size,
                hidden_dim=intermediate_size,
                output_dim=hidden_size,
                inner_lr=0.01
            )
            
            # Move to same device and dtype as original
            device = mlp.gate_proj.weight.device
            dtype = mlp.gate_proj.weight.dtype
            ttt_layer = ttt_layer.to(device=device, dtype=dtype)
            
            # Replace MLP with TTTLinear
            layer.mlp = ttt_layer
            ttt_layers.append(ttt_layer)
        
        return ttt_layers
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = None,
        **kwargs
    ) -> str:
        """Generate text from prompt using current (possibly learned) weights."""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Determine sampling strategy
        if do_sample is None:
            do_sample = temperature > 0
        
        # Build generation config
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample,
            "repetition_penalty": 1.2,  # Prevent repetition loops
            "no_repeat_ngram_size": 3,  # No repeating 3-grams
        }
        
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
            gen_kwargs["top_k"] = 50
        
        # Override with any explicit kwargs
        gen_kwargs.update(kwargs)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Decode and return
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def reset_learning(self) -> None:
        """Reset all TTT-Linear layers to initial weights."""
        for layer in self.ttt_layers:
            layer.reset_weights()
    
    def clear_context(self) -> None:
        """Clear KV cache (for demo: prove no context used)."""
        # The KV cache is managed internally by HuggingFace's generate()
        # Each generate() call with new inputs starts fresh
        # For explicit clearing, we can reset the past_key_values
        if hasattr(self.model, '_past_key_values'):
            self.model._past_key_values = None
    
    def get_total_weight_delta(self) -> float:
        """Sum of weight deltas across all TTT layers."""
        return sum(layer.get_weight_delta() for layer in self.ttt_layers)
