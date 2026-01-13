"""
TTTModel - Qwen2.5-0.5B with TTT-Linear layers.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from ttt_linear import TTTLinear

class TTTModel(nn.Module):
    def __init__(self, base_model: nn.Module, tokenizer, ttt_layers: List[TTTLinear]):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.ttt_layers = ttt_layers
        self.device = base_model.device

    @classmethod
    def from_pretrained(cls, model_name="Qwen/Qwen2.5-0.5B-Instruct", ttt_layer_indices=None, device="cuda"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load Base Model in Float16 (standard)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            dtype=torch.float16, 
            device_map=device
        )
        
        ttt_layers = cls._replace_mlp_layers(base_model, ttt_layer_indices)
        return cls(base_model, tokenizer, ttt_layers)
    
    @classmethod
    def _replace_mlp_layers(cls, model, layer_indices):
        ttt_layers = []
        transformer_layers = model.model.layers
        if layer_indices is None: layer_indices = list(range(len(transformer_layers)))
        
        print(f"Installing TTT Layers in: {layer_indices}")
        
        for idx in layer_indices:
            layer = transformer_layers[idx]
            mlp = layer.mlp
            
            # Create TTT Layer
            ttt_layer = TTTLinear(
                input_dim=mlp.gate_proj.in_features,
                hidden_dim=mlp.gate_proj.out_features,
                output_dim=mlp.gate_proj.in_features
            )
            
            # Copy Weights and CAST TO FLOAT32
            # This is the key fix for "Loss: nan"
            with torch.no_grad():
                ttt_layer.W_h.weight.data = mlp.gate_proj.weight.data.float()
                ttt_layer.W_up.weight.data = mlp.up_proj.weight.data.float()
                ttt_layer.W_out.weight.data = mlp.down_proj.weight.data.float()
            
            # Move to device (GPU) but keep as Float32
            ttt_layer.to(device=mlp.gate_proj.weight.device, dtype=torch.float32)
            
            # Replace
            layer.mlp = ttt_layer
            ttt_layers.append(ttt_layer)
            
        return ttt_layers

    def enable_ttt_learning(self):
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False
        for layer in self.ttt_layers:
            layer.W_h.weight.requires_grad = True
            
    def disable_ttt_learning(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def reset_learning(self):
        for layer in self.ttt_layers:
            layer.reset_weights()
            
    def clear_context(self):
        if hasattr(self.model, '_past_key_values'):
            self.model._past_key_values = None

    def forward(self, input_ids, labels=None):
        return self.model(input_ids=input_ids, labels=labels)

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7, **kwargs) -> str:
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Config
        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.2
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)