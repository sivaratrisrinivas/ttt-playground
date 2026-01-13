"""
Generator for TTT model inference.
"""

from __future__ import annotations
from time import perf_counter
from config import Answer

class Generator:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
    
    def generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.4) -> Answer:
        t0 = perf_counter()
        
        # SYSTEM PROMPT: Essential for 0.5B models to separate contexts
        full_prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant. You have learned a document via Test-Time Training.\n"
            "1. If the user asks about the document (Whales, Crystals, 2145), use the learned information.\n"
            "2. If the user asks general questions (People, Places, History), use your general knowledge.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        full_output = self.model.generate(
            full_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Clean output
        generated_text = full_output
        if "assistant\n" in generated_text:
            generated_text = generated_text.split("assistant\n")[-1]
            
        generated_text = generated_text.replace("<|im_end|>", "").strip()
        
        t1 = perf_counter()
        
        return Answer(
            text=generated_text,
            tokens_generated=len(self.tokenizer.encode(generated_text)),
            generation_time_seconds=t1 - t0,
        )
        
    def compare(self, prompt: str, max_tokens: int = 150, temperature: float = 0.4):
        pass