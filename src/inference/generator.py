"""
Generator for TTT model inference.

Wraps TTTModel to provide Q&A generation with Answer dataclass output,
and side-by-side comparison between TTT (learned) and base (reset) models.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from src.config import Answer

if TYPE_CHECKING:
    from src.models.ttt_model import TTTModel
    from transformers import PreTrainedTokenizer


class Generator:
    """
    Generate answers using TTT model.
    
    Provides generate() for single answers and compare() for TTT vs base.
    """
    
    def __init__(self, model: "TTTModel", tokenizer: "PreTrainedTokenizer") -> None:
        """
        Args:
            model: TTTModel instance (with learned weights)
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Answer:
        """
        Generate answer from prompt using current model weights.
        
        Args:
            prompt: Input text/question
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Answer with generated text, token count, and timing
        """
        t0 = perf_counter()
        
        # Generate using TTTModel's generate method
        full_output = self.model.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        
        t1 = perf_counter()
        
        # Extract just the generated part (remove prompt)
        generated_text = full_output[len(prompt):].strip()
        
        # Count tokens in generated portion
        tokens_generated = len(self.tokenizer.encode(generated_text, add_special_tokens=False))
        
        return Answer(
            text=generated_text,
            tokens_generated=tokens_generated,
            generation_time_seconds=t1 - t0,
        )
