"""
Generator for TTT model inference.

Wraps TTTModel to provide Q&A generation with Answer dataclass output,
and side-by-side comparison between TTT (learned) and base (reset) models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.ttt_model import TTTModel
    from transformers import PreTrainedTokenizer


class Generator:
    """
    Generate answers using TTT model.
    
    Step 7.2: __init__ only - stores model and tokenizer references.
    """
    
    def __init__(self, model: "TTTModel", tokenizer: "PreTrainedTokenizer") -> None:
        """
        Args:
            model: TTTModel instance (with learned weights)
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer
