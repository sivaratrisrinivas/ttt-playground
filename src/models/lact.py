"""
LaCT - Large Chunk TTT.

Efficient TTT learning by processing documents in chunks:
1. Forward pass on chunk (2048 tokens)
2. Accumulate gradients
3. Single weight update per chunk

This makes TTT viable on free-tier GPUs (T4) by reducing memory overhead.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from .ttt_model import TTTModel
from .ttt_linear import TTTLinear


class LaCTUpdater:
    """
    Large Chunk TTT - efficient document learning.
    
    Usage:
        updater = LaCTUpdater(model)
        for chunk in chunks:
            loss = updater.process_chunk(chunk.token_ids)
            updater.apply_update()
        # Or simply:
        metrics = updater.process_document(chunks)
    """
    
    def __init__(
        self,
        model: TTTModel,
        inner_lr: float = 0.01,
        max_grad_norm: float = 1.0,
        debug: bool = False,
    ):
        """
        Args:
            model: TTTModel with TTT-Linear layers
            inner_lr: Learning rate for weight updates
            max_grad_norm: Gradient clipping threshold
        """
        self.model = model
        self.inner_lr = inner_lr
        self.max_grad_norm = max_grad_norm
        self.debug = debug
        
        # Accumulated gradients for each TTT layer
        self._accumulated_grads: List[Optional[torch.Tensor]] = [
            None for _ in model.ttt_layers
        ]
        self._loss_history: List[float] = []
    
    def process_chunk(self, token_ids: List[int]) -> float:
        """
        Forward pass on chunk, compute self-supervised loss.
        
        Args:
            token_ids: List of token IDs (e.g., 2048 tokens)
            
        Returns:
            Loss value for this chunk
        """
        # Convert to tensor
        device = self.model.model.device
        tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
        
        # Need at least 2 tokens for next-token prediction
        if tokens.shape[1] < 2:
            return 0.0
        
        # Input and target for next-token prediction
        input_ids = tokens[:, :-1]
        target_ids = tokens[:, 1:]
        
        # Zero existing gradients (entire model to avoid accumulation/NaNs)
        self.model.model.zero_grad(set_to_none=True)
        
        # Forward pass through model (get logits)
        outputs = self.model.model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits  # [batch, seq, vocab]
        
        # Compute cross-entropy loss in FP32 for numerical stability
        logits_fp32 = logits.float()
        loss = nn.functional.cross_entropy(
            logits_fp32.view(-1, logits_fp32.size(-1)),
            target_ids.view(-1),
            reduction='mean'
        )
        
        loss_value = loss.item()
        self._loss_history.append(loss_value)
        
        # Backward pass to compute gradients
        loss.backward()
        
        # Step 5.3: Accumulate gradients for each TTT layer's W_h
        for i, ttt_layer in enumerate(self.model.ttt_layers):
            if ttt_layer.W_h.grad is not None:
                grad = ttt_layer.W_h.grad.detach()
                # Convert to FP32 for stable accumulation
                grad = grad.float()

                # Guard: if grad has NaN/Inf, skip accumulating for this layer
                if not torch.isfinite(grad).all():
                    if self.debug:
                        bad = (~torch.isfinite(grad)).sum().item()
                        print(f"[LaCT] non-finite grad in layer {i}: bad={bad}")
                    continue
                
                if self._accumulated_grads[i] is None:
                    self._accumulated_grads[i] = grad.clone()
                else:
                    self._accumulated_grads[i] += grad
        
        return loss_value
    
    def apply_update(self) -> None:
        """
        Apply accumulated gradients to update W_h in all TTT layers.
        """
        with torch.no_grad():
            for i, ttt_layer in enumerate(self.model.ttt_layers):
                grad = self._accumulated_grads[i]
                if grad is not None:
                    # Guard: skip non-finite grads entirely
                    if not torch.isfinite(grad).all():
                        if self.debug:
                            bad = (~torch.isfinite(grad)).sum().item()
                            print(f"[LaCT] skipping update for layer {i}: non-finite grad bad={bad}")
                        self._accumulated_grads[i] = None
                        continue

                    # Clip gradient
                    grad_norm = torch.norm(grad)
                    if (not torch.isfinite(grad_norm)) or grad_norm <= 0:
                        if self.debug:
                            print(f"[LaCT] skipping update for layer {i}: grad_norm={grad_norm}")
                        self._accumulated_grads[i] = None
                        continue

                    if grad_norm > self.max_grad_norm:
                        grad = grad * (self.max_grad_norm / grad_norm)
                    
                    # Convert grad to same dtype as W_h
                    grad = grad.to(dtype=ttt_layer.W_h.dtype)
                    
                    # Update W_h
                    ttt_layer.W_h.data -= self.inner_lr * grad

                    # Guard: if W_h became non-finite, revert this layer to initial weights
                    if not torch.isfinite(ttt_layer.W_h).all():
                        if self.debug:
                            bad = (~torch.isfinite(ttt_layer.W_h)).sum().item()
                            print(f"[LaCT] W_h became non-finite in layer {i} (bad={bad}); resetting layer")
                        ttt_layer.reset_weights()
                    
                    # Clear accumulated gradient
                    self._accumulated_grads[i] = None
    
    def process_document(
        self, 
        chunks: List,  # List[DocumentChunk]
        progress_callback=None
    ) -> dict:
        """
        Process entire document chunk by chunk.
        
        Args:
            chunks: List of DocumentChunk objects
            progress_callback: Optional callback(chunk_idx, total, loss)
            
        Returns:
            Dict with metrics: initial_loss, final_loss, total_chunks
        """
        if not chunks:
            return {"initial_loss": 0, "final_loss": 0, "total_chunks": 0}
        
        initial_loss = None
        final_loss = None
        
        for idx, chunk in enumerate(chunks):
            # Process chunk
            loss = self.process_chunk(chunk.token_ids)
            
            if initial_loss is None:
                initial_loss = loss
            final_loss = loss
            
            # Apply update after each chunk
            self.apply_update()
            
            # Progress callback
            if progress_callback is not None:
                progress_callback(idx, len(chunks), loss)
        
        return {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "total_chunks": len(chunks)
        }
    
    def get_loss_history(self) -> List[float]:
        """Return loss values from all processed chunks."""
        return self._loss_history.copy()
    
    def reset(self) -> None:
        """Reset accumulated gradients and loss history."""
        self._accumulated_grads = [None for _ in self.model.ttt_layers]
        self._loss_history = []
