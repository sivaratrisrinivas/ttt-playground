"""
TTT Trainer - Implements the Inner Loop (Learning).
"""

import time
import torch
from torch.optim import AdamW
from config import Document, LearningConfig, LearningMetrics

class TTTTrainer:
    def __init__(self, model, tokenizer, config: LearningConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def train_on_document(self, document: Document, progress_callback=None):
        self.model.enable_ttt_learning()
        
        params = [l.W_h.weight for l in self.model.ttt_layers]
        optimizer = AdamW(params, lr=self.config.inner_lr)
        
        total_loss = 0
        steps = 0
        
        for idx, chunk in enumerate(document.chunks):
            input_ids = torch.tensor([chunk.token_ids], device=self.model.device)
            
            # 1. Forward Pass
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            task_loss = outputs.loss
            
            # 2. Regularization (The Anchor)
            reg_loss = 0.0
            for layer in self.model.ttt_layers:
                diff = layer.W_h.weight - layer._W_h_initial
                reg_loss += torch.sum(diff ** 2)
            
            # FIXED: Reduced strength to 0.01 to balance the scales
            # Reg is ~40, Task is ~0.4.
            # 40 * 0.01 = 0.4. Now they are equal!
            strength = 0.01
            loss = task_loss + (strength * reg_loss)
            
            # LOGGING: Verify balance
            if idx == 0 and steps % 5 == 0: 
                print(f" [Step {steps}] Task: {task_loss.item():.4f} | Reg: {reg_loss.item():.4f} (x{strength}) = {loss.item():.4f}")

            if torch.isnan(loss):
                if progress_callback: progress_callback(idx, len(document.chunks), float('nan'))
                continue

            # 3. Update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            
            optimizer.step()
            optimizer.zero_grad()
            
            current_loss = task_loss.item()
            total_loss += current_loss
            steps += 1
            
            if progress_callback:
                progress_callback(idx, len(document.chunks), current_loss)
        
        self.model.disable_ttt_learning()
        
        return LearningMetrics(
            initial_loss=0.0,
            final_loss=total_loss / steps if steps > 0 else 0.0,
            loss_history=[],
            chunks_processed=steps,
            tokens_processed=0,
            learning_time_seconds=0,
            weight_delta_norm=0
        )