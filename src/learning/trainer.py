"""
Training orchestration for TTT learning.

Phase 6 builds a thin trainer that wires together:
- TTTModel (TinyLlama with TTTLinear layers)
- LaCTUpdater (chunk-wise gradient accumulation + update)
- MetricsTracker (loss history + LearningMetrics)
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Callable, Optional

from src.config import Document, LearningConfig
from src.learning.metrics import MetricsTracker

if TYPE_CHECKING:
    from src.models.ttt_model import TTTModel


@dataclass
class TTTTrainer:
    """
    Orchestrates TTT learning from documents.

    Step 6.4 only requires that __init__ works.
    """

    model: "TTTModel"
    config: LearningConfig

    def __post_init__(self) -> None:
        # Import here to avoid importing torch on module import
        # (some local dev envs may not have CUDA libs available).
        from src.models.lact import LaCTUpdater

        self.metrics = MetricsTracker()
        self.updater = LaCTUpdater(
            self.model,
            inner_lr=self.config.inner_lr,
            max_grad_norm=self.config.max_grad_norm,
        )

    def train_on_document(
        self,
        document: Document,
        *,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> "LearningMetrics":
        """
        Orchestrate learning over all chunks of a document.

        Step 6.5 target:
        - iterate chunks
        - compute loss per chunk + apply update (via LaCTUpdater)
        - return LearningMetrics with final_loss < initial_loss (usually)
        """
        from src.config import LearningMetrics  # avoid heavy imports at module import time

        # Reset per-run state
        self.metrics = MetricsTracker()
        self.updater.reset()
        self.model.reset_learning()

        t0 = perf_counter()

        tokens_processed = 0
        total = len(document.chunks)
        for chunk_idx, chunk in enumerate(document.chunks):
            tokens_processed += int(chunk.token_count)
            loss = self.updater.process_chunk(chunk.token_ids)
            self.updater.apply_update()
            self.metrics.record_loss(loss)
            if progress_callback is not None:
                progress_callback(chunk_idx, total, float(loss))

        t1 = perf_counter()

        # Weight delta across all TTT layers
        weight_delta = float(self.model.get_total_weight_delta())

        return self.metrics.get_metrics(
            tokens_processed=tokens_processed,
            learning_time_seconds=(t1 - t0),
            weight_delta_norm=weight_delta,
        )

