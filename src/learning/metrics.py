"""
Learning metrics utilities.

Tracks loss over LaCT chunks and produces LearningMetrics (Step 6.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MetricsTracker:
    """
    Minimal metrics tracker for Phase 6.

    Step 6.2 requires only loss_history recording.
    Step 6.3 will add get_metrics() -> LearningMetrics.
    """

    loss_history: List[float] = field(default_factory=list)

    def record_loss(self, loss: float) -> None:
        self.loss_history.append(float(loss))

