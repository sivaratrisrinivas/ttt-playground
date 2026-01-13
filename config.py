"""Simple data structures for learn-doc"""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class DocumentStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    LEARNING = "learning"
    READY = "ready"
    ERROR = "error"


@dataclass
class DocumentChunk:
    """Single chunk of document text"""
    index: int
    text: str
    token_ids: List[int]
    token_count: int
    start_page: Optional[int] = None
    end_page: Optional[int] = None


@dataclass
class Document:
    """Uploaded document with extracted content"""
    id: str
    filename: str
    page_count: int
    total_tokens: int
    chunks: List[DocumentChunk]
    status: DocumentStatus = DocumentStatus.READY
    error_message: Optional[str] = None


@dataclass
class LearningConfig:
    """TTT hyperparameters"""
    inner_lr: float = 0.01
    chunk_size: int = 2048
    max_grad_norm: float = 1.0
    mask_ratio: float = 0.15  # Percentage of tokens to mask


@dataclass
class LearningMetrics:
    """Metrics from TTT learning pass"""
    initial_loss: float
    final_loss: float
    loss_history: List[float]
    chunks_processed: int
    tokens_processed: int
    learning_time_seconds: float
    weight_delta_norm: float


@dataclass
class Answer:
    """Model response"""
    text: str
    tokens_generated: int
    generation_time_seconds: float
