"""Tests for config.py dataclasses"""
import pytest
from config import (
    DocumentStatus,
    DocumentChunk,
    Document,
    LearningConfig,
    LearningMetrics,
    Answer
)


class TestDocumentChunk:
    """Test DocumentChunk dataclass"""

    def test_valid_chunk(self):
        """Test creating a valid DocumentChunk"""
        chunk = DocumentChunk(
            index=0,
            text="Sample text content",
            token_ids=[1, 2, 3, 4, 5],
            token_count=5,
            start_page=1,
            end_page=1
        )
        assert chunk.index == 0
        assert chunk.text == "Sample text content"
        assert chunk.token_count == 5
        assert chunk.start_page == 1
        assert chunk.end_page == 1

    def test_chunk_optional_pages(self):
        """Test chunk without page info"""
        chunk = DocumentChunk(
            index=1,
            text="Another chunk",
            token_ids=[10, 20, 30],
            token_count=3
        )
        assert chunk.start_page is None
        assert chunk.end_page is None


class TestDocument:
    """Test Document dataclass"""

    def test_valid_document(self):
        """Test creating a valid Document"""
        chunks = [
            DocumentChunk(
                index=0,
                text="Chunk 1",
                token_ids=[1, 2],
                token_count=2
            ),
            DocumentChunk(
                index=1,
                text="Chunk 2",
                token_ids=[3, 4],
                token_count=2
            )
        ]
        doc = Document(
            id="test-uuid-123",
            filename="test.pdf",
            page_count=5,
            total_tokens=4,
            chunks=chunks,
            status=DocumentStatus.READY
        )
        assert doc.id == "test-uuid-123"
        assert doc.filename == "test.pdf"
        assert len(doc.chunks) == 2
        assert doc.status == DocumentStatus.READY
        assert doc.error_message is None

    def test_document_with_error(self):
        """Test Document with error status"""
        doc = Document(
            id="error-doc",
            filename="corrupt.pdf",
            page_count=0,
            total_tokens=0,
            chunks=[],
            status=DocumentStatus.ERROR,
            error_message="PDF extraction failed"
        )
        assert doc.status == DocumentStatus.ERROR
        assert doc.error_message == "PDF extraction failed"


class TestDocumentStatus:
    """Test DocumentStatus enum"""

    def test_all_statuses(self):
        """Test all enum values exist"""
        assert DocumentStatus.PENDING == "pending"
        assert DocumentStatus.EXTRACTING == "extracting"
        assert DocumentStatus.CHUNKING == "chunking"
        assert DocumentStatus.LEARNING == "learning"
        assert DocumentStatus.READY == "ready"
        assert DocumentStatus.ERROR == "error"


class TestLearningConfig:
    """Test LearningConfig dataclass"""

    def test_default_values(self):
        """Test default values"""
        config = LearningConfig()
        assert config.inner_lr == 0.01
        assert config.chunk_size == 2048
        assert config.max_grad_norm == 1.0
        assert config.mask_ratio == 0.15

    def test_custom_config(self):
        """Test custom LearningConfig values"""
        config = LearningConfig(
            inner_lr=0.005,
            chunk_size=1024,
            max_grad_norm=0.5,
            mask_ratio=0.2
        )
        assert config.inner_lr == 0.005
        assert config.chunk_size == 1024
        assert config.max_grad_norm == 0.5
        assert config.mask_ratio == 0.2


class TestLearningMetrics:
    """Test LearningMetrics dataclass"""

    def test_valid_metrics(self):
        """Test creating valid LearningMetrics"""
        metrics = LearningMetrics(
            initial_loss=2.5,
            final_loss=1.8,
            loss_history=[2.5, 2.2, 2.0, 1.8],
            chunks_processed=4,
            tokens_processed=8192,
            learning_time_seconds=12.5,
            weight_delta_norm=0.15
        )
        assert metrics.initial_loss == 2.5
        assert metrics.final_loss == 1.8
        assert len(metrics.loss_history) == 4
        assert metrics.chunks_processed == 4
        assert metrics.tokens_processed == 8192
        assert metrics.learning_time_seconds == 12.5
        assert metrics.weight_delta_norm == 0.15

    def test_metrics_loss_decrease(self):
        """Test that metrics can track loss decrease"""
        metrics = LearningMetrics(
            initial_loss=3.0,
            final_loss=1.5,
            loss_history=[3.0, 2.5, 2.0, 1.5],
            chunks_processed=4,
            tokens_processed=8192,
            learning_time_seconds=10.0,
            weight_delta_norm=0.2
        )
        assert metrics.final_loss < metrics.initial_loss
        assert metrics.loss_history[0] == metrics.initial_loss
        assert metrics.loss_history[-1] == metrics.final_loss


class TestAnswer:
    """Test Answer dataclass"""

    def test_answer_creation(self):
        """Test creating an Answer"""
        answer = Answer(
            text="TTT is test-time training.",
            tokens_generated=5,
            generation_time_seconds=0.5
        )
        assert answer.text == "TTT is test-time training."
        assert answer.tokens_generated == 5
        assert answer.generation_time_seconds == 0.5
