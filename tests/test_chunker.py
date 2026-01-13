"""Tests for DocumentChunker - Step 2.4, 2.5, 2.6"""
import pytest
from transformers import AutoTokenizer
from chunker import DocumentChunker
from config import DocumentChunk as ConfigDocumentChunk


class TestDocumentChunker:
    """Test DocumentChunker class"""
    
    @pytest.fixture
    def tokenizer(self):
        """Create Qwen2.5-1.5B tokenizer for testing"""
        # Use a small tokenizer for testing (Qwen2.5-1.5B if available, else GPT-2)
        try:
            return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        except:
            # Fallback to GPT-2 for local testing
            return AutoTokenizer.from_pretrained("gpt2")
    
    def test_init_with_tokenizer(self, tokenizer):
        """Test DocumentChunker initialization - Step 2.4"""
        chunker = DocumentChunker(tokenizer, chunk_size=2048)
        assert chunker.chunk_size == 2048
        assert chunker.tokenizer is tokenizer
    
    def test_init_default_chunk_size(self, tokenizer):
        """Test default chunk_size is 2048"""
        chunker = DocumentChunker(tokenizer)
        assert chunker.chunk_size == 2048
    
    def test_chunk_small_text(self, tokenizer):
        """Test chunking small text (single chunk)"""
        chunker = DocumentChunker(tokenizer, chunk_size=2048)
        text = "This is a short text that should fit in one chunk."
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].text == text
        assert chunks[0].token_count <= 2048
    
    def test_chunk_large_text_multiple_chunks(self, tokenizer):
        """Test chunking large text produces multiple chunks - Step 2.5"""
        chunker = DocumentChunker(tokenizer, chunk_size=2048)
        
        # Create text that will be ~5000 tokens
        # Each word is ~1 token, so we need ~5000 words
        words = ["word"] * 5000
        large_text = " ".join(words)
        
        chunks = chunker.chunk(large_text)
        
        # Should get at least 2 chunks (5000 tokens / 2048 = ~2.4 chunks)
        assert len(chunks) >= 2
        
        # Each chunk should be <= chunk_size
        for chunk in chunks:
            assert chunk.token_count <= 2048
            assert chunk.index >= 0
            assert isinstance(chunk, ConfigDocumentChunk)
    
    def test_chunk_preserves_all_tokens(self, tokenizer):
        """Test that chunking preserves all tokens - Step 2.6"""
        chunker = DocumentChunker(tokenizer, chunk_size=2048)
        
        # Create text
        text = "This is a test document with multiple sentences. " * 100
        
        # Get original tokenization
        original_token_ids = tokenizer.encode(text)
        
        # Chunk the text
        chunks = chunker.chunk(text)
        
        # Reconstruct token IDs from chunks
        reconstructed_token_ids = []
        for chunk in chunks:
            reconstructed_token_ids.extend(chunk.token_ids)
        
        # Verify all tokens preserved
        assert reconstructed_token_ids == original_token_ids
    
    def test_chunk_empty_text(self, tokenizer):
        """Test chunking empty text returns empty list"""
        chunker = DocumentChunker(tokenizer, chunk_size=2048)
        chunks = chunker.chunk("")
        assert len(chunks) == 0
    
    def test_chunk_chunks_are_sequential(self, tokenizer):
        """Test that chunk indices are sequential"""
        chunker = DocumentChunker(tokenizer, chunk_size=100)  # Small chunk size
        
        text = "word " * 500  # ~500 tokens
        chunks = chunker.chunk(text)
        
        # Check indices are sequential 0, 1, 2, ...
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
