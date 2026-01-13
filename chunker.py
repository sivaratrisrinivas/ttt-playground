"""Token-based text chunking for documents"""
from typing import List
from transformers import PreTrainedTokenizer
from config import DocumentChunk


class DocumentChunker:
    """Split text into token-aligned chunks"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chunk_size: int = 2048
    ):
        """
        Initialize chunker.
        
        Args:
            tokenizer: HuggingFace tokenizer (e.g., Qwen2.5-1.5B tokenizer)
            chunk_size: Maximum tokens per chunk (default: 2048)
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
    
    def chunk(self, text: str) -> List[DocumentChunk]:
        """
        Split text into chunks of chunk_size tokens.
        
        Args:
            text: Raw document text
            
        Returns:
            List of DocumentChunk objects
        """
        if not text.strip():
            return []
        
        # Tokenize entire text once
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Split into chunks
        chunks = []
        for i in range(0, len(token_ids), self.chunk_size):
            chunk_token_ids = token_ids[i:i + self.chunk_size]
            
            # Decode chunk tokens back to text
            chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
            
            # Create DocumentChunk
            chunk = DocumentChunk(
                index=len(chunks),  # Sequential index
                text=chunk_text,
                token_ids=chunk_token_ids,
                token_count=len(chunk_token_ids)
            )
            chunks.append(chunk)
        
        return chunks
