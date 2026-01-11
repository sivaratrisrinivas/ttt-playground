# TTT Playground - Technical Specification

> **Purpose**: Portfolio demo for Adaption Labs showcasing Test-Time Training - a model that learns from documents at inference time.

---

## 1. Project Overview and Core Objectives

### 1.1 Problem Statement

Current LLMs are "frozen" after training. To use document knowledge, you must:
- Stuff document into context window (expensive, limited)
- Fine-tune the model (slow, requires infrastructure)

TTT solves this: **the model learns from the document during inference**, then answers questions without the document in context.

### 1.2 Strategic Alignment with Adaption Labs

| Adaption Labs Thesis | How TTT Addresses It |
|---------------------|---------------------|
| Scaling is inefficient | Same model, adaptive weights - no bigger model needed |
| "Hardware Lottery" constraints | LaCT chunking makes TTT viable on free-tier GPUs |
| Continuous adaptation | Model evolves at inference time, not frozen |

### 1.3 Core Objectives

1. **Implement TTT-Linear layers** - Replace MLP with learnable hidden state
2. **LaCT optimization** - Chunk-based updates for GPU efficiency
3. **Document learning pipeline** - PDF → chunks → weight updates
4. **Demo interface** - Side-by-side comparison proving TTT works
5. **Deployable artifact** - HF Spaces URL for Adaption Labs to try

### 1.4 Success Criteria

- TTT model correctly answers questions base model cannot
- Learning completes in <60s for 50-page PDF on T4
- Side-by-side comparison is self-evident
- Works on free-tier GPU (Colab T4 / HF Spaces T4)

---

## 2. Tech Stack and Libraries

### 2.1 Core Framework

| Component | Library | Version | Rationale |
|-----------|---------|---------|-----------|
| Deep Learning | PyTorch | 2.1+ | Best ecosystem, torch.compile support |
| Model Base | transformers | 4.36+ | TinyLlama integration |
| Tokenization | transformers | 4.36+ | AutoTokenizer |
| Compilation | torch.compile | - | Fuse TTT forward/backward |

### 2.2 Document Processing

| Component | Library | Version | Rationale |
|-----------|---------|---------|-----------|
| PDF Extraction | PyMuPDF (fitz) | 1.23+ | Fast, reliable, no Java deps |
| Text Chunking | tiktoken | 0.5+ | Accurate token counting |

### 2.3 Interface & Deployment

| Component | Library | Version | Rationale |
|-----------|---------|---------|-----------|
| UI Framework | Gradio | 4.0+ | Simple, HF Spaces native |
| Hosting | HF Spaces | - | Free T4, persistent URL |
| Dev Environment | Google Colab | - | Free T4, fast iteration |

### 2.4 Utilities

| Component | Library | Rationale |
|-----------|---------|-----------|
| Progress bars | tqdm | Learning visualization |
| Logging | loguru | Clean debug output |
| Config | pydantic | Type-safe settings |

### 2.5 Full requirements.txt

```
torch>=2.1.0
transformers>=4.36.0
accelerate>=0.25.0
tiktoken>=0.5.0
PyMuPDF>=1.23.0
gradio>=4.0.0
tqdm>=4.66.0
loguru>=0.7.0
pydantic>=2.5.0
```

---

## 3. Detailed Data Models (Schema)

### 3.1 Document Processing Models

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class DocumentStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    LEARNING = "learning"
    READY = "ready"
    ERROR = "error"

class DocumentChunk(BaseModel):
    """Single chunk of document text"""
    index: int                          # Chunk position (0-indexed)
    text: str                           # Raw text content
    token_ids: List[int]                # Tokenized form
    token_count: int                    # Number of tokens
    start_page: Optional[int] = None    # Source page start
    end_page: Optional[int] = None      # Source page end

class Document(BaseModel):
    """Uploaded document with extracted content"""
    id: str                             # UUID
    filename: str                       # Original filename
    page_count: int                     # Total pages
    total_tokens: int                   # Total token count
    chunks: List[DocumentChunk]         # Chunked content
    status: DocumentStatus              # Processing status
    error_message: Optional[str] = None # Error if status=ERROR

class DocumentConstraints(BaseModel):
    """Validation constraints"""
    max_pages: int = 100
    max_file_size_mb: int = 50
    max_tokens: int = 100_000
    min_tokens: int = 500
    chunk_size: int = 2048              # LaCT chunk size
```

### 3.2 TTT Learning Models

```python
class LearningConfig(BaseModel):
    """TTT hyperparameters"""
    inner_lr: float = 0.01              # Inner loop learning rate
    chunk_size: int = 2048              # LaCT chunk size
    max_grad_norm: float = 1.0          # Gradient clipping
    loss_type: str = "next_token"       # "next_token" or "masked"

class LearningMetrics(BaseModel):
    """Metrics from TTT learning pass"""
    initial_loss: float                 # Loss before learning
    final_loss: float                   # Loss after learning
    loss_history: List[float]           # Per-chunk losses
    chunks_processed: int               # Number of chunks
    tokens_processed: int               # Total tokens
    learning_time_seconds: float        # Wall clock time
    weight_delta_norm: float            # L2 norm of weight change

class LearningState(BaseModel):
    """State of TTT learning for a session"""
    document_id: str                    # Associated document
    is_learned: bool = False            # Learning complete?
    metrics: Optional[LearningMetrics] = None
    # Note: actual W_h weights stored in model, not serialized here
```

### 3.3 Inference Models

```python
class Question(BaseModel):
    """User question"""
    text: str                           # Question text
    max_tokens: int = 256               # Max response length
    temperature: float = 0.7            # Sampling temperature

class Answer(BaseModel):
    """Model response"""
    text: str                           # Generated answer
    tokens_generated: int               # Response length
    generation_time_seconds: float      # Latency

class ComparisonResult(BaseModel):
    """Side-by-side comparison output"""
    question: Question
    ttt_answer: Answer                  # TTT model (learned)
    base_answer: Answer                 # Base model (no learning)
    document_id: str                    # Source document
```

### 3.4 Session Models

```python
class Session(BaseModel):
    """User session state"""
    id: str                             # Session UUID
    document: Optional[Document] = None # Uploaded doc
    learning_state: Optional[LearningState] = None
    questions_asked: List[Question] = []
    created_at: float                   # Timestamp
```

---

## 4. Architecture / Directory Structure

### 4.1 Project Layout

```
ttt-playground/
├── spec.md                    # This file - source of truth
├── README.md                  # Project overview, demo link, Adaption pitch
├── requirements.txt           # Python dependencies
│
├── notebooks/                 # Colab development notebooks
│   ├── 01_ttt_layer.ipynb    # TTT-Linear implementation + tests
│   ├── 02_model_mod.ipynb    # TinyLlama modification
│   ├── 03_lact.ipynb         # LaCT chunking implementation
│   ├── 04_pdf_pipeline.ipynb # Document processing
│   └── 05_integration.ipynb  # Full pipeline test
│
├── src/                       # Production code (copied to HF Space)
│   ├── __init__.py
│   │
│   ├── models/               # TTT model implementation
│   │   ├── __init__.py
│   │   ├── ttt_linear.py    # TTT-Linear layer class
│   │   ├── ttt_model.py     # Modified TinyLlama with TTT layers
│   │   └── lact.py          # Large Chunk TTT logic
│   │
│   ├── document/             # Document processing
│   │   ├── __init__.py
│   │   ├── pdf_parser.py    # PyMuPDF extraction
│   │   ├── chunker.py       # Token-based chunking
│   │   └── validator.py     # Constraint validation
│   │
│   ├── learning/             # TTT learning loop
│   │   ├── __init__.py
│   │   ├── trainer.py       # Document learning orchestration
│   │   └── metrics.py       # Loss tracking, visualization
│   │
│   ├── inference/            # Generation
│   │   ├── __init__.py
│   │   └── generator.py     # Q&A generation with TTT weights
│   │
│   └── config.py             # Pydantic settings
│
├── app.py                     # Gradio interface (HF Spaces entry)
│
└── tests/                     # Unit tests
    ├── test_ttt_linear.py
    ├── test_chunker.py
    └── test_learning.py
```

### 4.2 Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `models/ttt_linear.py` | TTT-Linear layer with inner-loop gradient update |
| `models/ttt_model.py` | TinyLlama with MLP→TTT-Linear replacement |
| `models/lact.py` | Chunked gradient accumulation, single update |
| `document/pdf_parser.py` | PDF→text extraction via PyMuPDF |
| `document/chunker.py` | Split text into 2048-token chunks |
| `learning/trainer.py` | Orchestrate: chunks → forward → loss → update |
| `learning/metrics.py` | Track loss curve, weight delta, timing |
| `inference/generator.py` | Generate answers using learned W_h |
| `app.py` | Gradio UI, session management, comparison |

### 4.3 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gradio UI (app.py)                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ PDF Upload   │  │ Learning     │  │ Q&A Interface        │  │
│  │ + Preview    │  │ Progress     │  │ + Comparison         │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
└─────────┼─────────────────┼─────────────────────┼───────────────┘
          │                 │                     │
          ▼                 ▼                     ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│ document/        │ │ learning/        │ │ inference/           │
│ ├─ pdf_parser    │ │ ├─ trainer       │ │ └─ generator         │
│ ├─ chunker       │ │ └─ metrics       │ │                      │
│ └─ validator     │ │                  │ │                      │
└────────┬─────────┘ └────────┬─────────┘ └──────────┬───────────┘
         │                    │                      │
         │                    ▼                      │
         │           ┌──────────────────┐            │
         └──────────►│ models/          │◄───────────┘
                     │ ├─ ttt_linear    │
                     │ ├─ ttt_model     │
                     │ └─ lact          │
                     └──────────────────┘
```

---

## 5. API / Interface Definitions

### 5.1 TTT-Linear Layer Interface

```python
class TTTLinear(nn.Module):
    """
    TTT-Linear layer that updates hidden weights during forward pass.
    Replaces standard MLP in transformer blocks.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        inner_lr: float = 0.01
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension (W_h shape)
            output_dim: Output dimension
            inner_lr: Learning rate for inner loop updates
        """
        ...
    
    def forward(
        self,
        x: torch.Tensor,
        learning: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with optional weight update.
        
        Args:
            x: Input tensor [batch, seq, dim]
            learning: If True, compute loss and update W_h
            
        Returns:
            Output tensor [batch, seq, dim]
        """
        ...
    
    def reset_weights(self) -> None:
        """Reset W_h to initial state (for new document)."""
        ...
    
    def get_weight_delta(self) -> float:
        """Return L2 norm of weight change from initial."""
        ...
```

### 5.2 TTT Model Interface

```python
class TTTModel:
    """
    TinyLlama with TTT-Linear layers replacing MLPs.
    """
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ttt_layers: List[int] = None,  # Which layers get TTT (default: all)
        device: str = "cuda"
    ) -> "TTTModel":
        """Load base model and replace specified MLP layers with TTT-Linear."""
        ...
    
    def learn_from_chunks(
        self,
        chunks: List[DocumentChunk],
        config: LearningConfig
    ) -> LearningMetrics:
        """
        Perform TTT learning on document chunks.
        
        Args:
            chunks: List of tokenized document chunks
            config: Learning hyperparameters
            
        Returns:
            Metrics from learning pass
        """
        ...
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """Generate response using current (learned) weights."""
        ...
    
    def reset_learning(self) -> None:
        """Reset all TTT-Linear layers to initial weights."""
        ...
    
    def clear_context(self) -> None:
        """Clear KV cache (for demo: prove no context used)."""
        ...
```

### 5.3 Document Processor Interface

```python
class PDFParser:
    """Extract text from PDF files."""
    
    def parse(self, file_bytes: bytes) -> Tuple[str, int]:
        """
        Extract text from PDF.
        
        Args:
            file_bytes: Raw PDF file content
            
        Returns:
            (extracted_text, page_count)
            
        Raises:
            PDFExtractionError: If extraction fails
        """
        ...

class DocumentChunker:
    """Split text into token-aligned chunks."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chunk_size: int = 2048
    ):
        ...
    
    def chunk(self, text: str) -> List[DocumentChunk]:
        """
        Split text into chunks of chunk_size tokens.
        
        Args:
            text: Raw document text
            
        Returns:
            List of DocumentChunk objects
        """
        ...

class DocumentValidator:
    """Validate documents against constraints."""
    
    def validate(
        self,
        file_bytes: bytes,
        constraints: DocumentConstraints
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if document meets constraints.
        
        Returns:
            (is_valid, error_message_if_invalid)
        """
        ...
```

### 5.4 Learning Trainer Interface

```python
class TTTTrainer:
    """Orchestrates TTT learning from documents."""
    
    def __init__(
        self,
        model: TTTModel,
        config: LearningConfig
    ):
        ...
    
    def train_on_document(
        self,
        document: Document,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ) -> LearningMetrics:
        """
        Run TTT learning on document.
        
        Args:
            document: Parsed and chunked document
            progress_callback: Called with (chunk_idx, total_chunks, loss)
            
        Returns:
            Learning metrics
        """
        ...
```

### 5.5 Gradio App Interface

```python
# app.py - Gradio interface functions

def upload_document(file: gr.File) -> Tuple[str, str, gr.update]:
    """
    Handle PDF upload.
    
    Returns:
        (status_message, document_preview, learning_button_update)
    """
    ...

def start_learning(session_id: str) -> Generator[Tuple[str, dict], None, None]:
    """
    Stream learning progress.
    
    Yields:
        (progress_text, metrics_dict)
    """
    ...

def ask_question(
    session_id: str,
    question: str
) -> Tuple[str, str]:
    """
    Get TTT and base model answers.
    
    Returns:
        (ttt_answer, base_answer)
    """
    ...

def reset_session() -> Tuple[gr.update, ...]:
    """Clear session state for new document."""
    ...
```

---

## 6. Testing Strategy

### 6.1 Test Categories

| Category | Purpose | Location |
|----------|---------|----------|
| Unit Tests | Individual component correctness | `tests/` |
| Integration Tests | Pipeline end-to-end | `notebooks/05_integration.ipynb` |
| Manual Tests | Demo UX verification | Checklist below |

### 6.2 Unit Tests

#### 6.2.1 TTT-Linear Layer Tests (`tests/test_ttt_linear.py`)

```python
def test_forward_shape():
    """Output shape matches input shape."""
    layer = TTTLinear(768, 2048, 768)
    x = torch.randn(1, 128, 768)
    y = layer(x, learning=False)
    assert y.shape == x.shape

def test_learning_updates_weights():
    """Weights change when learning=True."""
    layer = TTTLinear(768, 2048, 768)
    w_before = layer.W_h.clone()
    x = torch.randn(1, 128, 768)
    layer(x, learning=True)
    assert not torch.allclose(layer.W_h, w_before)

def test_no_learning_preserves_weights():
    """Weights unchanged when learning=False."""
    layer = TTTLinear(768, 2048, 768)
    w_before = layer.W_h.clone()
    x = torch.randn(1, 128, 768)
    layer(x, learning=False)
    assert torch.allclose(layer.W_h, w_before)

def test_reset_restores_initial():
    """reset_weights() restores original state."""
    layer = TTTLinear(768, 2048, 768)
    w_initial = layer.W_h.clone()
    layer(torch.randn(1, 128, 768), learning=True)
    layer.reset_weights()
    assert torch.allclose(layer.W_h, w_initial)

def test_gradient_flow():
    """Gradients flow through layer correctly."""
    layer = TTTLinear(768, 2048, 768)
    x = torch.randn(1, 128, 768, requires_grad=True)
    y = layer(x, learning=False)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
```

#### 6.2.2 Document Chunker Tests (`tests/test_chunker.py`)

```python
def test_chunk_size_respected():
    """Each chunk has ≤ chunk_size tokens."""
    chunker = DocumentChunker(tokenizer, chunk_size=2048)
    chunks = chunker.chunk(long_text)
    for chunk in chunks:
        assert chunk.token_count <= 2048

def test_no_content_lost():
    """All tokens from original text present in chunks."""
    chunker = DocumentChunker(tokenizer, chunk_size=2048)
    chunks = chunker.chunk(text)
    reconstructed_tokens = []
    for chunk in chunks:
        reconstructed_tokens.extend(chunk.token_ids)
    original_tokens = tokenizer.encode(text)
    assert reconstructed_tokens == original_tokens

def test_short_document_single_chunk():
    """Short document produces single chunk."""
    chunker = DocumentChunker(tokenizer, chunk_size=2048)
    chunks = chunker.chunk("Hello world.")
    assert len(chunks) == 1

def test_empty_input_handling():
    """Empty string returns empty list."""
    chunker = DocumentChunker(tokenizer, chunk_size=2048)
    chunks = chunker.chunk("")
    assert len(chunks) == 0
```

#### 6.2.3 Learning Tests (`tests/test_learning.py`)

```python
def test_loss_decreases():
    """Loss decreases over chunks (model is learning)."""
    trainer = TTTTrainer(model, config)
    metrics = trainer.train_on_document(test_document)
    assert metrics.final_loss < metrics.initial_loss

def test_learning_time_reasonable():
    """Learning completes within expected time."""
    trainer = TTTTrainer(model, config)
    metrics = trainer.train_on_document(test_document)
    # 30 chunks * 2s per chunk = 60s max
    assert metrics.learning_time_seconds < 120

def test_weight_delta_nonzero():
    """Weights actually changed after learning."""
    trainer = TTTTrainer(model, config)
    metrics = trainer.train_on_document(test_document)
    assert metrics.weight_delta_norm > 0
```

### 6.3 Integration Tests (Notebook)

Run in `notebooks/05_integration.ipynb`:

1. **Full Pipeline Test**
   - Upload test PDF → Extract → Chunk → Learn → Clear context → Q&A
   - Verify TTT answer contains document-specific info
   - Verify base answer does NOT contain document-specific info

2. **Memory Test**
   - Process 50-page PDF on T4
   - Monitor VRAM usage (should stay under 14GB)
   - No OOM errors

3. **Latency Test**
   - Measure learning time for various document sizes
   - Target: <2s per 2048-token chunk

### 6.4 Manual Test Checklist (Demo QA)

```
[ ] PDF upload works for various PDFs
[ ] Progress bar updates during learning
[ ] "Context Cleared" indicator visible before Q&A
[ ] TTT answer is relevant to document
[ ] Base answer is generic/wrong
[ ] Side-by-side comparison is visually clear
[ ] Reset button clears state correctly
[ ] Error messages shown for invalid PDFs
[ ] Demo works on HF Spaces (not just local)
```

### 6.5 Test Data

| Document | Pages | Tokens | Purpose |
|----------|-------|--------|---------|
| `test_short.pdf` | 3 | ~1500 | Single chunk test |
| `test_medium.pdf` | 20 | ~10K | Standard test case |
| `test_long.pdf` | 80 | ~50K | Stress test |
| `test_corrupt.pdf` | - | - | Error handling test |

### 6.6 CI/CD (Optional Future)

```yaml
# .github/workflows/test.yml (if we add GitHub Actions)
- Run pytest on CPU (TTT layer logic)
- Run notebook tests on Colab via papermill (GPU tests)
- Deploy to HF Spaces on main branch push
```

---

## 7. Development Environment

### 7.1 Google Colab Setup

```python
# Cell 1: Install dependencies
!pip install torch transformers accelerate tiktoken PyMuPDF gradio tqdm loguru pydantic

# Cell 2: Mount Drive (for saving checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Clone repo (once we push to GitHub)
!git clone https://github.com/YOUR_USERNAME/ttt-playground.git
%cd ttt-playground

# Cell 4: Verify GPU
!nvidia-smi
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 7.2 HF Spaces Deployment

```yaml
# README.md header for HF Spaces
---
title: TTT Playground
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---
```

---

## 8. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TTT doesn't learn useful info | Medium | High | Start with extractive Q&A, simple docs |
| T4 OOM | Medium | Medium | Gradient checkpointing, fewer TTT layers |
| Learning too slow | Low | Medium | torch.compile, reduce layers |
| Base model too weak | Low | Medium | Test Phi-2 (2.7B) as backup |
| Colab disconnects | High | Low | Save checkpoints to Drive |

---

## 9. Timeline

| Week | Milestone |
|------|-----------|
| Week 1 | TTT-Linear layer + TinyLlama integration + basic learning loop |
| Week 2 | PDF pipeline + LaCT + Gradio UI + HF Spaces deploy |
| Buffer | Polish, edge cases, README pitch |

---

## 10. References

- Sun et al. 2024 - "Learning to (Learn at Test Time)" - TTT paper
- Sara Hooker 2020 - "The Hardware Lottery" ([arXiv:2009.06489](https://arxiv.org/abs/2009.06489))
- Adaption Labs - [https://adaptionlabs.ai/](https://adaptionlabs.ai/)
- TinyLlama - [https://huggingface.co/TinyLlama](https://huggingface.co/TinyLlama)
