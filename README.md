# Learn-Doc

> A Test-Time Training demo: the model *learns* from a document, then answers questions **without the document in context**.

---

## What This Does

1. **Upload PDF** - Document is parsed and chunked
2. **Start Learning** - Model updates weights using masked token prediction
3. **Context Cleared** - KV cache is wiped (proof: no cheating)
4. **Ask Questions** - Model answers from learned weights only
5. **Compare** - See TTT answer vs Base model (which can't answer)

---

## Why This Matters

Current LLMs are frozen after training. To use document knowledge:
- Stuff it in context (expensive, limited to ~128K tokens)
- Fine-tune (slow, needs infrastructure)

**TTT solves this**: the model learns from documents *during inference*.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the CLI

```bash
# Learn from a PDF
python cli/cli.py learn docs/sample.txt

# Ask a question
python cli/cli.py ask "What is the main topic of this document?"

# Compare TTT vs Base model
python cli/cli.py compare "What is the capital of France?"

# Check status
python cli/cli.py status

# Reset for new document
python cli/cli.py reset
```

### 3. Use Your Own PDF

```bash
# Convert your PDF to text first (or use any text file)
python cli/cli.py learn path/to/your/document.pdf
```

---

## Commands

| Command | Description |
|---------|-------------|
| `learn <file>` | Learn from a PDF or text file |
| `ask <question>` | Ask a question (after learning) |
| `compare <question>` | Compare TTT vs Base model answers |
| `reset` | Clear session for new document |
| `status` | Show current status |
| `help` | Show help message |

---

## Architecture

```
Document → Extract → Chunk → TTT Learning (Masked Token) → Clear Context → Q&A
                                          ↓
                            W_h ← W_h - η∇L_masked(x, W_h)
```

- **TTT-Linear layers**: Replace MLPs, update hidden weights during inference
- **Masked token prediction**: Self-supervised task that forces document understanding
- **Base model**: Qwen2.5-1.5B-Instruct

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Framework | PyTorch |
| Model | Qwen2.5-1.5B-Instruct |
| PDF | PyMuPDF |
| Interface | CLI (Command Line) |

---

## Test Document

A sample test document is included at `docs/sample.txt`. Convert it to PDF or use it directly to test the system.

---

## References

- Sun et al. 2024 - "Learning to (Learn at Test Time)"
- Sara Hooker 2020 - ["The Hardware Lottery"](https://arxiv.org/abs/2009.06489)

