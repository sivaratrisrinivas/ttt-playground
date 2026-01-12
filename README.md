# TTT Playground

> A Test-Time Training demo: upload a PDF, the model *learns* it, then answers questions **without the document in context**.

[![Open Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sivaratrisrinivas/ttt-playground/blob/main/notebooks/06_gradio_demo.ipynb)

---

## Try It Now

**Click the badge above** → Run all cells → Get a public URL → Upload PDF → Demo!

That's it. Free T4 GPU, no setup required.

---

## What This Does

1. **Upload PDF** - Document is parsed and chunked
2. **Start Learning** - Model updates weights on each chunk (watch loss decrease!)
3. **Context Cleared** - KV cache is wiped (proof: no cheating)
4. **Ask Questions** - Model answers from learned weights only
5. **Compare** - See TTT answer vs Base model (which can't answer)

---

## Why This Matters

Current LLMs are frozen after training. To use document knowledge:
- Stuff it in context (expensive, limited to ~128K tokens)
- Fine-tune (slow, needs infrastructure)

**TTT solves this**: the model learns from documents *during inference*.

This directly addresses [Adaption Labs](https://adaptionlabs.ai/)' core thesis:
- **Anti-scaling**: Same model, adaptive weights
- **Hardware Lottery**: Works on free-tier GPUs via LaCT chunking
- **Continuous Adaptation**: Model evolves at inference time

---

## Recommended Demo PDFs

For best results, use **text-heavy PDFs** (not scanned images):

| PDF | Why It's Good | Link |
|-----|--------------|------|
| **The Hardware Lottery** | Short, factual, relevant to Adaption Labs | [Download](https://arxiv.org/pdf/2009.06489.pdf) |
| **Attention Is All You Need** | Famous paper, lots of facts | [Download](https://arxiv.org/pdf/1706.03762.pdf) |

**Example questions:**
- "Who wrote this paper?"
- "What is the main argument?"
- "What year was this published?"

---

## Architecture

```
PDF → Extract → Chunk (2048 tok) → TTT Learning → Clear Context → Q&A
                                        ↓
                              W_h ← W_h - η∇L(x, W_h)
```

- **TTT-Linear layers**: Replace MLPs, update hidden weights during inference
- **LaCT**: Large Chunk TTT - batch gradient updates for GPU efficiency
- **Base model**: TinyLlama-1.1B-Chat

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Framework | PyTorch |
| Model | TinyLlama-1.1B-Chat |
| PDF | PyMuPDF |
| UI | Gradio |
| Hosting | Google Colab (T4) |

---

## References

- Sun et al. 2024 - "Learning to (Learn at Test Time)"
- Sara Hooker 2020 - ["The Hardware Lottery"](https://arxiv.org/abs/2009.06489)
- [Adaption Labs](https://adaptionlabs.ai/)

---

## License

MIT
