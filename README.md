# TTT Playground

> A Test-Time Training demo: upload a PDF, the model *learns* it, then answers questions **without the document in context**.

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/YOUR_USERNAME/ttt-playground)

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

## Demo

1. Upload a PDF
2. Watch the model *learn* (loss decreases)
3. Context is **cleared**
4. Ask questions → model answers from learned weights
5. Compare: TTT model (learned) vs Base model (no doc)

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
| Hosting | HF Spaces (T4) |

---

## Local Development

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/ttt-playground.git
cd ttt-playground

# Install
pip install -r requirements.txt

# Run (needs GPU)
python app.py
```

Or use Google Colab: [Open Notebook](https://colab.research.google.com/)

---

## References

- Sun et al. 2024 - "Learning to (Learn at Test Time)"
- Sara Hooker 2020 - ["The Hardware Lottery"](https://arxiv.org/abs/2009.06489)
- [Adaption Labs](https://adaptionlabs.ai/)

---

## License

MIT
