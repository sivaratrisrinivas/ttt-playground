"""
TTT Playground - Gradio UI

Upload a PDF → model learns it → ask questions (TTT vs Base comparison).
"""
import os
import time
import inspect
from typing import Any, Optional, Tuple

import gradio as gr
import torch

from src.config import Document, DocumentStatus, LearningConfig
from src.document.pdf_parser import PDFExtractionError, PDFParser
from src.document.chunker import DocumentChunker
from src.inference.generator import Generator
from src.learning.trainer import TTTTrainer
from src.models.ttt_model import TTTModel

_MODEL: Optional[TTTModel] = None


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model() -> TTTModel:
    """Lazily load model (cached globally)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = TTTModel.from_pretrained(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=_device()
        )
    return _MODEL


def _format_metrics(metrics) -> str:
    """Format LearningMetrics as markdown."""
    return "\n".join(
        [
            "### Learning metrics",
            f"- **initial_loss**: `{metrics.initial_loss:.4f}`",
            f"- **final_loss**: `{metrics.final_loss:.4f}`",
            f"- **chunks_processed**: `{metrics.chunks_processed}`",
            f"- **tokens_processed**: `{metrics.tokens_processed}`",
            f"- **learning_time_seconds**: `{metrics.learning_time_seconds:.2f}`",
            f"- **weight_delta_norm**: `{metrics.weight_delta_norm:.4f}`",
        ]
    )


def _safe_err(msg: str) -> Tuple[str, str]:
    return f"### Error\n\n{msg}", ""


def start_learning(
    pdf_file: Optional[str],
    chunk_size: int,
    inner_lr: float,
    progress=gr.Progress(track_tqdm=False),
) -> Tuple[str, str, bool, dict]:
    """
    Process PDF and run TTT learning.

    Returns:
        status_md, metrics_md, context_cleared, session_state
    """
    if not pdf_file:
        status, metrics = _safe_err("No PDF provided.")
        return status, metrics, False, {}

    model = get_model()

    # Parse PDF
    parser = PDFParser()
    try:
        pdf_bytes = open(pdf_file, "rb").read()
    except Exception as e:
        status, metrics = _safe_err(f"Failed to read file: {e}")
        return status, metrics, False, {}

    try:
        text, page_count = parser.parse(pdf_bytes)
    except PDFExtractionError as e:
        status, metrics = _safe_err(f"PDF parse failed: {e}")
        return status, metrics, False, {}

    if not text.strip():
        status, metrics = _safe_err("PDF extracted empty text.")
        return status, metrics, False, {}

    # Chunk
    chunker = DocumentChunker(model.tokenizer, chunk_size=chunk_size)
    chunks = chunker.chunk(text)

    if not chunks:
        status, metrics = _safe_err("Chunker produced 0 chunks.")
        return status, metrics, False, {}

    doc = Document(
        id=os.path.basename(pdf_file),
        filename=os.path.basename(pdf_file),
        page_count=page_count,
        total_tokens=sum(c.token_count for c in chunks),
        chunks=chunks,
        status=DocumentStatus.READY,
    )

    # Train
    model.reset_learning()
    trainer = TTTTrainer(model=model, config=LearningConfig(inner_lr=inner_lr))

    t0 = time.time()

    def cb(idx: int, total_chunks: int, loss: float) -> None:
        p = (idx + 1) / max(total_chunks, 1)
        progress(p, desc=f"Learning chunk {idx+1}/{total_chunks} (loss={loss:.4f})")

    progress(0, desc="Starting learning…")
    metrics_obj = trainer.train_on_document(doc, progress_callback=cb)
    dt = time.time() - t0

    # Clear context explicitly after learning
    model.clear_context()

    status_md = "\n".join(
        [
            "### Status",
            f"- **file**: `{os.path.basename(pdf_file)}`",
            f"- **pages**: `{page_count}`",
            f"- **chunks**: `{len(chunks)}`",
            f"- **tokens**: `{doc.total_tokens}`",
            f"- **device**: `{_device()}`",
            f"- **elapsed**: `{dt:.2f}s`",
            "",
            "✅ Learning completed and context cleared.",
        ]
    )
    metrics_md = _format_metrics(metrics_obj)

    session_state: dict[str, Any] = {
        "has_doc": True,
        "filename": os.path.basename(pdf_file),
        "page_count": page_count,
        "chunk_size": chunk_size,
        "total_chunks": len(chunks),
        "total_tokens": doc.total_tokens,
    }
    return status_md, metrics_md, True, session_state


def answer_question(
    question: str,
    max_tokens: int,
    temperature: float,
    session_state: dict,
) -> Tuple[str, str]:
    """Generate answers from TTT model and Base model."""
    if not session_state or not session_state.get("has_doc"):
        return "", "No learned document in session. Upload + Start Learning first."

    if not question.strip():
        return "", "Empty question."

    model = get_model()
    gen = Generator(model=model, tokenizer=model.tokenizer)
    try:
        ttt_ans, base_ans = gen.compare(
            question, max_tokens=max_tokens, temperature=temperature
        )
    except Exception as e:
        return "", f"Generation failed: {e}"

    return ttt_ans.text, base_ans.text


def reset_session() -> Tuple[str, str, bool, dict, str, str]:
    """Reset all state for new document."""
    model = get_model()
    model.reset_learning()
    model.clear_context()
    return (
        "### Status\n\nSession reset. Upload a PDF to start again.",
        "",
        False,
        {},
        "",
        "",
    )


def build_ui() -> gr.Blocks:
    """Build the Gradio UI."""
    with gr.Blocks(title="TTT Playground") as demo:
        gr.Markdown("# TTT Playground")
        gr.Markdown("Upload a PDF → learn → ask questions (TTT vs Base).")

        session_state = gr.State({})

        with gr.Row():
            pdf_file = gr.File(label="PDF", file_types=[".pdf"], type="filepath")
            chunk_size = gr.Slider(
                label="Chunk size (tokens)",
                minimum=256,
                maximum=4096,
                step=128,
                value=2048,
            )

        with gr.Row():
            inner_lr = gr.Slider(
                label="Inner LR",
                minimum=0.001,
                maximum=0.05,
                step=0.001,
                value=0.01,
            )
            start_btn = gr.Button("Start Learning", variant="primary")
            reset_btn = gr.Button("Reset", variant="secondary")

        status_md = gr.Markdown("### Status\n\nIdle.")
        metrics_md = gr.Markdown("")
        context_cleared = gr.Checkbox(
            label="Context Cleared",
            value=False,
            interactive=False,
        )

        gr.Markdown("---\n## Q&A")
        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="Ask something…")
            submit_btn = gr.Button("Submit", variant="primary")

        with gr.Row():
            ttt_out = gr.Textbox(label="TTT Answer", lines=10)
            base_out = gr.Textbox(label="Base Answer", lines=10)

        with gr.Accordion("Generation settings", open=False):
            max_tokens = gr.Slider(
                label="Max new tokens", minimum=8, maximum=512, step=8, value=64
            )
            temperature = gr.Slider(
                label="Temperature", minimum=0.0, maximum=1.5, step=0.1, value=0.0
            )

        start_btn.click(
            fn=start_learning,
            inputs=[pdf_file, chunk_size, inner_lr],
            outputs=[status_md, metrics_md, context_cleared, session_state],
        )

        submit_btn.click(
            fn=answer_question,
            inputs=[question, max_tokens, temperature, session_state],
            outputs=[ttt_out, base_out],
        )

        reset_btn.click(
            fn=reset_session,
            inputs=[],
            outputs=[status_md, metrics_md, context_cleared, session_state, ttt_out, base_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()

    # Filter launch kwargs by signature (Gradio version compatibility)
    sig = inspect.signature(demo.launch)
    allowed = set(sig.parameters.keys())
    requested = {
        "server_name": os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        "server_port": int(os.getenv("PORT", "7860")),
        "show_api": False,
    }
    filtered = {k: v for k, v in requested.items() if k in allowed}

    demo.launch(**filtered)
