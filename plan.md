# TTT Playground - Implementation Plan

> Atomic, testable steps. Execute sequentially. Check off as you complete.

---

## Phase 0: Setup

- [ ] **Step 0.1**: Create GitHub repo `ttt-playground`, push existing files (spec.md, README.md, requirements.txt)
  - **Test**: `git remote -v` shows GitHub URL

- [ ] **Step 0.2**: Create Colab notebook, clone repo, verify GPU access
  - **Test**: `torch.cuda.is_available()` returns `True`, `nvidia-smi` shows T4

- [ ] **Step 0.3**: Install dependencies in Colab
  - **Test**: `import torch, transformers, fitz, gradio` all succeed

---

## Phase 1: Data Models

- [ ] **Step 1.1**: Create `src/config.py` with Pydantic models: `DocumentChunk`, `Document`, `DocumentConstraints`
  - **Test**: Instantiate each model with sample data, no validation errors

- [ ] **Step 1.2**: Add `LearningConfig`, `LearningMetrics` models to `src/config.py`
  - **Test**: Instantiate with defaults, verify default values match spec

- [ ] **Step 1.3**: Add `Question`, `Answer`, `ComparisonResult` models to `src/config.py`
  - **Test**: Instantiate each, serialize to JSON, deserialize back

---

## Phase 2: Document Processing

- [ ] **Step 2.1**: Create `src/document/__init__.py` (empty)
  - **Test**: `from src.document import *` succeeds

- [ ] **Step 2.2**: Create `src/document/pdf_parser.py` with `PDFParser.parse(file_bytes) -> (text, page_count)`
  - **Test**: Parse a sample PDF, verify text length > 0 and page_count > 0

- [ ] **Step 2.3**: Add error handling to `PDFParser` for corrupt/invalid PDFs
  - **Test**: Pass garbage bytes, verify raises `PDFExtractionError`

- [ ] **Step 2.4**: Create `src/document/chunker.py` with `DocumentChunker.__init__(tokenizer, chunk_size)`
  - **Test**: Instantiate with TinyLlama tokenizer, no errors

- [ ] **Step 2.5**: Implement `DocumentChunker.chunk(text) -> List[DocumentChunk]`
  - **Test**: Chunk a 5000-token text with chunk_size=2048, get 3 chunks, each ≤2048 tokens

- [ ] **Step 2.6**: Verify chunker preserves all tokens (no content lost)
  - **Test**: Concatenate all chunk token_ids, compare to original tokenization

- [ ] **Step 2.7**: Create `src/document/validator.py` with `DocumentValidator.validate(file_bytes, constraints)`
  - **Test**: Pass 200-page PDF, get `(False, "exceeds max_pages")`

---

## Phase 3: TTT-Linear Layer

- [ ] **Step 3.1**: Create `src/models/__init__.py` (empty)
  - **Test**: `from src.models import *` succeeds

- [ ] **Step 3.2**: Create `src/models/ttt_linear.py` with `TTTLinear.__init__(input_dim, hidden_dim, output_dim, inner_lr)`
  - **Test**: Instantiate `TTTLinear(768, 2048, 768)`, verify `W_h.shape == (2048, 768)`

- [ ] **Step 3.3**: Implement `TTTLinear.forward(x, learning=False)` - inference mode only
  - **Test**: Pass `[1, 128, 768]` tensor, output shape `[1, 128, 768]`

- [ ] **Step 3.4**: Store initial weights in `TTTLinear._W_h_initial` for reset
  - **Test**: After init, `torch.allclose(W_h, _W_h_initial)` is True

- [ ] **Step 3.5**: Implement `TTTLinear.forward(x, learning=True)` - compute self-supervised loss, update W_h
  - **Test**: Call with `learning=True`, verify `W_h` differs from `_W_h_initial`

- [ ] **Step 3.6**: Implement `TTTLinear.reset_weights()`
  - **Test**: Learn, reset, verify `torch.allclose(W_h, _W_h_initial)`

- [ ] **Step 3.7**: Implement `TTTLinear.get_weight_delta() -> float`
  - **Test**: Learn, call `get_weight_delta()`, verify > 0

- [ ] **Step 3.8**: Verify gradient flow through TTTLinear in inference mode
  - **Test**: Backward pass on output, verify input.grad is not None

---

## Phase 4: TinyLlama Integration

- [ ] **Step 4.1**: Create `src/models/ttt_model.py` with `TTTModel` class skeleton
  - **Test**: Class imports without errors

- [ ] **Step 4.2**: Implement `TTTModel.from_pretrained()` - load TinyLlama, no modifications yet
  - **Test**: Load model, run `model.generate("Hello")`, get valid output

- [ ] **Step 4.3**: Identify MLP layers in TinyLlama architecture (print layer names)
  - **Test**: Print `model.named_modules()`, identify MLP module path pattern

- [ ] **Step 4.4**: Replace ONE MLP layer (layer 0) with TTTLinear
  - **Test**: `model.model.layers[0].mlp` is instance of `TTTLinear`

- [ ] **Step 4.5**: Verify model still runs forward pass after single layer replacement
  - **Test**: `model.generate("Hello")` produces output (may be garbage, that's ok)

- [ ] **Step 4.6**: Replace ALL MLP layers with TTTLinear (or configurable subset)
  - **Test**: Count TTTLinear instances == number of transformer layers

- [ ] **Step 4.7**: Implement `TTTModel.reset_learning()` - reset all TTTLinear layers
  - **Test**: Learn on text, reset, verify all `get_weight_delta()` == 0

- [ ] **Step 4.8**: Implement `TTTModel.clear_context()` - clear KV cache
  - **Test**: Generate, clear, generate again - no error

---

## Phase 5: LaCT (Large Chunk TTT)

- [ ] **Step 5.1**: Create `src/models/lact.py` with `LaCTUpdater` class skeleton
  - **Test**: Class imports without errors

- [ ] **Step 5.2**: Implement `LaCTUpdater.process_chunk(tokens)` - forward + compute loss
  - **Test**: Process 2048 tokens, get scalar loss value

- [ ] **Step 5.3**: Implement gradient accumulation across chunk
  - **Test**: Process chunk, verify accumulated gradients are non-None

- [ ] **Step 5.4**: Implement `LaCTUpdater.apply_update()` - single weight update from accumulated grads
  - **Test**: Process chunk, apply update, verify `get_weight_delta() > 0`

- [ ] **Step 5.5**: Implement `LaCTUpdater.process_document(chunks)` - loop over all chunks
  - **Test**: Process 5 chunks, verify weights updated, loss decreases

---

## Phase 6: Learning Pipeline

- [ ] **Step 6.1**: Create `src/learning/__init__.py` (empty)
  - **Test**: `from src.learning import *` succeeds

- [ ] **Step 6.2**: Create `src/learning/metrics.py` with `MetricsTracker` class
  - **Test**: Instantiate, call `record_loss(0.5)`, verify `loss_history == [0.5]`

- [ ] **Step 6.3**: Implement `MetricsTracker.get_metrics() -> LearningMetrics`
  - **Test**: Record losses, get metrics, verify `initial_loss` and `final_loss` correct

- [ ] **Step 6.4**: Create `src/learning/trainer.py` with `TTTTrainer.__init__(model, config)`
  - **Test**: Instantiate with model and config

- [ ] **Step 6.5**: Implement `TTTTrainer.train_on_document(document)` - orchestrate learning
  - **Test**: Train on 3-chunk doc, get `LearningMetrics` with `final_loss < initial_loss`

- [ ] **Step 6.6**: Add progress callback support to `TTTTrainer.train_on_document()`
  - **Test**: Pass callback, verify called with (chunk_idx, total, loss) for each chunk

---

## Phase 7: Inference

- [ ] **Step 7.1**: Create `src/inference/__init__.py` (empty)
  - **Test**: `from src.inference import *` succeeds

- [ ] **Step 7.2**: Create `src/inference/generator.py` with `Generator.__init__(model, tokenizer)`
  - **Test**: Instantiate with TTTModel

- [ ] **Step 7.3**: Implement `Generator.generate(prompt, max_tokens, temperature) -> Answer`
  - **Test**: Generate answer, verify `Answer.text` is non-empty string

- [ ] **Step 7.4**: Implement `Generator.compare(prompt) -> (ttt_answer, base_answer)`
  - **Test**: After learning, compare returns two different answers

---

## Phase 8: Integration Test (Notebook)

- [ ] **Step 8.1**: Create `notebooks/05_integration.ipynb`
  - **Test**: Notebook opens in Colab

- [ ] **Step 8.2**: End-to-end test: PDF → parse → chunk → learn → clear → Q&A
  - **Test**: TTT answer contains doc-specific info, base answer doesn't

- [ ] **Step 8.3**: Memory test: Process 20-page PDF, monitor VRAM
  - **Test**: Peak VRAM < 14GB, no OOM

- [ ] **Step 8.4**: Latency test: Measure time per chunk
  - **Test**: Average < 3s per 2048-token chunk on T4

---

## Phase 9: Gradio UI

- [ ] **Step 9.1**: Create `app.py` skeleton with Gradio Blocks layout
  - **Test**: `python app.py` launches UI on localhost

- [ ] **Step 9.2**: Implement PDF upload component + file preview
  - **Test**: Upload PDF, see filename displayed

- [ ] **Step 9.3**: Implement "Start Learning" button + progress bar
  - **Test**: Click button, progress bar updates during learning

- [ ] **Step 9.4**: Add learning metrics display (loss curve, tokens processed)
  - **Test**: After learning, see loss value and token count

- [ ] **Step 9.5**: Add "Context Cleared" indicator
  - **Test**: After learning complete, indicator visible

- [ ] **Step 9.6**: Implement question input + submit button
  - **Test**: Type question, click submit, see response

- [ ] **Step 9.7**: Implement side-by-side comparison display (TTT vs Base)
  - **Test**: After Q&A, see two answer columns

- [ ] **Step 9.8**: Implement "Reset" button to clear session
  - **Test**: Click reset, UI clears, can upload new PDF

- [ ] **Step 9.9**: Add error handling UI (invalid PDF, too short, etc.)
  - **Test**: Upload corrupt PDF, see error message

---

## Phase 10: Deployment

- [ ] **Step 10.1**: Create HF Spaces repo, add README.md with Spaces metadata
  - **Test**: Spaces page shows "Building"

- [ ] **Step 10.2**: Push code to HF Spaces repo
  - **Test**: Spaces builds successfully

- [ ] **Step 10.3**: Test demo on HF Spaces with sample PDF
  - **Test**: Full flow works on live URL

- [ ] **Step 10.4**: Update README with live demo link, Adaption Labs pitch
  - **Test**: README shows working badge/link

---

## Phase 11: Polish

- [ ] **Step 11.1**: Add sample PDF for one-click demo
  - **Test**: Demo works without user uploading anything

- [ ] **Step 11.2**: Optimize loading time (cache model weights)
  - **Test**: Cold start < 60s on HF Spaces

- [ ] **Step 11.3**: Final QA: all manual test checklist items pass
  - **Test**: See spec.md Section 6.4 checklist

---

## Summary Stats

- **Total Steps**: 58
- **Phase 0 (Setup)**: 3 steps
- **Phase 1 (Models)**: 3 steps
- **Phase 2 (Documents)**: 7 steps
- **Phase 3 (TTT Layer)**: 8 steps
- **Phase 4 (TinyLlama)**: 8 steps
- **Phase 5 (LaCT)**: 5 steps
- **Phase 6 (Learning)**: 6 steps
- **Phase 7 (Inference)**: 4 steps
- **Phase 8 (Integration)**: 4 steps
- **Phase 9 (UI)**: 9 steps
- **Phase 10 (Deploy)**: 4 steps
- **Phase 11 (Polish)**: 3 steps
