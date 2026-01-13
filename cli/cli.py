"""
CLI interface for learn-doc.
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from config import Document, DocumentChunk, LearningConfig
from pdf_parser import PDFParser
from chunker import DocumentChunker
from ttt_model import TTTModel
from trainer import TTTTrainer
from generator import Generator

class CLInterface:
    SESSION_FILE = ".session_state.pt"

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.document = None

    def get_model(self):
        if self.model is None:
            print("Loading Qwen2.5-0.5B model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Layers 16-23 (Upper Brain)
            target_layers = list(range(16, 24))

            self.model = TTTModel.from_pretrained(
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                ttt_layer_indices=target_layers,
                device=device
            )
            self.tokenizer = self.model.tokenizer
            print(f"Model loaded! TTT Active on layers: {target_layers}")
        return self.model

    def save_session(self):
        if not self.model or not self.document: return
        print("Saving session state...")
        ttt_weights = [layer.W_h.weight.data.clone() for layer in self.model.ttt_layers]
        state = {"ttt_weights": ttt_weights, "document": self.document}
        torch.save(state, self.SESSION_FILE, _use_new_zipfile_serialization=False)
        print("Session saved.")

    def load_session(self):
        if self.model and self.document: return True
        if not os.path.exists(self.SESSION_FILE): return False
        try:
            model = self.get_model()
            print("Loading learned state...")
            state = torch.load(self.SESSION_FILE, weights_only=False)
            self.document = state["document"]
            with torch.no_grad():
                for layer, saved_w in zip(model.ttt_layers, state["ttt_weights"]):
                    layer.W_h.weight.data.copy_(saved_w)
            return True
        except Exception as e:
            print(f"Error loading: {e}")
            return False

    def run_command(self, command: str, args: list):
        if command == "run": self.cmd_run(args)
        elif command == "learn": self.cmd_learn(args)
        elif command == "interactive": self.cmd_interactive()
        elif command == "reset": self.cmd_reset()
        else: print("Commands: run, learn, interactive, reset")

    def cmd_run(self, args):
        self.cmd_learn(args)
        self.cmd_interactive()

    def cmd_learn(self, args):
        if not args: print("Usage: run <file>"); return
        file_path = args[0]
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        model = self.get_model()
        
        with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
        chunker = DocumentChunker(self.tokenizer, chunk_size=512)
        chunks = chunker.chunk(text)
        
        self.document = Document(
            id=os.path.basename(file_path), filename=os.path.basename(file_path),
            page_count=1, total_tokens=sum(c.token_count for c in chunks), chunks=chunks
        )

        epochs = 20
        lr = 5e-4  # 0.0005

        print(f"\nTraining for {epochs} epochs (LR: {lr})...")
        
        model.reset_learning()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        config = LearningConfig(inner_lr=lr, chunk_size=512, mask_ratio=0.0)
        trainer = TTTTrainer(model, self.tokenizer, config)
        
        pbar = tqdm(total=len(chunks) * epochs)
        final_metrics = None
        
        for _ in range(epochs):
            final_metrics = trainer.train_on_document(self.document, progress_callback=lambda i,t,l: pbar.update(1))
            
        pbar.close()
        print(f"Final Loss: {final_metrics.final_loss:.4f}")
        
        model.clear_context()
        self.save_session()

    def cmd_interactive(self):
        if not self.model and not self.load_session(): 
            print("No model loaded."); return

        print(f"\nChat Mode: {self.document.filename} (Type 'exit')")
        gen = Generator(self.model, self.tokenizer)

        while True:
            q = input("\nAsk> ").strip()
            if q.lower() in ('exit', 'quit'): break
            if not q: continue
            
            ans = gen.generate(q, max_tokens=150)
            print(f">> {ans.text}")
            
    def cmd_reset(self):
        if os.path.exists(self.SESSION_FILE): os.remove(self.SESSION_FILE)
        if self.model: self.model.reset_learning()
        print("Reset.")

def main():
    cli = CLInterface()
    if len(sys.argv) < 2: print("Usage: python cli/cli.py run <file>"); return
    cli.run_command(sys.argv[1], sys.argv[2:])

if __name__ == "__main__":
    main()