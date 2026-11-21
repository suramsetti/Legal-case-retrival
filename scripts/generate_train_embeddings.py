"""Generates InLegalBERT full-doc embeddings for train_queries and train_candidates."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.embeddings.model_loader import ModelLoader
from src.embeddings.embedding_generator import EmbeddingGenerator

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "IL-TUR-PCR"
MODELS_DIR = PROJECT_ROOT / "models" / "embeddings" / "inlegalbert"

def main():
    print("=================================================================")
    print("GENERATING TRAIN PARTITION INLEGALBERT EMBEDDINGS")
    print("=================================================================")

    # Configure PyTorch CPU threads
    num_threads = min(8, torch.get_num_threads())
    torch.set_num_threads(num_threads)
    print(f"Configured PyTorch CPU threads: {num_threads}")

    model, tokenizer, device = ModelLoader.load_model_and_tokenizer("inlegalbert", device="cpu")
    generator = EmbeddingGenerator(model, tokenizer, device=device, batch_size=64)

    # 1. train_queries (827 documents)
    tq_file = PROCESSED_DIR / "train_queries" / "documents.parquet"
    tq_save_dir = MODELS_DIR / "train_queries"
    tq_save_file = tq_save_dir / "fulldoc_embeddings.npy"

    if tq_save_file.exists():
        print(f"train_queries embeddings already exist at {tq_save_file}, skipping.")
    else:
        print(f"\n--- Encoding train_queries (827 documents) ---")
        tq_df = pd.read_parquet(tq_file)
        t0 = time.time()
        embs = generator.encode_texts(tq_df["full_text"].values, show_progress=True, progress_interval=100)
        tq_save_dir.mkdir(parents=True, exist_ok=True)
        np.save(tq_save_file, embs)
        print(f"Saved train_queries embeddings ({embs.shape}) in {time.time()-t0:.2f}s to {tq_save_file}")

    # 2. train_candidates (4,320 documents)
    tc_file = PROCESSED_DIR / "train_candidates" / "documents.parquet"
    tc_save_dir = MODELS_DIR / "train_candidates"
    tc_save_file = tc_save_dir / "fulldoc_embeddings.npy"

    if tc_save_file.exists():
        print(f"train_candidates embeddings already exist at {tc_save_file}, skipping.")
    else:
        print(f"\n--- Encoding train_candidates (4,320 documents) ---")
        tc_df = pd.read_parquet(tc_file)
        t0 = time.time()
        embs = generator.encode_texts(tc_df["full_text"].values, show_progress=True, progress_interval=500)
        tc_save_dir.mkdir(parents=True, exist_ok=True)
        np.save(tc_save_file, embs)
        print(f"Saved train_candidates embeddings ({embs.shape}) in {time.time()-t0:.2f}s to {tc_save_file}")

    print("\nTrain embedding generation complete!")

if __name__ == "__main__":
    main()

