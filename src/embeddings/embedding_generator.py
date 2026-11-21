"""
Embedding Generator for Legal Texts using Transformer Encoders.

Supports batched inference with attention-mask-aware mean pooling, L2 normalization,
checkpointing, and resumable execution for candidate chunks and query documents.
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from src.embeddings.pooling import masked_mean_pooling, l2_normalize


class EmbeddingGenerator:
    """Generates and manages dense embeddings for text corpora using transformer backbones."""

    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
        use_fp16: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model.to(self.device)
        self.model.eval()

    def encode_texts(
        self,
        texts: Sequence[str],
        show_progress: bool = True,
        progress_interval: int = 250,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> np.ndarray:
        """
        Encodes a sequence of text strings into an L2-normalized float32 numpy array.
        
        Args:
            texts: Sequence of strings.
            show_progress: Whether to log periodic throughput.
            progress_interval: Number of chunks between log messages.
            log_callback: Optional logging callback function.
            
        Returns:
            np.ndarray: Array of shape (n_texts, hidden_size) with dtype float32.
        """
        n_texts = len(texts)
        if n_texts == 0:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)

        embeddings_list: List[np.ndarray] = []
        t0 = time.time()
        last_log_time = t0

        for start_idx in range(0, n_texts, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_texts)
            batch_texts = list(texts[start_idx:end_idx])

            # Dynamic padding within batch up to self.max_length
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model(**inputs)
                # Masked mean pooling over transformer hidden states
                pooled = masked_mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                # L2 normalize pooled representations
                normed = l2_normalize(pooled)
                batch_np = normed.detach().cpu().to(torch.float32).numpy()

            embeddings_list.append(batch_np)

            # Logging throughput
            if show_progress and (end_idx % progress_interval == 0 or end_idx == n_texts):
                now = time.time()
                elapsed = now - t0
                rate = end_idx / max(elapsed, 1e-6)
                msg = f"  Encoded {end_idx}/{n_texts} chunks ({rate:.1f} chunks/sec | elapsed {elapsed:.1f}s)"
                if log_callback:
                    log_callback(msg)
                else:
                    print(msg, flush=True)

        return np.vstack(embeddings_list)

    def encode_and_save_split(
        self,
        split_name: str,
        chunks_df: pd.DataFrame,
        output_dir: Path,
        model_name: str,
        checkpoint_interval: int = 5000,
    ) -> Tuple[np.ndarray, Path]:
        """
        Encodes a partition's chunks with checkpointing and saves both embeddings and metadata.
        
        Args:
            split_name: Partition name (e.g. 'dev_candidates').
            chunks_df: DataFrame containing 'doc_id', 'chunk_id', 'chunk_idx', and 'text'.
            output_dir: Target directory under models/embeddings/{model_slug}/{split_name}.
            model_name: Identifier of the transformer model.
            checkpoint_interval: Chunks interval for persisting checkpoint arrays.
            
        Returns:
            (embeddings_array, metadata_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        npy_path = output_dir / "embeddings.npy"
        meta_path = output_dir / "metadata.parquet"
        ckpt_path = output_dir / "checkpoint.npy"

        n_chunks = len(chunks_df)
        hidden_dim = self.model.config.hidden_size

        # If already completed, load and return
        if npy_path.exists() and meta_path.exists():
            print(f"Loading existing embeddings from {npy_path}...")
            embeddings = np.load(npy_path)
            if embeddings.shape[0] == n_chunks and embeddings.shape[1] == hidden_dim:
                return embeddings, meta_path

        # Resumable checkpointing check
        start_chunk = 0
        existing_chunks = []
        if ckpt_path.exists():
            ckpt = np.load(ckpt_path)
            if ckpt.shape[1] == hidden_dim and ckpt.shape[0] < n_chunks:
                start_chunk = ckpt.shape[0]
                existing_chunks.append(ckpt)
                print(f"Resuming encoding from checkpoint at chunk {start_chunk}/{n_chunks}...")

        texts = chunks_df["text"].values
        new_embeddings = []

        t0 = time.time()
        for idx in range(start_chunk, n_chunks, self.batch_size):
            end_idx = min(idx + self.batch_size, n_chunks)
            batch_texts = list(texts[idx:end_idx])

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model(**inputs)
                pooled = masked_mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                normed = l2_normalize(pooled)
                batch_np = normed.detach().cpu().to(torch.float32).numpy()

            new_embeddings.append(batch_np)

            # Periodic checkpointing
            if (end_idx - start_chunk) % checkpoint_interval == 0 or end_idx == n_chunks:
                current_all = np.vstack(existing_chunks + new_embeddings)
                np.save(ckpt_path, current_all)
                rate = end_idx / max(time.time() - t0, 1e-6)
                print(f"  [{split_name}] Encoded {end_idx}/{n_chunks} chunks ({rate:.1f} chunks/sec) - Checkpoint saved.")

        final_embeddings = np.vstack(existing_chunks + new_embeddings)
        assert final_embeddings.shape == (n_chunks, hidden_dim), (
            f"Shape mismatch: {final_embeddings.shape} vs expected ({n_chunks}, {hidden_dim})"
        )

        # Save final artifacts
        np.save(npy_path, final_embeddings)

        # Remove temporary checkpoint file if complete
        if ckpt_path.exists():
            ckpt_path.unlink()

        # Build and save metadata
        metadata_df = chunks_df[["doc_id", "chunk_id", "chunk_idx"]].copy()
        metadata_df["split"] = split_name
        metadata_df["model"] = model_name
        metadata_df["embedding_dim"] = hidden_dim
        metadata_df.to_parquet(meta_path, index=False)

        print(f"Successfully saved {split_name} embeddings ({final_embeddings.shape}) to {npy_path}")
        return final_embeddings, meta_path

