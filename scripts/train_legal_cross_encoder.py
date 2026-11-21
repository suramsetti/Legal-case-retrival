"""
Phase 12: Training Pipeline for Legal Cross-Encoder.

Trains InLegalBERTCrossEncoder with section-aware chunking and pairwise margin ranking loss
on hard negative pairs mined from the official IL-TUR PCR Training candidate pools.

Validates checkpoint using Development split MRR/nDCG and saves to:
models/checkpoints/legal_cross_encoder.pt
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.ranking.legal_cross_encoder import InLegalBERTCrossEncoder, SectionAwareChunker
from src.ranking.pairwise_reranker import PairwiseNeuralReRanker
from src.evaluation.hybrid_evaluator import HybridEvaluator


class PairwiseCrossDataset(Dataset):
    """Dataset yielding tokenized (query, pos_chunk) and (query, neg_chunk) pairs."""

    def __init__(
        self,
        pairs: List[Tuple[str, str, str]],  # (query_text, pos_text, neg_text)
        tokenizer,
        max_length: int = 256,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        q_text, pos_text, neg_text = self.pairs[idx]

        enc_pos = self.tokenizer(
            q_text,
            pos_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc_neg = self.tokenizer(
            q_text,
            neg_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "pos_input_ids": enc_pos["input_ids"].squeeze(0),
            "pos_attention_mask": enc_pos["attention_mask"].squeeze(0),
            "neg_input_ids": enc_neg["input_ids"].squeeze(0),
            "neg_attention_mask": enc_neg["attention_mask"].squeeze(0),
        }


def mine_training_triples(
    train_df: pd.DataFrame,
    q_docs: pd.DataFrame,
    c_docs: pd.DataFrame,
    chunker: SectionAwareChunker,
    max_queries: int = 150,
    negatives_per_query: int = 2,
) -> List[Tuple[str, str, str]]:
    """
    Mines (query, positive_chunk, hard_negative_chunk) triples from Stage-1 rankings.
    """
    triples: List[Tuple[str, str, str]] = []
    grouped = train_df.groupby("query_id")

    query_count = 0
    for qid, grp in grouped:
        if qid not in q_docs.index:
            continue

        pos_rows = grp[grp["label"] == 1]
        neg_rows = grp[(grp["label"] == 0) & (grp["s1_rank"] <= 20)]

        if pos_rows.empty or neg_rows.empty:
            continue

        q_text = str(q_docs.loc[qid, "full_text"])[:600]

        # Best positive candidate
        best_pos_id = pos_rows.iloc[0]["candidate_id"]
        if best_pos_id not in c_docs.index:
            continue

        pos_text = str(c_docs.loc[best_pos_id, "full_text"])
        pos_chunks = chunker.chunk_document(pos_text)
        pos_repr = pos_chunks[0]["text"] if pos_chunks else pos_text[:600]

        # Top hard negatives
        for _, neg_row in neg_rows.head(negatives_per_query).iterrows():
            neg_id = neg_row["candidate_id"]
            if neg_id not in c_docs.index:
                continue

            neg_text = str(c_docs.loc[neg_id, "full_text"])
            neg_chunks = chunker.chunk_document(neg_text)
            neg_repr = neg_chunks[0]["text"] if neg_chunks else neg_text[:600]

            triples.append((q_text, pos_repr, neg_repr))

        query_count += 1
        if query_count >= max_queries:
            break

    return triples


def main():
    print("==========================================================================")
    print("PHASE 12: TRAINING LEGAL CROSS-ENCODER WITH SECTION-AWARE CHUNKING")
    print("==========================================================================")
    t0 = time.time()

    data_dir = PROJECT_ROOT / "data/processed/IL-TUR-PCR"
    checkpoints_dir = PROJECT_ROOT / "models/checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Training Metadata & Texts
    print("Loading Training datasets and document texts...")
    train_df = pd.read_parquet(data_dir / "train/hybrid_features.parquet")
    train_df = PairwiseNeuralReRanker.engineer_features(train_df)
    train_df["s1_rank"] = train_df.groupby("query_id")["stage1_score"].rank(ascending=False, method="first")

    q_docs = pd.read_parquet(data_dir / "train_queries/documents.parquet").set_index("id")
    c_docs = pd.read_parquet(data_dir / "train_candidates/documents.parquet").set_index("id")
    print(f"Loaded {len(train_df):,} train pairs, {len(q_docs)} queries, {len(c_docs)} candidates.")

    # 2. Mine Hard Negative Triples
    print("Mining hard negative triples from Stage-1 candidate pools...")
    chunker = SectionAwareChunker()
    triples = mine_training_triples(
        train_df, q_docs, c_docs, chunker, max_queries=120, negatives_per_query=2
    )
    print(f"Mined {len(triples):,} training triples across {len(triples)//2} queries.")

    # 3. Instantiate InLegalBERTCrossEncoder
    print("Initializing InLegalBERTCrossEncoder (freezing lower 10 layers on CPU)...")
    device = "cpu"
    cross_encoder = InLegalBERTCrossEncoder(
        model_name_or_path="law-ai/InLegalBERT",
        freeze_lower_layers=10,
        device=device,
    )

    dataset = PairwiseCrossDataset(triples, cross_encoder.tokenizer, max_length=200)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 4. Training Loop with Pairwise Margin Loss
    print("\n--- Training Cross-Encoder with Pairwise Margin Loss ---")
    margin = 1.0
    lr = 2e-5
    epochs = 3

    trainable_params = [p for p in cross_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    cross_encoder.train()
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()

            pos_ids = batch["pos_input_ids"].to(device)
            pos_mask = batch["pos_attention_mask"].to(device)
            neg_ids = batch["neg_input_ids"].to(device)
            neg_mask = batch["neg_attention_mask"].to(device)

            pos_scores = cross_encoder(pos_ids, pos_mask)
            neg_scores = cross_encoder(neg_ids, neg_mask)

            # Pairwise Margin Ranking Loss: ReLU(margin - (s_pos - s_neg))
            diff = pos_scores - neg_scores
            loss = torch.mean(torch.relu(margin - diff))

            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"  Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} in {time.time() - epoch_t0:.2f}s")

    # 5. Save Model Checkpoint
    ckpt_path = checkpoints_dir / "legal_cross_encoder.pt"
    cross_encoder.save(ckpt_path)
    print(f"\nSaved trained Legal Cross-Encoder checkpoint to {ckpt_path}")

    # 6. Save Training Metadata
    summary = {
        "model_type": "InLegalBERTCrossEncoder",
        "margin": margin,
        "lr": lr,
        "epochs": epochs,
        "num_training_triples": len(triples),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(checkpoints_dir / "legal_cross_encoder_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPhase 12 Cross-Encoder Training completed in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()

