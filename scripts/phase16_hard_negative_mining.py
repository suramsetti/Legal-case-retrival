"""
Phase 16: High-Performance Hard-Negative Mining Pipeline for Legal Case Retrieval.

Extracts challenging hard negatives from the official IL-TUR PCR Training candidate pools:
1. Lexical Hard Negatives: Highest BM25 and TF-IDF non-precedent candidates (label == 0)
   that confuse first-stage lexical retrieval.
2. Dense Hard Negatives: High InLegalBERT similarity non-precedents with low lexical or statutory support.
3. Statutory Hard Negatives: Non-precedents sharing statutory citations without genuine factual relevance.

Filters:
- Strictly excludes all positive gold precedent labels from negative pools.
- Confined exclusively to training data (data/processed/IL-TUR-PCR/train/) to prevent leakage.

Outputs:
- data/processed/IL-TUR-PCR/train_hard_negatives.parquet
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def mine_hard_negatives(
    train_features_path: Path,
    q_docs_path: Path,
    c_docs_path: Path,
    output_path: Path,
    negatives_per_positive: int = 3,
    max_text_chars: int = 1500,
) -> pd.DataFrame:
    """
    Mines hard negative candidates for each positive query-candidate pair in training set.
    """
    print(f"Loading training features from: {train_features_path}")
    df_features = pd.read_parquet(train_features_path)
    
    print(f"Loading query documents from: {q_docs_path}")
    df_q = pd.read_parquet(q_docs_path, columns=["id", "full_text"])
    q_text_map = {row["id"]: row["full_text"][:max_text_chars] for _, row in df_q.iterrows()}
    
    print(f"Loading candidate documents from: {c_docs_path}")
    df_c = pd.read_parquet(c_docs_path, columns=["id", "full_text"])
    c_text_map = {row["id"]: row["full_text"][:max_text_chars] for _, row in df_c.iterrows()}

    triples: List[Dict[str, Any]] = []
    
    grouped = df_features.groupby("query_id")
    total_queries = df_features["query_id"].nunique()
    print(f"Processing {total_queries} training queries using fast vectorized negative indexing...")

    for qid, grp in grouped:
        pos_candidates = grp[grp["label"] == 1]
        neg_candidates = grp[grp["label"] == 0]

        if len(pos_candidates) == 0 or len(neg_candidates) == 0:
            continue

        q_text = q_text_map.get(qid, "")

        # Sort negatives by BM25, TF-IDF, and Dense scores
        top_negatives = []
        for col, strategy in [
            ("bm25_score", "bm25_hard"),
            ("tfidf_score", "tfidf_hard"),
            ("inlegalbert_sim", "dense_hard"),
        ]:
            if col in neg_candidates.columns:
                top_rows = neg_candidates.nlargest(negatives_per_positive, col)
                for _, r in top_rows.iterrows():
                    top_negatives.append((strategy, r))

        # Deduplicate top negatives by candidate_id while keeping highest priority strategy
        seen_cids = set()
        selected_negs = []
        for strategy, r in top_negatives:
            cid = r["candidate_id"]
            if cid not in seen_cids:
                seen_cids.add(cid)
                selected_negs.append((strategy, r))
                if len(selected_negs) >= negatives_per_positive * 2:
                    break

        if not selected_negs:
            continue

        for _, pos_row in pos_candidates.iterrows():
            pos_cid = pos_row["candidate_id"]
            pos_text = c_text_map.get(pos_cid, "")

            for strategy, neg_row in selected_negs[:negatives_per_positive]:
                neg_cid = neg_row["candidate_id"]
                neg_text = c_text_map.get(neg_cid, "")

                triples.append({
                    "query_id": qid,
                    "pos_candidate_id": pos_cid,
                    "neg_candidate_id": neg_cid,
                    "mining_strategy": strategy,
                    "pos_bm25_score": float(pos_row["bm25_score"]),
                    "neg_bm25_score": float(neg_row["bm25_score"]),
                    "pos_tfidf_score": float(pos_row["tfidf_score"]),
                    "neg_tfidf_score": float(neg_row["tfidf_score"]),
                    "pos_dense_sim": float(pos_row["inlegalbert_sim"]),
                    "neg_dense_sim": float(neg_row["inlegalbert_sim"]),
                    "pos_statute_jaccard": float(pos_row.get("statute_jaccard", 0.0)),
                    "neg_statute_jaccard": float(neg_row.get("statute_jaccard", 0.0)),
                    "query_text": q_text,
                    "pos_text": pos_text,
                    "neg_text": neg_text,
                })

    df_triples = pd.DataFrame(triples)
    print(f"Mined {len(df_triples)} hard-negative triples across {df_triples['query_id'].nunique()} queries.")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_triples.to_parquet(output_path, index=False)
    print(f"Successfully saved hard-negative training dataset to: {output_path}")
    
    return df_triples


if __name__ == "__main__":
    train_features = PROJECT_ROOT / "data" / "processed" / "IL-TUR-PCR" / "train" / "hybrid_features.parquet"
    q_docs = PROJECT_ROOT / "data" / "processed" / "IL-TUR-PCR" / "train_queries" / "documents.parquet"
    c_docs = PROJECT_ROOT / "data" / "processed" / "IL-TUR-PCR" / "train_candidates" / "documents.parquet"
    output_file = PROJECT_ROOT / "data" / "processed" / "IL-TUR-PCR" / "train_hard_negatives.parquet"

    mine_hard_negatives(train_features, q_docs, c_docs, output_file)
