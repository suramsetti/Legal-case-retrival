"""
Phase 18: Jud-IPL Forensic Audit and Research Task Feasibility Analysis.

Performs a rigorous audit of the standalone Jud-IPL dataset:
- data/processed/judipl/documents.parquet
- data/processed/judipl/chunks.parquet
- data/processed/judipl/legal_features.json

Generates:
- results/tables/phase18_judipl_column_audit.csv
- results/tables/phase18_judipl_label_distribution.csv
- results/tables/phase18_judipl_task_feasibility.csv
"""

import sys
import os
import re
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

DOCS_PATH = PROJECT_ROOT / "data" / "processed" / "judipl" / "documents.parquet"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "judipl" / "chunks.parquet"
FEATS_PATH = PROJECT_ROOT / "data" / "processed" / "judipl" / "legal_features.json"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def run_phase18_audit():
    print("=" * 80)
    print("PHASE 18A & 18B: JUD-IPL DATASET FORENSIC AUDIT & FEASIBILITY ANALYSIS")
    print("=" * 80)
    t0 = time.time()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading Jud-IPL processed artifacts...")
    df = pd.read_parquet(DOCS_PATH)
    print(f"  Documents loaded: {len(df):,} rows, {len(df.columns)} columns")

    chunks_df = pd.read_parquet(CHUNKS_PATH) if CHUNKS_PATH.exists() else pd.DataFrame()
    print(f"  Chunks loaded: {len(chunks_df):,} rows")

    with open(FEATS_PATH) as f:
        feats = json.load(f)
    print(f"  Legal features loaded: {len(feats):,} entries")

    # -------------------------------------------------------------
    # 1. Column Audit
    # -------------------------------------------------------------
    print("\n[2] Performing Column Audit...")
    col_audit = []
    for col in df.columns:
        s = df[col]
        non_null = s.notnull().sum()
        non_empty = ((s != "") & (s.notnull())).sum()
        dtype = str(s.dtype)
        nunique = s.nunique()
        sample_val = str(s.dropna().iloc[0])[:50] if len(s.dropna()) > 0 else ""
        col_audit.append({
            "column_name": col,
            "data_type": dtype,
            "total_count": len(df),
            "non_null_count": int(non_null),
            "non_empty_count": int(non_empty),
            "completeness_pct": round(float(non_empty / len(df) * 100), 2),
            "unique_values": int(nunique),
            "sample_value": sample_val
        })

    col_df = pd.DataFrame(col_audit)
    col_csv = TABLES_DIR / "phase18_judipl_column_audit.csv"
    col_df.to_csv(col_csv, index=False)
    print(f"  Saved column audit: {col_csv}")

    # -------------------------------------------------------------
    # 2. Label & Category Distribution
    # -------------------------------------------------------------
    print("\n[3] Auditing Labels and Metadata...")
    label_dist = df.groupby(["case_category", "label"]).size().unstack(fill_value=0)
    label_dist["Total"] = label_dist.sum(axis=1)
    
    label_csv = TABLES_DIR / "phase18_judipl_label_distribution.csv"
    label_dist.to_csv(label_csv)
    print(f"  Saved label distribution: {label_csv}")

    # -------------------------------------------------------------
    # 3. Exact and Near-Duplicate Analysis
    # -------------------------------------------------------------
    print("\n[4] Analyzing Duplicates and Leakage Vectors...")
    exact_text_dups = int(df.duplicated(subset=["full_text"]).sum())
    prefix_dups = int(df["full_text"].str[:500].duplicated().sum())
    id_unique = int(df["id"].nunique())

    print(f"  Unique Kanoon IDs: {id_unique:,}")
    print(f"  Exact full_text duplicates: {exact_text_dups:,} ({exact_text_dups/len(df)*100:.2f}%)")
    print(f"  Near-duplicates (first 500 chars): {prefix_dups:,} ({prefix_dups/len(df)*100:.2f}%)")

    # -------------------------------------------------------------
    # 4. Text Quality and Truncation
    # -------------------------------------------------------------
    chars = df["char_length"]
    words = df["word_length"]
    sents = df["sentence_count"]
    trunc_count = int((chars >= 32765).sum())

    print(f"  Character length: median={chars.median():.0f}, mean={chars.mean():.1f}, min={chars.min()}, max={chars.max()}")
    print(f"  Word count: median={words.median():.0f}, mean={words.mean():.1f}, min={words.min()}, max={words.max()}")
    print(f"  Sentence count: median={sents.median():.0f}, mean={sents.mean():.1f}, min={sents.min()}, max={sents.max()}")
    print(f"  Documents truncated at 32,765 chars (CSV cell limit): {trunc_count:,} ({trunc_count/len(df)*100:.2f}%)")

    # -------------------------------------------------------------
    # 5. Proof Sentence Analysis
    # -------------------------------------------------------------
    has_proof = (df["proof_sentence"] != "") & (df["proof_sentence"].notnull())
    print(f"  Proof sentence coverage: {has_proof.sum():,} / {len(df):,} ({has_proof.mean()*100:.2f}%)")

    # -------------------------------------------------------------
    # 6. Task Feasibility Evaluation
    # -------------------------------------------------------------
    print("\n[5] Evaluating Task Feasibility (Phase 18B)...")
    tasks = [
        {
            "task_id": "T1",
            "task_name": "Legal Outcome Prediction (LJP)",
            "supervision_type": "Gold standard (3-class & binary)",
            "available_samples": 42005,
            "label_coverage_pct": 100.0,
            "feasibility_verdict": "HIGHLY FEASIBLE (RECOMMENDED PRIMARY BENCHMARK)",
            "risk_mitigation": "Must strictly strip/mask dispositive holding sentences from input text to avoid label leakage.",
            "recommended_metrics": "Accuracy, Macro-F1, Weighted-F1, Precision, Recall, ROC-AUC"
        },
        {
            "task_id": "T2",
            "task_name": "Legal Holding Extraction (Proof Sentence Detection)",
            "supervision_type": "Gold standard sentence rationale",
            "available_samples": 35139,  # non-truncated
            "label_coverage_pct": 83.65,
            "feasibility_verdict": "FEASIBLE (RECOMMENDED SECONDARY TASK)",
            "risk_mitigation": "Exclude documents truncated at 32,765 chars where proof sentence occurs past the cutoff.",
            "recommended_metrics": "Top-1 Sentence Accuracy, MRR, Sentence Precision, Recall, F1"
        },
        {
            "task_id": "T3",
            "task_name": "Legal Domain / Category Classification",
            "supervision_type": "Case category metadata (Civil/Criminal)",
            "available_samples": 34658,
            "label_coverage_pct": 82.51,
            "feasibility_verdict": "FEASIBLE (AUXILIARY TASK)",
            "risk_mitigation": "Filter out 7,348 records with empty case_category.",
            "recommended_metrics": "Macro-F1, Accuracy, Balanced Accuracy"
        },
        {
            "task_id": "T4",
            "task_name": "Unsupervised Legal Judgment Clustering & Similarity",
            "supervision_type": "Unsupervised / Weak pseudo-labels",
            "available_samples": 42006,
            "label_coverage_pct": "N/A (Unsupervised)",
            "feasibility_verdict": "FEASIBLE (REPRESENTATION ANALYSIS ONLY)",
            "risk_mitigation": "Evaluate via extrinsic cluster alignment (NMI, ARI) against known metadata; do not claim IR relevance.",
            "recommended_metrics": "Silhouette Score, Davies-Bouldin, NMI, ARI"
        },
        {
            "task_id": "T5",
            "task_name": "Prior Case Retrieval / Precedent Discovery",
            "supervision_type": "NONE (No query-precedent relevance pairs exist in Jud-IPL)",
            "available_samples": 0,
            "label_coverage_pct": 0.0,
            "feasibility_verdict": "NOT DEFENSIBLE / REJECTED",
            "risk_mitigation": "DO NOT FABRICATE RELEVANCE LABELS. Jud-IPL is an outcome prediction dataset, NOT a retrieval benchmark.",
            "recommended_metrics": "N/A (Cannot compute P@1, MRR, nDCG without ground truth)"
        }
    ]

    task_df = pd.DataFrame(tasks)
    task_csv = TABLES_DIR / "phase18_judipl_task_feasibility.csv"
    task_df.to_csv(task_csv, index=False)
    print(f"  Saved task feasibility table: {task_csv}")

    print("\n" + "=" * 80)
    print(f"Phase 18 Audit completed in {time.time()-t0:.2f}s!")
    print("=" * 80)
    return col_df, label_dist, task_df


if __name__ == "__main__":
    run_phase18_audit()

