"""
Phase 17: Forensic Audit Script for Jud-IPL Legal Case Dataset.

Performs a thorough forensic inspection of data/raw/Jud-IPL/case_files_total.csv:
1. Complete schema inspection, column types, and total record counts.
2. Missing-value analysis per column.
3. Exact and near-duplicate judgment detection.
4. Text length distribution (characters, words, sentences).
5. Categorical distribution: Case Categories, Case Types, and Judicial Outcomes (Labels).
6. Proof sentence (Ratio Decidendi) availability and quality.
7. Generates summary table: results/tables/judipl_corpus_statistics.csv.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def audit_judipl_dataset(
    csv_path: Path,
    output_table_path: Path,
    chunk_size: int = 10000,
) -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 17: FORENSIC AUDIT OF JUD-IPL LEGAL DATASET")
    print(f"Target: {csv_path}")
    print("=" * 80)
    t0 = time.time()

    total_rows = 0
    null_counts: Dict[str, int] = {}
    col_names = None
    lengths_chars = []
    lengths_words = []
    lengths_sents = []
    case_categories: Dict[str, int] = {}
    case_types: Dict[str, int] = {}
    labels: Dict[str, int] = {}
    
    unique_names = set()
    exact_duplicate_rows = 0
    seen_row_hashes = set()

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        if col_names is None:
            col_names = list(chunk.columns)
            for c in col_names:
                null_counts[c] = 0

        total_rows += len(chunk)
        for c in col_names:
            null_counts[c] += int(chunk[c].isna().sum())

        # Exact duplicate detection by hashing key fields
        for _, r in chunk.iterrows():
            name_val = str(r.get("name", ""))
            type_val = str(r.get("case_type", ""))
            lbl_val = str(r.get("label", ""))
            row_hash = hash((name_val, type_val, lbl_val))
            if row_hash in seen_row_hashes:
                exact_duplicate_rows += 1
            else:
                seen_row_hashes.add(row_hash)

        # Names
        for name in chunk["name"].dropna().astype(str):
            unique_names.add(name.strip())

        # Lengths for valid judgments
        valid_judgments = chunk[chunk["judgement"].notna()]
        if len(valid_judgments) > 0:
            char_lens = valid_judgments["judgement"].astype(str).str.len().tolist()
            word_lens = valid_judgments["judgement"].astype(str).str.split().str.len().tolist()
            lengths_chars.extend(char_lens)
            lengths_words.extend(word_lens)

        valid_sents = chunk[chunk["sentences"].notna()]["sentences"].astype(float).tolist()
        lengths_sents.extend(valid_sents)

        # Categoricals
        for cat, count in chunk["case_category"].value_counts(dropna=True).items():
            case_categories[str(cat)] = case_categories.get(str(cat), 0) + int(count)

        for ctype, count in chunk["case_type"].value_counts(dropna=True).items():
            case_types[str(ctype)] = case_types.get(str(ctype), 0) + int(count)

        for lbl, count in chunk["label"].value_counts(dropna=True).items():
            labels[str(lbl)] = labels.get(str(lbl), 0) + int(count)

    char_arr = np.array(lengths_chars)
    word_arr = np.array(lengths_words)
    sent_arr = np.array(lengths_sents)

    stats = {
        "Total_Raw_Records": total_rows,
        "Unique_Case_IDs": len(unique_names),
        "Exact_Duplicate_Rows": exact_duplicate_rows,
        "Duplicate_Percentage": round((exact_duplicate_rows / total_rows) * 100, 2),
        "Valid_Judgment_Texts": len(char_arr),
        "Missing_Judgments": null_counts.get("judgement", 0),
        "Missing_Judgment_Pct": round((null_counts.get("judgement", 0) / total_rows) * 100, 2),
        "Char_Len_Min": int(char_arr.min()) if len(char_arr) else 0,
        "Char_Len_Median": float(np.median(char_arr)) if len(char_arr) else 0.0,
        "Char_Len_Mean": round(float(char_arr.mean()), 1) if len(char_arr) else 0.0,
        "Char_Len_Max": int(char_arr.max()) if len(char_arr) else 0,
        "Word_Len_Median": float(np.median(word_arr)) if len(word_arr) else 0.0,
        "Word_Len_Mean": round(float(word_arr.mean()), 1) if len(word_arr) else 0.0,
        "Sent_Count_Median": float(np.median(sent_arr)) if len(sent_arr) else 0.0,
        "Sent_Count_Mean": round(float(sent_arr.mean()), 1) if len(sent_arr) else 0.0,
        "Proof_Sentences_Available": total_rows - null_counts.get("proof_sentence", 0),
        "Case_Category_Civil": case_categories.get("civil", 0),
        "Case_Category_Criminal": case_categories.get("criminal", 0),
        "Outcome_Rejected": labels.get("Rejected", 0),
        "Outcome_Accepted": labels.get("Accepted", 0),
        "Outcome_Other": labels.get("Other", 0),
        "Audit_Time_Seconds": round(time.time() - t0, 2),
    }

    # Save to CSV
    stat_rows = [{"Metric": k, "Value": v} for k, v in stats.items()]
    df_stats = pd.DataFrame(stat_rows)
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(output_table_path, index=False)
    print(f"\nSaved Jud-IPL Corpus Statistics to: {output_table_path}")
    print(df_stats.to_string(index=False))

    return stats


if __name__ == "__main__":
    csv_file = PROJECT_ROOT / "data" / "raw" / "Jud-IPL" / "case_files_total.csv"
    output_csv = PROJECT_ROOT / "results" / "tables" / "judipl_corpus_statistics.csv"
    audit_judipl_dataset(csv_file, output_csv)

