"""
Phase 17: Legal Document Preprocessing Pipeline for Jud-IPL.

Processes data/raw/Jud-IPL/case_files_total.csv:
1. Deduplication: Drops exact duplicate rows (preserving unique numeric IDs).
2. Quality filtering: Discards null and empty judgments.
3. ID Normalization: Standardizes numeric Kanoon IDs with zero-padding (matching IL-TUR convention).
4. Metadata Extraction: Parses bench, date, party names, and appeal types from case_info.
5. Legal Feature Extraction: Extracts statutes and citations using StatuteExtractor and CitationExtractor.
6. Section-Aware Chunking: Produces section chunks for a bounded sample.
7. Aligns gold proof_sentence (ratio decidendi / holding).

Outputs:
- data/processed/judipl/documents.parquet
- data/processed/judipl/chunks.parquet
- data/processed/judipl/legal_features.json
"""

import sys
import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure stdout is unbuffered for log visibility
sys.stdout = sys.stderr if False else __import__("io").TextIOWrapper(
    sys.stdout.buffer, line_buffering=True
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.legal_features.statute_extractor import StatuteExtractor
from src.legal_features.citation_extractor import CitationExtractor
from src.models.phase16_cross_encoder import Phase16SectionChunker


def parse_case_info(info_str: str) -> Dict[str, Any]:
    """Extracts structured metadata fields from raw case_info text."""
    if not isinstance(info_str, str) or not info_str.strip():
        return {"petitioner": "", "respondent": "", "date": "", "bench": "", "citation": ""}

    text = info_str.lower()

    # Date extraction (DD/MM/YYYY or DD-MM-YYYY)
    date_match = re.search(r"date\s*of\s*judgment\s*[:\s]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{4})", text)
    date_val = date_match.group(1) if date_match else ""

    # Bench extraction
    bench_match = re.search(r"bench\s*[:\s]*([a-z\s,\.\(\)]+?)(?:citation|act|headnote|judgment|$)", text)
    bench_val = bench_match.group(1).strip()[:100] if bench_match else ""

    # Citation extraction
    cite_match = re.search(r"citation\s*[:\s]*([a-z0-9\s,\.\(\)\[\]]+?)(?:act|headnote|judgment|$)", text)
    cite_val = cite_match.group(1).strip()[:100] if cite_match else ""

    # Petitioner vs Respondent
    pet_match = re.search(r"petitioner\s*[:\s]*([a-z0-9\s,\.\-]+?)\s+vs\.?", text)
    pet_val = pet_match.group(1).strip()[:100] if pet_match else ""

    resp_match = re.search(r"respondent\s*[:\s]*([a-z0-9\s,\.\-]+?)(?:date|bench|citation|$)", text)
    resp_val = resp_match.group(1).strip()[:100] if resp_match else ""

    return {
        "petitioner": pet_val,
        "respondent": resp_val,
        "date": date_val,
        "bench": bench_val,
        "citation": cite_val,
    }


def preprocess_judipl_corpus(
    csv_path: Path,
    output_dir: Path,
    chunk_size: int = 5000,
    max_chunk_samples: int = 10000,
    legal_feature_sample: int = 20000,
) -> pd.DataFrame:
    print("=" * 80, flush=True)
    print("PHASE 17: PREPROCESSING JUD-IPL LEGAL CORPUS", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    statute_extractor = StatuteExtractor()
    citation_extractor = CitationExtractor()
    section_chunker = Phase16SectionChunker()

    # ---- Step 1: Load and deduplicate ----
    print("\n[Step 1] Loading CSV (streaming)...", flush=True)
    chunks_list = []
    total_read = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        total_read += len(chunk)
        chunks_list.append(chunk)
        if total_read % 10000 == 0:
            print(f"  Read {total_read:,} rows so far...", flush=True)

    df_raw = pd.concat(chunks_list, ignore_index=True)
    print(f"  Total rows read: {len(df_raw):,}", flush=True)

    # Normalize IDs
    df_raw["_name_clean"] = df_raw["name"].astype(str).str.replace(".txt", "", regex=False).str.strip()
    df_raw = df_raw[df_raw["_name_clean"].str.isdigit()].copy()
    df_raw["_num_id"] = df_raw["_name_clean"].astype(int)

    # Deduplicate on numeric ID (keep first occurrence)
    before_dedup = len(df_raw)
    df_raw = df_raw.drop_duplicates(subset=["_num_id"], keep="first").reset_index(drop=True)
    duplicate_dropped = before_dedup - len(df_raw)
    print(f"  Duplicate rows dropped: {duplicate_dropped:,}", flush=True)

    # Filter null/empty judgments
    df_raw["_judgment_valid"] = df_raw["judgement"].apply(
        lambda x: isinstance(x, str) and len(x.strip()) > 50
    )
    null_dropped = (~df_raw["_judgment_valid"]).sum()
    df_valid = df_raw[df_raw["_judgment_valid"]].reset_index(drop=True)
    print(f"  Null/empty judgment rows dropped: {null_dropped:,}", flush=True)
    print(f"  Valid documents remaining: {len(df_valid):,}", flush=True)

    # ---- Step 2: Vectorized metadata ----
    print("\n[Step 2] Extracting metadata fields...", flush=True)
    std_ids = df_valid["_num_id"].apply(lambda x: f"{x:010d}")
    df_valid = df_valid.assign(id=std_ids)

    case_infos = df_valid["case_info"].fillna("").tolist()
    meta_list = [parse_case_info(ci) for ci in case_infos]
    meta_df = pd.DataFrame(meta_list)

    char_lengths = df_valid["judgement"].str.len()
    word_lengths = df_valid["judgement"].str.split().str.len()

    # Sentence count from column if available
    sent_col = pd.to_numeric(df_valid["sentences"], errors="coerce").fillna(0)
    # For rows with 0 or NaN, estimate from text
    need_sent_est = sent_col <= 0
    est_sent = df_valid.loc[need_sent_est, "judgement"].apply(
        lambda x: len(re.split(r"[.!?]+", x))
    )
    sent_col = sent_col.copy()
    sent_col.loc[need_sent_est] = est_sent
    sent_col = sent_col.astype(int)

    proof_sentences = df_valid["proof_sentence"].where(
        df_valid["proof_sentence"].notna(), ""
    ).astype(str).str.strip()

    print(f"  Metadata extraction complete for {len(df_valid):,} documents.", flush=True)

    # ---- Step 3: Legal feature extraction (sampled for speed) ----
    print(f"\n[Step 3] Legal feature extraction (first {legal_feature_sample:,} docs)...", flush=True)
    t3 = time.time()
    n_feat = min(legal_feature_sample, len(df_valid))
    statute_counts = []
    citation_counts = []
    statute_keys_list = []
    citation_keys_list = []

    for i in range(len(df_valid)):
        text = df_valid["judgement"].iloc[i]
        lead = text[:2000]  # Only first 2000 chars for speed
        if i < n_feat:
            try:
                statutes = statute_extractor.extract(lead)
                citations = citation_extractor.extract(lead)
                sk = sorted(set(s.normalized_key for s in statutes))
                ck = sorted(set(c.normalized_key for c in citations))
            except Exception:
                sk, ck = [], []
        else:
            sk, ck = [], []
        statute_keys_list.append(sk)
        citation_keys_list.append(ck)
        statute_counts.append(len(sk))
        citation_counts.append(len(ck))
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t3
            print(f"  Legal features: {i+1:,}/{len(df_valid):,} docs in {elapsed:.1f}s", flush=True)

    print(f"  Legal feature extraction done in {time.time() - t3:.1f}s", flush=True)

    # ---- Step 4: Build documents DataFrame ----
    print("\n[Step 4] Building documents DataFrame...", flush=True)
    df_docs = pd.DataFrame({
        "id": df_valid["id"].values,
        "raw_id": df_valid["_num_id"].values,
        "raw_name": df_valid["name"].astype(str).values,
        "case_category": df_valid["case_category"].fillna("").astype(str).values,
        "case_type": df_valid["case_type"].fillna("").astype(str).values,
        "label": df_valid["label"].fillna("").astype(str).values,
        "petitioner": meta_df["petitioner"].values,
        "respondent": meta_df["respondent"].values,
        "date_of_judgment": meta_df["date"].values,
        "bench": meta_df["bench"].values,
        "citation_meta": meta_df["citation"].values,
        "proof_sentence": proof_sentences.values,
        "char_length": char_lengths.values,
        "word_length": word_lengths.values,
        "sentence_count": sent_col.values,
        "statute_count": statute_counts,
        "citation_count": citation_counts,
        "full_text": df_valid["judgement"].values,
        "source": "Jud-IPL",
    })

    # ---- Step 5: Legal features JSON ----
    print("\n[Step 5] Building legal features JSON...", flush=True)
    legal_features_dict: Dict[str, Any] = {}
    for i, row in df_docs.iterrows():
        legal_features_dict[row["id"]] = {
            "statutes": statute_keys_list[i],
            "citations": citation_keys_list[i],
            "proof_sentence": row["proof_sentence"],
            "outcome_label": row["label"],
        }

    # ---- Step 6: Section-aware chunking (bounded sample) ----
    print(f"\n[Step 6] Section-aware chunking (first {max_chunk_samples:,} docs)...", flush=True)
    t6 = time.time()
    chunk_records: List[Dict[str, Any]] = []
    n_chunk = min(max_chunk_samples, len(df_docs))
    for i in range(n_chunk):
        row = df_docs.iloc[i]
        text = row["full_text"][:4000]
        try:
            sec_chunks = section_chunker.chunk_document(text)
        except Exception:
            sec_chunks = []
        for idx, c in enumerate(sec_chunks):
            chunk_records.append({
                "chunk_id": f"{row['id']}_chk_{idx}",
                "document_id": row["id"],
                "section_type": c.get("section_type", "unknown"),
                "rhetorical_weight": c.get("rhetorical_weight", 1.0),
                "text": c.get("text", ""),
            })
        if (i + 1) % 2000 == 0:
            print(f"  Chunking: {i+1:,}/{n_chunk:,} in {time.time()-t6:.1f}s", flush=True)

    df_chunks = pd.DataFrame(chunk_records)
    print(f"  Chunking done: {len(df_chunks):,} chunks from {n_chunk:,} docs in {time.time()-t6:.1f}s", flush=True)

    # ---- Step 7: Save outputs ----
    print("\n[Step 7] Saving outputs...", flush=True)
    docs_path = output_dir / "documents.parquet"
    chunks_path = output_dir / "chunks.parquet"
    features_path = output_dir / "legal_features.json"

    df_docs.to_parquet(docs_path, index=False)
    print(f"  Saved: {docs_path}", flush=True)

    df_chunks.to_parquet(chunks_path, index=False)
    print(f"  Saved: {chunks_path}", flush=True)

    with open(features_path, "w") as f:
        json.dump(legal_features_dict, f, separators=(",", ":"))
    print(f"  Saved: {features_path}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f"Preprocessing Summary:", flush=True)
    print(f"  Total rows read      : {total_read:,}", flush=True)
    print(f"  Duplicates dropped   : {duplicate_dropped:,}", flush=True)
    print(f"  Null texts dropped   : {null_dropped:,}", flush=True)
    print(f"  Final unique docs    : {len(df_docs):,}", flush=True)
    print(f"  Chunks generated     : {len(df_chunks):,}", flush=True)
    print(f"  Total time           : {time.time() - t0:.1f}s", flush=True)
    print("=" * 80, flush=True)
    print("Jud-IPL preprocessing COMPLETE.", flush=True)
    return df_docs


if __name__ == "__main__":
    csv_file = PROJECT_ROOT / "data" / "raw" / "Jud-IPL" / "case_files_total.csv"
    out_dir = PROJECT_ROOT / "data" / "processed" / "judipl"
    preprocess_judipl_corpus(csv_file, out_dir)
