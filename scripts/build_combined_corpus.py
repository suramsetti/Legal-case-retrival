"""
Phase 17: Build Combined Corpus (IL-TUR PCR + Jud-IPL).

Merges the IL-TUR candidate judgment pool with preprocessed Jud-IPL documents.
Deduplication prefers IL-TUR text when IDs overlap.

Outputs:
- data/processed/combined_corpus/documents.parquet
- data/processed/combined_corpus/legal_features.json
- results/tables/judipl_combined_corpus_stats.csv
"""

import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np


ILTUR_SPLITS = ["train_candidates", "dev_candidates", "test_candidates"]
ILTUR_BASE = PROJECT_ROOT / "data" / "processed" / "IL-TUR-PCR"
JUDIPL_DIR = PROJECT_ROOT / "data" / "processed" / "judipl"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "combined_corpus"
STATS_PATH = PROJECT_ROOT / "results" / "tables" / "judipl_combined_corpus_stats.csv"


def load_iltur_candidates() -> pd.DataFrame:
    """Load all IL-TUR candidate documents from the three splits."""
    frames = []
    for split in ILTUR_SPLITS:
        split_dir = ILTUR_BASE / split
        # Try standard path patterns
        for candidate_file in [
            split_dir / "documents.parquet",
            split_dir / "candidates.parquet",
            ILTUR_BASE / f"{split}.parquet",
        ]:
            if candidate_file.exists():
                df = pd.read_parquet(candidate_file)
                df["_split"] = split
                frames.append(df)
                print(f"  Loaded IL-TUR {split}: {len(df):,} rows from {candidate_file.name}", flush=True)
                break
        else:
            print(f"  WARNING: No candidate file found for {split}", flush=True)

    if not frames:
        raise FileNotFoundError("No IL-TUR candidate files found. Check data/processed/IL-TUR-PCR/")

    df_all = pd.concat(frames, ignore_index=True)
    print(f"  IL-TUR total (raw concat): {len(df_all):,} rows", flush=True)
    return df_all


def normalize_iltur_docs(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize IL-TUR candidate dataframe to a standard schema."""
    # Identify ID column
    id_col = None
    for c in ["id", "case_id", "doc_id", "judgment_id"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        raise ValueError(f"Cannot find ID column in IL-TUR df. Columns: {df.columns.tolist()}")

    # Identify text column
    text_col = None
    for c in ["full_text", "text", "judgment_text", "judgement"]:
        if c in df.columns:
            text_col = c
            break

    std_df = pd.DataFrame()
    std_df["id"] = df[id_col].astype(str).str.strip()
    std_df["full_text"] = df[text_col].fillna("") if text_col else ""
    std_df["source"] = "IL-TUR"

    # Copy through extra columns if they exist
    for col in ["char_length", "word_length", "sentence_count", "statute_count", "citation_count",
                "case_category", "case_type", "label"]:
        if col in df.columns:
            std_df[col] = df[col].values
        else:
            std_df[col] = None

    # Compute char_length if missing
    if std_df["char_length"].isna().all() and text_col:
        std_df["char_length"] = df[text_col].fillna("").str.len().values

    return std_df


def build_combined_corpus() -> pd.DataFrame:
    print("=" * 80, flush=True)
    print("PHASE 17: BUILD COMBINED CORPUS (IL-TUR + JUD-IPL)", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load IL-TUR ----
    print("\n[Step 1] Loading IL-TUR candidates...", flush=True)
    df_iltur_raw = load_iltur_candidates()
    df_iltur = normalize_iltur_docs(df_iltur_raw)
    # Deduplicate IL-TUR on ID (should already be unique per split but just in case)
    df_iltur = df_iltur.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    n_iltur = len(df_iltur)
    print(f"  Unique IL-TUR documents: {n_iltur:,}", flush=True)

    # ---- Load Jud-IPL ----
    print("\n[Step 2] Loading Jud-IPL preprocessed documents...", flush=True)
    judipl_docs_path = JUDIPL_DIR / "documents.parquet"
    if not judipl_docs_path.exists():
        raise FileNotFoundError(
            f"Jud-IPL documents.parquet not found at {judipl_docs_path}. "
            "Run scripts/preprocess_judipl.py first."
        )
    df_judipl = pd.read_parquet(judipl_docs_path)
    print(f"  Jud-IPL documents loaded: {len(df_judipl):,}", flush=True)

    # Standardize columns
    judipl_std = pd.DataFrame()
    judipl_std["id"] = df_judipl["id"].astype(str)
    judipl_std["full_text"] = df_judipl["full_text"].fillna("")
    judipl_std["source"] = "Jud-IPL"
    for col in ["char_length", "word_length", "sentence_count", "statute_count", "citation_count",
                "case_category", "case_type", "label"]:
        judipl_std[col] = df_judipl[col].values if col in df_judipl.columns else None
    n_judipl = len(judipl_std)

    # ---- Merge with deduplication (prefer IL-TUR when IDs overlap) ----
    print("\n[Step 3] Merging with deduplication (IL-TUR preferred on overlap)...", flush=True)
    iltur_ids = set(df_iltur["id"].tolist())
    judipl_new = judipl_std[~judipl_std["id"].isin(iltur_ids)].copy()
    overlap = judipl_std[judipl_std["id"].isin(iltur_ids)]
    n_overlap = len(overlap)
    n_new_judipl = len(judipl_new)

    print(f"  IL-TUR docs            : {n_iltur:,}", flush=True)
    print(f"  Jud-IPL total          : {n_judipl:,}", flush=True)
    print(f"  ID overlap (kept IL-TUR): {n_overlap:,}", flush=True)
    print(f"  New Jud-IPL only docs  : {n_new_judipl:,}", flush=True)

    df_combined = pd.concat([df_iltur, judipl_new], ignore_index=True)
    n_combined = len(df_combined)
    print(f"  Combined corpus size   : {n_combined:,}", flush=True)

    # ---- Load and merge legal features ----
    print("\n[Step 4] Merging legal features...", flush=True)
    combined_features: dict = {}

    judipl_feats_path = JUDIPL_DIR / "legal_features.json"
    if judipl_feats_path.exists():
        with open(judipl_feats_path) as f:
            judipl_feats = json.load(f)
        combined_features.update(judipl_feats)
        print(f"  Loaded {len(judipl_feats):,} Jud-IPL legal feature entries", flush=True)

    # Load IL-TUR legal features if available
    for feat_path in [
        ILTUR_BASE / "legal_features.json",
        ILTUR_BASE / "train_candidates" / "legal_features.json",
    ]:
        if feat_path.exists():
            with open(feat_path) as f:
                iltur_feats = json.load(f)
            combined_features.update(iltur_feats)  # IL-TUR overwrites Jud-IPL on overlap
            print(f"  Loaded {len(iltur_feats):,} IL-TUR legal feature entries from {feat_path.name}", flush=True)
            break

    # ---- Save outputs ----
    print("\n[Step 5] Saving combined corpus...", flush=True)
    combined_docs_path = OUT_DIR / "documents.parquet"
    combined_feats_path = OUT_DIR / "legal_features.json"

    df_combined.to_parquet(combined_docs_path, index=False)
    print(f"  Saved: {combined_docs_path}", flush=True)

    with open(combined_feats_path, "w") as f:
        json.dump(combined_features, f, separators=(",", ":"))
    print(f"  Saved: {combined_feats_path}", flush=True)

    # ---- Corpus statistics ----
    print("\n[Step 6] Computing corpus statistics...", flush=True)
    stats_rows = []

    def corpus_stats(df: pd.DataFrame, name: str) -> dict:
        char_lens = df["char_length"].dropna()
        return {
            "corpus": name,
            "total_documents": len(df),
            "mean_char_length": round(float(char_lens.mean()), 1) if len(char_lens) > 0 else 0,
            "median_char_length": round(float(char_lens.median()), 1) if len(char_lens) > 0 else 0,
            "sources": df["source"].value_counts().to_dict() if "source" in df.columns else {},
        }

    stats_rows.append({
        "corpus": "IL-TUR only",
        "total_documents": n_iltur,
        "mean_char_length": round(float(df_iltur["char_length"].dropna().mean()), 1) if "char_length" in df_iltur.columns else 0,
        "median_char_length": round(float(df_iltur["char_length"].dropna().median()), 1) if "char_length" in df_iltur.columns else 0,
        "source_breakdown": "IL-TUR",
        "id_overlap_with_other": n_overlap,
        "new_unique_docs": n_iltur,
    })
    stats_rows.append({
        "corpus": "Jud-IPL only",
        "total_documents": n_judipl,
        "mean_char_length": round(float(judipl_std["char_length"].dropna().mean()), 1) if "char_length" in judipl_std.columns else 0,
        "median_char_length": round(float(judipl_std["char_length"].dropna().median()), 1) if "char_length" in judipl_std.columns else 0,
        "source_breakdown": "Jud-IPL",
        "id_overlap_with_other": n_overlap,
        "new_unique_docs": n_new_judipl,
    })
    stats_rows.append({
        "corpus": "Combined (IL-TUR + Jud-IPL)",
        "total_documents": n_combined,
        "mean_char_length": round(float(df_combined["char_length"].dropna().mean()), 1) if "char_length" in df_combined.columns else 0,
        "median_char_length": round(float(df_combined["char_length"].dropna().median()), 1) if "char_length" in df_combined.columns else 0,
        "source_breakdown": "IL-TUR + Jud-IPL",
        "id_overlap_with_other": n_overlap,
        "new_unique_docs": n_combined,
    })

    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(STATS_PATH, index=False)
    print(f"  Saved corpus stats: {STATS_PATH}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("Combined Corpus Build COMPLETE.", flush=True)
    print(f"  IL-TUR docs     : {n_iltur:,}", flush=True)
    print(f"  New Jud-IPL docs: {n_new_judipl:,}", flush=True)
    print(f"  Total combined  : {n_combined:,}", flush=True)
    print(f"  Time elapsed    : {time.time()-t0:.1f}s", flush=True)
    print("=" * 80, flush=True)
    return df_combined


if __name__ == "__main__":
    build_combined_corpus()

