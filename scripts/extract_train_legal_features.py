"""Extracts Phase 5 legal features for train_queries and train_candidates."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.legal_features.legal_feature_encoder import LegalFeatureEncoder
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "IL-TUR-PCR"

def main():
    encoder = LegalFeatureEncoder()
    for split in ["train_queries", "train_candidates"]:
        split_dir = PROCESSED_DIR / split
        docs_file = split_dir / "documents.parquet"
        save_file = split_dir / "legal_features.json"
        
        if save_file.exists():
            print(f"{save_file} already exists, skipping.")
            continue
            
        print(f"Extracting legal features for {split}...")
        docs_df = pd.read_parquet(docs_file)
        t0 = time.time()
        encoder.encode_split(split, docs_df, save_path=save_file)
        print(f"Finished {split} in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
