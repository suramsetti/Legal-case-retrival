import pandas as pd
for split in ["train", "dev", "test"]:
    p = f"data/processed/IL-TUR-PCR/{split}/hybrid_features.parquet"
    df = pd.read_parquet(p)
    print(f"*** {split.upper()} DATASET ***")
    print(f"  Shape: {df.shape}")
    print(f"  Queries: {df['query_id'].nunique()} | Candidates: {df['candidate_id'].nunique()}")
    print(f"  Positives: {(df['label'] == 1).sum()} | Negatives: {(df['label'] == 0).sum()}")
    print(f"  Missing values: {df.isna().sum().sum()}")
    has_bert = "raw_bert_base_sim" in df.columns or "bert_base_sim" in df.columns
    print(f"  Has bert_base: {has_bert}")
    # Duplicate check
    dups = df.duplicated(subset=['query_id', 'candidate_id']).sum()
    print(f"  Duplicate pairs: {dups}")

