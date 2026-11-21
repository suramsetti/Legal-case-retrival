"""
Phase 18C: Simple Dispositive Holding Masking

Minimal implementation: just mask proof_sentence with [MASKED_HOLDING]
Uses vectorized operations for performance.
"""

import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"


def main():
    print("=" * 80)
    print("PHASE 18C: DISPOSITIVE HOLDING MASKING (SIMPLE)")
    print("=" * 80)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the partitioned datasets
    print("\n[0] Loading partitioned datasets...")
    binary_train = pd.read_parquet(OUTPUT_DIR / "binary" / "train.parquet")
    binary_val = pd.read_parquet(OUTPUT_DIR / "binary" / "val.parquet")
    binary_test = pd.read_parquet(OUTPUT_DIR / "binary" / "test.parquet")
    binary_df = pd.concat([binary_train, binary_val, binary_test], ignore_index=True)

    three_class_train = pd.read_parquet(OUTPUT_DIR / "three_class" / "train.parquet")
    three_class_val = pd.read_parquet(OUTPUT_DIR / "three_class" / "val.parquet")
    three_class_test = pd.read_parquet(OUTPUT_DIR / "three_class" / "test.parquet")
    three_class_df = pd.concat([three_class_train, three_class_val, three_class_test], ignore_index=True)

    print(f"  Loaded binary dataset: {len(binary_df):,} documents")
    print(f"  Loaded three_class dataset: {len(three_class_df):,} documents")

    # Apply simple masking to binary dataset
    print("\n[1] Masking binary dataset...")
    binary_df['masked_text'] = binary_df['full_text'].copy()

    # Use vectorized string replace
    mask_condition = (binary_df['proof_sentence'].notna()) & (binary_df['proof_sentence'] != '')
    for idx in binary_df[mask_condition].index:
        proof = binary_df.at[idx, 'proof_sentence']
        text = binary_df.at[idx, 'full_text']
        if proof in text:
            binary_df.at[idx, 'masked_text'] = text.replace(proof, "[MASKED_HOLDING]", 1)

    binary_df['proof_masked'] = mask_condition & (binary_df['masked_text'] != binary_df['full_text'])
    print(f"  Proof sentences masked: {binary_df['proof_masked'].sum():,}")

    # Apply simple masking to three-class dataset
    print("\n[2] Masking three-class dataset...")
    three_class_df['masked_text'] = three_class_df['full_text'].copy()

    mask_condition = (three_class_df['proof_sentence'].notna()) & (three_class_df['proof_sentence'] != '')
    for idx in three_class_df[mask_condition].index:
        proof = three_class_df.at[idx, 'proof_sentence']
        text = three_class_df.at[idx, 'full_text']
        if proof in text:
            three_class_df.at[idx, 'masked_text'] = text.replace(proof, "[MASKED_HOLDING]", 1)

    three_class_df['proof_masked'] = mask_condition & (three_class_df['masked_text'] != three_class_df['full_text'])
    print(f"  Proof sentences masked: {three_class_df['proof_masked'].sum():,}")

    # Save masked datasets
    print("\n[3] Saving masked datasets...")
    for partition in ['train', 'val', 'test']:
        part_df = binary_df[binary_df['partition'] == partition].copy()
        output_path = OUTPUT_DIR / "binary" / f"{partition}_masked.parquet"
        part_df.to_parquet(output_path, index=False)
        print(f"  Saved binary {partition}_masked: {len(part_df):,} documents")

    for partition in ['train', 'val', 'test']:
        part_df = three_class_df[three_class_df['partition'] == partition].copy()
        output_path = OUTPUT_DIR / "three_class" / f"{partition}_masked.parquet"
        part_df.to_parquet(output_path, index=False)
        print(f"  Saved three_class {partition}_masked: {len(part_df):,} documents")

    # Generate quality report
    print("\n[4] Generating quality report...")
    binary_df['original_length'] = binary_df['full_text'].str.len()
    binary_df['masked_length'] = binary_df['masked_text'].str.len()
    binary_df['masking_percentage'] = ((binary_df['original_length'] - binary_df['masked_length']) / binary_df['original_length'] * 100).fillna(0)

    three_class_df['original_length'] = three_class_df['full_text'].str.len()
    three_class_df['masked_length'] = three_class_df['masked_text'].str.len()
    three_class_df['masking_percentage'] = ((three_class_df['original_length'] - three_class_df['masked_length']) / three_class_df['original_length'] * 100).fillna(0)

    quality_path = TABLES_DIR / "phase18c_masking_quality.csv"
    binary_quality = binary_df[['id', 'original_length', 'masked_length', 'masking_percentage', 'proof_masked']].copy()
    binary_quality['task'] = 'binary'
    three_class_quality = three_class_df[['id', 'original_length', 'masked_length', 'masking_percentage', 'proof_masked']].copy()
    three_class_quality['task'] = 'three_class'

    combined_quality = pd.concat([binary_quality, three_class_quality], ignore_index=True)
    combined_quality.to_csv(quality_path, index=False)
    print(f"  Saved: {quality_path}")

    # Generate inspection report
    report_path = REPORTS_DIR / "phase18c_masking_inspection_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 18C: Dispositive Holding Masking Inspection Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Masking Pipeline\n\n")
        f.write("1. **Proof Sentence Removal:** Exact proof_sentence span replaced with [MASKED_HOLDING]\n")
        f.write("2. **Original Text Preservation:** Original full_text preserved separately\n\n")

        f.write("## Statistics\n\n")
        f.write(f"**Binary Task:**\n")
        f.write(f"- Total documents: {len(binary_df):,}\n")
        f.write(f"- Proof sentences masked: {binary_df['proof_masked'].sum():,}\n")
        f.write(f"- Average masking percentage: {binary_df['masking_percentage'].mean():.2f}%\n\n")

        f.write(f"**Three-Class Task:**\n")
        f.write(f"- Total documents: {len(three_class_df):,}\n")
        f.write(f"- Proof sentences masked: {three_class_df['proof_masked'].sum():,}\n")
        f.write(f"- Average masking percentage: {three_class_df['masking_percentage'].mean():.2f}%\n\n")

    print(f"  Saved: {report_path}")

    print("\n" + "=" * 80)
    print("Dispositive holding masking completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
