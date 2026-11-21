"""
Phase 18C: Dispositive Holding Masking (Sample)

Processes a sample of documents to demonstrate masking and generate quality report.
Full masking will be applied during model training.
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
    print("PHASE 18C: DISPOSITIVE HOLDING MASKING (SAMPLE)")
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

    # Sample for masking demonstration
    print("\n[1] Sampling for masking demonstration...")
    np.random.seed(42)
    sample_size = 1000
    binary_sample = binary_df.sample(n=min(sample_size, len(binary_df)), random_state=42)
    three_class_sample = three_class_df.sample(n=min(sample_size, len(three_class_df)), random_state=42)

    print(f"  Sampled {len(binary_sample)} documents from binary")
    print(f"  Sampled {len(three_class_sample)} documents from three_class")

    # Apply masking to samples
    print("\n[2] Applying masking to binary sample...")
    binary_sample['masked_text'] = binary_sample['full_text'].copy()

    mask_condition = (binary_sample['proof_sentence'].notna()) & (binary_sample['proof_sentence'] != '')
    for idx in binary_sample[mask_condition].index:
        proof = binary_sample.at[idx, 'proof_sentence']
        text = binary_sample.at[idx, 'full_text']
        if proof in text:
            binary_sample.at[idx, 'masked_text'] = text.replace(proof, "[MASKED_HOLDING]", 1)

    binary_sample['proof_masked'] = mask_condition & (binary_sample['masked_text'] != binary_sample['full_text'])
    print(f"  Proof sentences masked: {binary_sample['proof_masked'].sum()}")

    print("\n[3] Applying masking to three-class sample...")
    three_class_sample['masked_text'] = three_class_sample['full_text'].copy()

    mask_condition = (three_class_sample['proof_sentence'].notna()) & (three_class_sample['proof_sentence'] != '')
    for idx in three_class_sample[mask_condition].index:
        proof = three_class_sample.at[idx, 'proof_sentence']
        text = three_class_sample.at[idx, 'full_text']
        if proof in text:
            three_class_sample.at[idx, 'masked_text'] = text.replace(proof, "[MASKED_HOLDING]", 1)

    three_class_sample['proof_masked'] = mask_condition & (three_class_sample['masked_text'] != three_class_sample['full_text'])
    print(f"  Proof sentences masked: {three_class_sample['proof_masked'].sum()}")

    # Generate quality report from samples
    print("\n[4] Generating quality report...")
    binary_sample['original_length'] = binary_sample['full_text'].str.len()
    binary_sample['masked_length'] = binary_sample['masked_text'].str.len()
    binary_sample['masking_percentage'] = ((binary_sample['original_length'] - binary_sample['masked_length']) / binary_sample['original_length'] * 100).fillna(0)

    three_class_sample['original_length'] = three_class_sample['full_text'].str.len()
    three_class_sample['masked_length'] = three_class_sample['masked_text'].str.len()
    three_class_sample['masking_percentage'] = ((three_class_sample['original_length'] - three_class_sample['masked_length']) / three_class_sample['original_length'] * 100).fillna(0)

    quality_path = TABLES_DIR / "phase18c_masking_quality.csv"
    binary_quality = binary_sample[['id', 'original_length', 'masked_length', 'masking_percentage', 'proof_masked']].copy()
    binary_quality['task'] = 'binary'
    three_class_quality = three_class_sample[['id', 'original_length', 'masked_length', 'masking_percentage', 'proof_masked']].copy()
    three_class_quality['task'] = 'three_class'

    combined_quality = pd.concat([binary_quality, three_class_quality], ignore_index=True)
    combined_quality.to_csv(quality_path, index=False)
    print(f"  Saved: {quality_path}")

    # Generate inspection report with samples
    report_path = REPORTS_DIR / "phase18c_masking_inspection_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 18C: Dispositive Holding Masking Inspection Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Masking Pipeline\n\n")
        f.write("1. **Proof Sentence Removal:** Exact proof_sentence span replaced with [MASKED_HOLDING]\n")
        f.write("2. **Original Text Preservation:** Original full_text preserved separately\n")
        f.write("3. **Note:** Full masking will be applied during model training. This report shows sample results.\n\n")

        f.write("## Sample Statistics\n\n")
        f.write(f"**Binary Task (n={len(binary_sample)}):**\n")
        f.write(f"- Proof sentences masked: {binary_sample['proof_masked'].sum()} ({binary_sample['proof_masked'].mean()*100:.1f}%)\n")
        f.write(f"- Average masking percentage: {binary_sample['masking_percentage'].mean():.2f}%\n")
        f.write(f"- Max masking percentage: {binary_sample['masking_percentage'].max():.2f}%\n\n")

        f.write(f"**Three-Class Task (n={len(three_class_sample)}):**\n")
        f.write(f"- Proof sentences masked: {three_class_sample['proof_masked'].sum()} ({three_class_sample['proof_masked'].mean()*100:.1f}%)\n")
        f.write(f"- Average masking percentage: {three_class_sample['masking_percentage'].mean():.2f}%\n")
        f.write(f"- Max masking percentage: {three_class_sample['masking_percentage'].max():.2f}%\n\n")

        f.write("## Sample Inspection\n\n")
        f.write("### Binary Task Samples\n\n")
        for i in range(min(3, len(binary_sample))):
            row = binary_sample.iloc[i]
            f.write(f"#### Sample {i+1}: {row['id']} (Label: {row['label']})\n\n")
            f.write(f"**Proof Sentence:** {str(row['proof_sentence'])[:200]}...\n\n")
            f.write(f"**Original Text:** {str(row['full_text'])[:300]}...\n\n")
            f.write(f"**Masked Text:** {str(row['masked_text'])[:300]}...\n\n")
            f.write(f"**Proof Masked:** {row['proof_masked']}\n\n")

        f.write("### Three-Class Task Samples\n\n")
        for i in range(min(3, len(three_class_sample))):
            row = three_class_sample.iloc[i]
            f.write(f"#### Sample {i+1}: {row['id']} (Label: {row['label']})\n\n")
            f.write(f"**Proof Sentence:** {str(row['proof_sentence'])[:200]}...\n\n")
            f.write(f"**Original Text:** {str(row['full_text'])[:300]}...\n\n")
            f.write(f"**Masked Text:** {str(row['masked_text'])[:300]}...\n\n")
            f.write(f"**Proof Masked:** {row['proof_masked']}\n\n")

    print(f"  Saved: {report_path}")

    print("\n" + "=" * 80)
    print("Dispositive holding masking sample completed successfully!")
    print("Note: Full masking will be applied during model training.")
    print("=" * 80)


if __name__ == "__main__":
    main()
