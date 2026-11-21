"""
Phase 18C: Dispositive Holding Masking Pipeline (Optimized)

Creates a leakage-aware text preprocessing pipeline.
Optimized for performance on large datasets.
"""

import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

DOCS_PATH = PROJECT_ROOT / "data" / "processed" / "judipl" / "documents.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"

# Ruling phrases to detect (label-independent)
RULING_PHRASES = [
    r'\bappeal (?:is )?(?:allowed|allowed with costs|dismissed|dismissed with costs)\b',
    r'\bpetition (?:is )?(?:allowed|allowed with costs|dismissed|dismissed with costs)\b',
    r'\b(?:writ|review) (?:is )?(?:allowed|dismissed)\b',
    r'\b(?:the )?(?:appeal|petition) (?:succeeds|fails)\b',
    r'\b(?:we )?(?:allow|dismiss) (?:the )?(?:appeal|petition)\b',
    r'\bjudgment (?:is )?(?:affirmed|set aside|modified)\b',
    r'\b(?:conviction|acquittal) (?:is )?(?:upheld|set aside)\b',
    r'\bsentence (?:is )?(?:upheld|set aside|enhanced|reduced)\b',
]

# Compile regex pattern for performance
RULING_PATTERN = re.compile('|'.join(RULING_PHRASES), re.IGNORECASE)


def mask_proof_sentence(text: str, proof_sentence: str) -> Tuple[str, bool]:
    """
    Remove the exact proof_sentence span from text.

    Returns:
        masked_text: Text with proof_sentence replaced with [MASKED_HOLDING]
        was_masked: Whether masking occurred
    """
    if pd.isna(text) or text == "" or pd.isna(proof_sentence) or proof_sentence == "":
        return text, False

    # Find and replace exact proof_sentence
    if proof_sentence in text:
        masked_text = text.replace(proof_sentence, "[MASKED_HOLDING]", 1)
        return masked_text, True

    return text, False


def has_ruling_phrase(text: str) -> bool:
    """
    Check if text contains any ruling phrases.
    """
    if pd.isna(text) or text == "":
        return False
    return bool(RULING_PATTERN.search(text))


def apply_masking_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full masking pipeline to all documents.

    Pipeline:
    1. Remove exact proof_sentence
    2. Check for ruling phrases (flag only, don't mask to preserve context)
    3. Record masking statistics
    """
    print("[1] Applying dispositive holding masking pipeline...")

    df = df.copy()

    # Vectorized proof sentence masking
    print("  Masking proof sentences...")
    df['masked_text'] = df.apply(
        lambda row: mask_proof_sentence(row['full_text'], row['proof_sentence'])[0],
        axis=1
    )
    df['proof_masked'] = df.apply(
        lambda row: mask_proof_sentence(row['full_text'], row['proof_sentence'])[1],
        axis=1
    )

    # Check for ruling phrases
    print("  Detecting ruling phrases...")
    df['has_ruling_phrase'] = df['masked_text'].apply(has_ruling_phrase)

    # Statistics
    total_masked_proof = df['proof_masked'].sum()
    total_has_ruling = df['has_ruling_phrase'].sum()

    print(f"  Total documents with proof_sentence masked: {total_masked_proof:,}")
    print(f"  Total documents with ruling phrases detected: {total_has_ruling:,}")

    return df


def generate_masking_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate statistics on masking quality."""
    print("\n[2] Generating masking quality report...")

    quality_stats = []

    for idx, row in df.iterrows():
        original_length = len(row['full_text'])
        masked_length = len(row['masked_text'])

        quality_stats.append({
            'id': row['id'],
            'original_length': original_length,
            'masked_length': masked_length,
            'masked_characters': original_length - masked_length,
            'masking_percentage': (original_length - masked_length) / original_length * 100 if original_length > 0 else 0,
            'proof_sentence_masked': row['proof_masked'],
            'has_ruling_phrase': row['has_ruling_phrase']
        })

    quality_df = pd.DataFrame(quality_stats)

    # Summary statistics
    print(f"  Average masking percentage: {quality_df['masking_percentage'].mean():.2f}%")
    print(f"  Max masking percentage: {quality_df['masking_percentage'].max():.2f}%")
    print(f"  Documents with proof_sentence masked: {quality_df['proof_sentence_masked'].sum():,}")
    print(f"  Documents with ruling phrases detected: {quality_df['has_ruling_phrase'].sum():,}")

    return quality_df


def save_masked_datasets(binary_df, three_class_df):
    """Save masked datasets."""
    print("\n[3] Saving masked datasets...")

    # Save binary partitions with masking
    for partition in ['train', 'val', 'test']:
        part_df = binary_df[binary_df['partition'] == partition].copy()
        output_path = OUTPUT_DIR / "binary" / f"{partition}_masked.parquet"
        part_df.to_parquet(output_path, index=False)
        print(f"  Saved binary {partition}_masked: {len(part_df):,} documents")

    # Save three-class partitions with masking
    for partition in ['train', 'val', 'test']:
        part_df = three_class_df[three_class_df['partition'] == partition].copy()
        output_path = OUTPUT_DIR / "three_class" / f"{partition}_masked.parquet"
        part_df.to_parquet(output_path, index=False)
        print(f"  Saved three_class {partition}_masked: {len(part_df):,} documents")


def sample_masked_documents(df: pd.DataFrame, n_samples: int = 10) -> List[Dict]:
    """Sample documents for manual inspection."""
    print(f"\n[4] Sampling {n_samples} documents for manual inspection...")

    samples = []
    sampled_indices = np.random.choice(df.index, size=min(n_samples, len(df)), replace=False)

    for idx in sampled_indices:
        row = df.iloc[idx]
        samples.append({
            'id': row['id'],
            'label': row['label'],
            'proof_sentence': row['proof_sentence'][:200] + '...' if len(row['proof_sentence']) > 200 else row['proof_sentence'],
            'original_text': row['full_text'][:300] + '...' if len(row['full_text']) > 300 else row['full_text'],
            'masked_text': row['masked_text'][:300] + '...' if len(row['masked_text']) > 300 else row['masked_text'],
            'proof_masked': row['proof_masked'],
            'has_ruling_phrase': row['has_ruling_phrase']
        })

    return samples


def main():
    print("=" * 80)
    print("PHASE 18C: DISPOSITIVE HOLDING MASKING PIPELINE")
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

    # Apply masking to binary dataset
    print("\n" + "=" * 80)
    print("MASKING BINARY DATASET")
    print("=" * 80)
    binary_df = apply_masking_pipeline(binary_df)
    binary_quality = generate_masking_quality_report(binary_df)

    # Apply masking to three-class dataset
    print("\n" + "=" * 80)
    print("MASKING THREE-CLASS DATASET")
    print("=" * 80)
    three_class_df = apply_masking_pipeline(three_class_df)
    three_class_quality = generate_masking_quality_report(three_class_df)

    # Save masked datasets
    save_masked_datasets(binary_df, three_class_df)

    # Save quality statistics
    quality_path = TABLES_DIR / "phase18c_masking_quality.csv"
    combined_quality = pd.concat([
        binary_quality.assign(task='binary'),
        three_class_quality.assign(task='three_class')
    ], ignore_index=True)
    combined_quality.to_csv(quality_path, index=False)
    print(f"\n  Saved masking quality: {quality_path}")

    # Sample for manual inspection
    np.random.seed(42)
    binary_samples = sample_masked_documents(binary_df, 5)
    three_class_samples = sample_masked_documents(three_class_df, 5)

    # Generate inspection report
    report_path = REPORTS_DIR / "phase18c_masking_inspection_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 18C: Dispositive Holding Masking Inspection Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Masking Pipeline\n\n")
        f.write("1. **Proof Sentence Removal:** Exact proof_sentence span replaced with [MASKED_HOLDING]\n")
        f.write("2. **Ruling Phrase Detection:** Detected but not masked (to preserve context)\n")
        f.write("3. **Original Text Preservation:** Original full_text preserved separately\n\n")

        f.write("## Binary Task Samples\n\n")
        for i, sample in enumerate(binary_samples, 1):
            f.write(f"### Sample {i}: {sample['id']} (Label: {sample['label']})\n\n")
            f.write(f"**Proof Sentence:** {sample['proof_sentence']}\n\n")
            f.write(f"**Original Text:** {sample['original_text']}\n\n")
            f.write(f"**Masked Text:** {sample['masked_text']}\n\n")
            f.write(f"**Proof Masked:** {sample['proof_masked']}\n")
            f.write(f"**Has Ruling Phrase:** {sample['has_ruling_phrase']}\n\n")

        f.write("## Three-Class Task Samples\n\n")
        for i, sample in enumerate(three_class_samples, 1):
            f.write(f"### Sample {i}: {sample['id']} (Label: {sample['label']})\n\n")
            f.write(f"**Proof Sentence:** {sample['proof_sentence']}\n\n")
            f.write(f"**Original Text:** {sample['original_text']}\n\n")
            f.write(f"**Masked Text:** {sample['masked_text']}\n\n")
            f.write(f"**Proof Masked:** {sample['proof_masked']}\n")
            f.write(f"**Has Ruling Phrase:** {sample['has_ruling_phrase']}\n\n")

    print(f"  Saved inspection report: {report_path}")

    print("\n" + "=" * 80)
    print("Dispositive holding masking completed successfully!")
    print("=" * 80)

    return binary_df, three_class_df, combined_quality


if __name__ == "__main__":
    main()
