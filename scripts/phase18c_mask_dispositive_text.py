"""
Phase 18C: Dispositive Holding Masking Pipeline

Creates a leakage-aware text preprocessing pipeline that:
- Removes the exact proof_sentence span from model input
- Identifies and masks the concluding ruling paragraph
- Detects ruling phrases (appeal allowed, appeal dismissed, etc.)
- Preserves relevant facts, legal arguments and reasoning
- Records masked character spans and masking reasons
- Preserves original text separately

The target label is never used to modify the input text.
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


def mask_proof_sentence(text: str, proof_sentence: str) -> Tuple[str, List[Dict]]:
    """
    Remove the exact proof_sentence span from text.

    Returns:
        masked_text: Text with proof_sentence replaced with [MASKED_HOLDING]
        masking_info: List of dicts with masking details
    """
    if pd.isna(text) or text == "" or pd.isna(proof_sentence) or proof_sentence == "":
        return text, []

    masking_info = []
    masked_text = text

    # Find and replace exact proof_sentence
    if proof_sentence in text:
        start_idx = text.find(proof_sentence)
        end_idx = start_idx + len(proof_sentence)
        masked_text = text.replace(proof_sentence, "[MASKED_HOLDING]", 1)

        masking_info.append({
            'reason': 'proof_sentence_removal',
            'start': start_idx,
            'end': end_idx,
            'original_text': proof_sentence,
            'masked_text': '[MASKED_HOLDING]'
        })

    return masked_text, masking_info


def detect_ruling_paragraphs(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect paragraphs containing ruling phrases.

    Returns:
        List of (start_idx, end_idx, reason) tuples
    """
    if pd.isna(text) or text == "":
        return []

    # Split into paragraphs
    paragraphs = text.split('\n')
    ruling_paragraphs = []

    current_pos = 0
    for para in paragraphs:
        if para.strip():
            # Check if paragraph contains ruling phrases
            for pattern in RULING_PHRASES:
                if re.search(pattern, para, re.IGNORECASE):
                    ruling_paragraphs.append((current_pos, current_pos + len(para), 'ruling_phrase'))
                    break
        current_pos += len(para) + 1  # +1 for newline

    return ruling_paragraphs


def mask_ruling_paragraphs(text: str, ruling_paragraphs: List[Tuple[int, int, str]]) -> Tuple[str, List[Dict]]:
    """
    Mask paragraphs containing ruling phrases.

    Returns:
        masked_text: Text with ruling paragraphs masked
        masking_info: List of dicts with masking details
    """
    if not ruling_paragraphs:
        return text, []

    masking_info = []
    # Sort by start position (reverse order to preserve indices)
    ruling_paragraphs_sorted = sorted(ruling_paragraphs, key=lambda x: x[0], reverse=True)

    masked_text = text
    for start, end, reason in ruling_paragraphs_sorted:
        original_text = text[start:end]
        masked_text = masked_text[:start] + "[MASKED_RULING]" + masked_text[end:]

        masking_info.append({
            'reason': reason,
            'start': start,
            'end': end,
            'original_text': original_text[:100] + '...' if len(original_text) > 100 else original_text,
            'masked_text': '[MASKED_RULING]'
        })

    return masked_text, masking_info


def apply_masking_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full masking pipeline to all documents.

    Pipeline:
    1. Remove exact proof_sentence
    2. Detect and mask ruling paragraphs
    3. Record all masking operations
    4. Preserve original text
    """
    print("[1] Applying dispositive holding masking pipeline...")

    df = df.copy()
    df['masked_text'] = df['full_text']
    df['masking_info'] = [[] for _ in range(len(df))]

    total_masked_proof = 0
    total_masked_ruling = 0

    for idx, row in df.iterrows():
        text = row['full_text']
        proof_sentence = row['proof_sentence']

        # Step 1: Remove proof_sentence
        masked_text, proof_masking = mask_proof_sentence(text, proof_sentence)
        if proof_masking:
            total_masked_proof += 1

        # Step 2: Detect and mask ruling paragraphs
        ruling_paragraphs = detect_ruling_paragraphs(masked_text)
        masked_text, ruling_masking = mask_ruling_paragraphs(masked_text, ruling_paragraphs)
        if ruling_masking:
            total_masked_ruling += 1

        # Combine masking info
        all_masking = proof_masking + ruling_masking

        df.at[idx, 'masked_text'] = masked_text
        df.at[idx, 'masking_info'] = all_masking

        if idx % 5000 == 0:
            print(f"  Processed {idx:,}/{len(df):,} documents...")

    print(f"  Total documents with proof_sentence masked: {total_masked_proof:,}")
    print(f"  Total documents with ruling paragraphs masked: {total_masked_ruling:,}")

    return df


def generate_masking_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate statistics on masking quality."""
    print("\n[2] Generating masking quality report...")

    quality_stats = []

    for idx, row in df.iterrows():
        masking_info = row['masking_info']
        original_length = len(row['full_text'])
        masked_length = len(row['masked_text'])

        proof_masked = any(m['reason'] == 'proof_sentence_removal' for m in masking_info)
        ruling_masked = any(m['reason'] == 'ruling_phrase' for m in masking_info)

        quality_stats.append({
            'id': row['id'],
            'original_length': original_length,
            'masked_length': masked_length,
            'masked_characters': original_length - masked_length,
            'masking_percentage': (original_length - masked_length) / original_length * 100 if original_length > 0 else 0,
            'proof_sentence_masked': proof_masked,
            'ruling_paragraph_masked': ruling_masked,
            'total_masking_operations': len(masking_info)
        })

    quality_df = pd.DataFrame(quality_stats)

    # Summary statistics
    print(f"  Average masking percentage: {quality_df['masking_percentage'].mean():.2f}%")
    print(f"  Max masking percentage: {quality_df['masking_percentage'].max():.2f}%")
    print(f"  Documents with proof_sentence masked: {quality_df['proof_sentence_masked'].sum():,}")
    print(f"  Documents with ruling paragraphs masked: {quality_df['ruling_paragraph_masked'].sum():,}")

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
            'masking_operations': len(row['masking_info'])
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
        f.write("2. **Ruling Paragraph Detection:** Paragraphs containing ruling phrases replaced with [MASKED_RULING]\n")
        f.write("3. **Original Text Preservation:** Original full_text preserved separately\n\n")

        f.write("## Binary Task Samples\n\n")
        for i, sample in enumerate(binary_samples, 1):
            f.write(f"### Sample {i}: {sample['id']} (Label: {sample['label']})\n\n")
            f.write(f"**Proof Sentence:** {sample['proof_sentence']}\n\n")
            f.write(f"**Original Text:** {sample['original_text']}\n\n")
            f.write(f"**Masked Text:** {sample['masked_text']}\n\n")
            f.write(f"**Masking Operations:** {sample['masking_operations']}\n\n")

        f.write("## Three-Class Task Samples\n\n")
        for i, sample in enumerate(three_class_samples, 1):
            f.write(f"### Sample {i}: {sample['id']} (Label: {sample['label']})\n\n")
            f.write(f"**Proof Sentence:** {sample['proof_sentence']}\n\n")
            f.write(f"**Original Text:** {sample['original_text']}\n\n")
            f.write(f"**Masked Text:** {sample['masked_text']}\n\n")
            f.write(f"**Masking Operations:** {sample['masking_operations']}\n\n")

    print(f"  Saved inspection report: {report_path}")

    print("\n" + "=" * 80)
    print("Dispositive holding masking completed successfully!")
    print("=" * 80)

    return binary_df, three_class_df, combined_quality


if __name__ == "__main__":
    main()
