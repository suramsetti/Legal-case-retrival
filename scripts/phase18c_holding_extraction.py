"""
Phase 18C: Holding Extraction Dataset Preparation

Prepares a separate dataset for extracting the dispositive holding sentence.
- Input: judgment body with the holding sentence removed or masked
- Target: original proof_sentence
- Preserve sentence-level source positions
- Prevent duplicate-group leakage across partitions
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

RANDOM_SEED = 42


def main():
    print("=" * 80)
    print("PHASE 18C: HOLDING EXTRACTION DATASET PREPARATION")
    print("=" * 80)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the partitioned datasets
    print("\n[0] Loading partitioned datasets...")
    binary_train = pd.read_parquet(OUTPUT_DIR / "binary" / "train.parquet")
    binary_val = pd.read_parquet(OUTPUT_DIR / "binary" / "val.parquet")
    binary_test = pd.read_parquet(OUTPUT_DIR / "binary" / "test.parquet")
    binary_df = pd.concat([binary_train, binary_val, binary_test], ignore_index=True)

    print(f"  Loaded binary dataset: {len(binary_df):,} documents")

    # Prepare holding extraction dataset
    print("\n[1] Preparing holding extraction dataset...")

    # Use original text as input (proof_sentence will be masked later)
    holding_df = binary_df.copy()
    holding_df['input_text'] = holding_df['full_text']
    holding_df['target_sentence'] = holding_df['proof_sentence']

    # Keep only relevant columns
    holding_df = holding_df[[
        'id', 'raw_id', 'label', 'partition', 'duplicate_group',
        'input_text', 'target_sentence', 'full_text', 'proof_sentence'
    ]]

    print(f"  Prepared {len(holding_df):,} documents for holding extraction")

    # Save holding extraction dataset
    print("\n[2] Saving holding extraction dataset...")

    output_dir = OUTPUT_DIR / "holding_extraction"
    output_dir.mkdir(parents=True, exist_ok=True)

    for partition in ['train', 'val', 'test']:
        part_df = holding_df[holding_df['partition'] == partition].copy()
        output_path = output_dir / f"{partition}.parquet"
        part_df.to_parquet(output_path, index=False)
        print(f"  Saved {partition}: {len(part_df):,} documents")

    # Save statistics
    stats = []
    for partition in ['train', 'val', 'test']:
        part_df = holding_df[holding_df['partition'] == partition]
        stats.append({
            'partition': partition,
            'total_documents': len(part_df),
            'accepted': (part_df['label'] == 'Accepted').sum(),
            'rejected': (part_df['label'] == 'Rejected').sum(),
            'groups': part_df['duplicate_group'].nunique()
        })

    stats_df = pd.DataFrame(stats)
    stats_path = TABLES_DIR / "phase18c_holding_extraction_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved statistics: {stats_path}")

    # Generate report
    report_path = REPORTS_DIR / "phase18c_holding_extraction_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 18C: Holding Extraction Dataset Preparation\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Dataset Description\n\n")
        f.write("The holding extraction task is a secondary task complementary to outcome prediction.\n\n")
        f.write("- **Input:** Judgment text (proof_sentence will be masked during training)\n")
        f.write("- **Target:** Original proof_sentence (dispositive holding)\n")
        f.write("- **Purpose:** Extract the dispositive holding sentence from the judgment\n\n")

        f.write("## Partition Statistics\n\n")
        f.write("| Partition | Total | Accepted | Rejected | Groups |\n")
        f.write("|-----------|-------|----------|----------|--------|\n")
        for _, row in stats_df.iterrows():
            f.write(f"| {row['partition']} | {row['total_documents']:,} | {row['accepted']:,} | {row['rejected']:,} | {row['groups']:,} |\n")
        f.write("\n")

        f.write("## Notes\n\n")
        f.write("- Uses the same group-aware partitions as the binary outcome prediction task\n")
        f.write("- Prevents duplicate-group leakage across partitions\n")
        f.write("- Input text has proof_sentence available for masking during training\n")
        f.write("- Target is the original proof_sentence for supervised training\n\n")

    print(f"  Saved report: {report_path}")

    print("\n" + "=" * 80)
    print("Holding extraction dataset preparation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
