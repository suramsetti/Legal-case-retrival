"""
Phase 18C: Jud-IPL Group-Aware Dataset Partitioning (Minimal)

Creates leakage-free partitions by keeping exact duplicates together.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

DOCS_PATH = PROJECT_ROOT / "data" / "processed" / "judipl" / "documents.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

RANDOM_SEED = 42


def main():
    print("=" * 80)
    print("PHASE 18C: JUD-IPL GROUP-AWARE DATASET PARTITIONING")
    print("=" * 80)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[0] Loading Jud-IPL data...")
    df = pd.read_parquet(DOCS_PATH)
    print(f"  Loaded {len(df):,} documents")

    # Identify duplicate groups
    print("\n[1] Identifying duplicate groups...")
    df['duplicate_group'] = df['id']  # Default: each document in its own group

    # Find exact duplicates
    dup_mask = df.duplicated(subset=['full_text'], keep=False)
    dup_df = df[dup_mask].copy()

    if len(dup_df) > 0:
        unique_texts = dup_df['full_text'].unique()
        text_to_group = {t: f"dup_group_{i}" for i, t in enumerate(unique_texts)}
        dup_df['duplicate_group'] = dup_df['full_text'].map(text_to_group)
        df.loc[dup_df.index, 'duplicate_group'] = dup_df['duplicate_group']
        print(f"  Exact duplicate groups: {len(unique_texts)}")
        print(f"  Documents in groups: {len(dup_df)}")
    else:
        print(f"  No exact duplicates found")

    # Helper function to split with group awareness
    def group_aware_split(df_subset, task_name):
        print(f"\n[2] Splitting {task_name}...")

        # Initialize partition
        df_subset = df_subset.copy()
        df_subset['partition'] = 'unassigned'

        # Get unique groups
        groups = df_subset['duplicate_group'].unique()
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(groups)

        # Assign groups to partitions (70/15/15)
        n_groups = len(groups)
        n_train = int(n_groups * 0.7)
        n_val = int(n_groups * 0.15)

        train_groups = groups[:n_train]
        val_groups = groups[n_train:n_train + n_val]
        test_groups = groups[n_train + n_val:]

        df_subset.loc[df_subset['duplicate_group'].isin(train_groups), 'partition'] = 'train'
        df_subset.loc[df_subset['duplicate_group'].isin(val_groups), 'partition'] = 'val'
        df_subset.loc[df_subset['duplicate_group'].isin(test_groups), 'partition'] = 'test'

        # Statistics
        train_count = (df_subset['partition'] == 'train').sum()
        val_count = (df_subset['partition'] == 'val').sum()
        test_count = (df_subset['partition'] == 'test').sum()

        print(f"  Train: {train_count:,} ({train_count/len(df_subset)*100:.2f}%)")
        print(f"  Val: {val_count:,} ({val_count/len(df_subset)*100:.2f}%)")
        print(f"  Test: {test_count:,} ({test_count/len(df_subset)*100:.2f}%)")

        # Label distribution
        print(f"  Label distribution:")
        for partition in ['train', 'val', 'test']:
            part_df = df_subset[df_subset['partition'] == partition]
            print(f"    {partition}:")
            for label in sorted(df_subset['label'].unique()):
                count = (part_df['label'] == label).sum()
                print(f"      {label}: {count:,} ({count/len(part_df)*100:.2f}%)")

        return df_subset

    # Binary task
    print("\n" + "=" * 80)
    print("BINARY TASK: Accepted vs Rejected")
    print("=" * 80)
    binary_df = df[df['label'].isin(['Accepted', 'Rejected'])].copy()
    binary_df = group_aware_split(binary_df, 'binary')

    binary_output_dir = OUTPUT_DIR / "binary"
    binary_output_dir.mkdir(parents=True, exist_ok=True)

    for partition in ['train', 'val', 'test']:
        part_df = binary_df[binary_df['partition'] == partition].copy()
        output_path = binary_output_dir / f"{partition}.parquet"
        part_df.to_parquet(output_path, index=False)
        print(f"  Saved {partition}: {len(part_df):,} documents")

    # Three-class task
    print("\n" + "=" * 80)
    print("THREE-CLASS TASK: Accepted vs Rejected vs Other")
    print("=" * 80)
    three_class_df = df[df['label'].isin(['Accepted', 'Rejected', 'Other'])].copy()
    three_class_df = group_aware_split(three_class_df, 'three_class')

    three_class_output_dir = OUTPUT_DIR / "three_class"
    three_class_output_dir.mkdir(parents=True, exist_ok=True)

    for partition in ['train', 'val', 'test']:
        part_df = three_class_df[three_class_df['partition'] == partition].copy()
        output_path = three_class_output_dir / f"{partition}.parquet"
        part_df.to_parquet(output_path, index=False)
        print(f"  Saved {partition}: {len(part_df):,} documents")

    # Generate statistics
    print("\n[3] Generating statistics tables...")

    stats = []
    for partition in ['train', 'val', 'test']:
        part_df = binary_df[binary_df['partition'] == partition]
        stats.append({
            'task': 'binary',
            'partition': partition,
            'total_documents': len(part_df),
            'accepted': (part_df['label'] == 'Accepted').sum(),
            'rejected': (part_df['label'] == 'Rejected').sum(),
            'groups': part_df['duplicate_group'].nunique()
        })

    for partition in ['train', 'val', 'test']:
        part_df = three_class_df[three_class_df['partition'] == partition]
        stats.append({
            'task': 'three_class',
            'partition': partition,
            'total_documents': len(part_df),
            'accepted': (part_df['label'] == 'Accepted').sum(),
            'rejected': (part_df['label'] == 'Rejected').sum(),
            'other': (part_df['label'] == 'Other').sum(),
            'groups': part_df['duplicate_group'].nunique()
        })

    stats_df = pd.DataFrame(stats)
    stats_path = TABLES_DIR / "phase18c_partition_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved: {stats_path}")

    # Label distribution
    dist = []
    for partition in ['train', 'val', 'test']:
        part_df = binary_df[binary_df['partition'] == partition]
        total = len(part_df)
        for label in ['Accepted', 'Rejected']:
            count = (part_df['label'] == label).sum()
            dist.append({
                'task': 'binary',
                'partition': partition,
                'label': label,
                'count': count,
                'percentage': count / total * 100 if total > 0 else 0
            })

    for partition in ['train', 'val', 'test']:
        part_df = three_class_df[three_class_df['partition'] == partition]
        total = len(part_df)
        for label in ['Accepted', 'Rejected', 'Other']:
            count = (part_df['label'] == label).sum()
            dist.append({
                'task': 'three_class',
                'partition': partition,
                'label': label,
                'count': count,
                'percentage': count / total * 100 if total > 0 else 0
            })

    dist_df = pd.DataFrame(dist)
    dist_path = TABLES_DIR / "phase18c_label_distribution.csv"
    dist_df.to_csv(dist_path, index=False)
    print(f"  Saved: {dist_path}")

    print("\n" + "=" * 80)
    print("Group-aware partitioning completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
