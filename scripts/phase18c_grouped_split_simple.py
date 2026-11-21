"""
Phase 18C: Jud-IPL Group-Aware Dataset Partitioning (Simplified)

Creates leakage-free partitions by grouping documents based on:
- Exact full-text duplicates
- Shared appeal or connected-case identifiers

All members of the same duplicate group must remain in the same partition.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

DOCS_PATH = PROJECT_ROOT / "data" / "processed" / "judipl" / "documents.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

RANDOM_SEED = 42


def create_duplicate_groups(df):
    """
    Create duplicate groups based on:
    1. Exact full-text duplicates
    2. Shared numeric IDs
    """
    print("[1] Creating duplicate groups...")

    # Initialize group assignment
    df = df.copy()
    df['duplicate_group'] = df['id']  # Default: each document in its own group

    # 1. Exact full-text duplicates
    print("  Finding exact full-text duplicates...")
    dup_mask = df.duplicated(subset=['full_text'], keep=False)
    dup_df = df[dup_mask].copy()

    if len(dup_df) > 0:
        # Assign group IDs to duplicates
        unique_texts = dup_df['full_text'].unique()
        text_to_group = {t: f"hash_group_{i}" for i, t in enumerate(unique_texts)}
        dup_df['duplicate_group'] = dup_df['full_text'].map(text_to_group)
        df.loc[dup_df.index, 'duplicate_group'] = dup_df['duplicate_group']
        print(f"    Exact duplicate groups found: {len(unique_texts)}")
    else:
        print(f"    Exact duplicate groups found: 0")

    # 2. Shared numeric IDs
    print("  Checking for connected case identifiers...")
    df['numeric_id'] = df['raw_id'].astype(str).str.extract(r'(\d+)')[0]

    singleton_mask = df['duplicate_group'] == df['id']
    id_dup_mask = singleton_mask & df.duplicated(subset=['numeric_id'], keep=False) & (df['numeric_id'] != '')
    id_dup_df = df[id_dup_mask].copy()

    if len(id_dup_df) > 0:
        unique_ids = id_dup_df['numeric_id'].unique()
        current_id = len(dup_df['full_text'].unique()) if len(dup_df) > 0 else 0
        id_to_group = {h: f"id_group_{current_id + i}" for i, h in enumerate(unique_ids)}
        id_dup_df['duplicate_group'] = id_dup_df['numeric_id'].map(id_to_group)
        df.loc[id_dup_df.index, 'duplicate_group'] = id_dup_df['duplicate_group']
        print(f"    Connected case groups found: {len(unique_ids)}")
    else:
        print(f"    Connected case groups found: 0")

    # Statistics
    total_groups = df['duplicate_group'].nunique()
    grouped_docs = (df['duplicate_group'] != df['id']).sum()
    singleton_docs = (df['duplicate_group'] == df['id']).sum()

    print(f"\n  Total duplicate groups: {total_groups:,}")
    print(f"  Documents in groups: {grouped_docs:,} ({grouped_docs/len(df)*100:.2f}%)")
    print(f"  Singleton documents: {singleton_docs:,} ({singleton_docs/len(df)*100:.2f}%)")

    return df


def group_aware_stratified_split(df, label_col, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Perform group-aware stratified split.
    """
    print(f"\n[2] Performing group-aware stratified split for {label_col}...")

    # Filter out unlabeled/empty labels
    df_filtered = df[df[label_col].notna() & (df[label_col] != '')].copy()
    print(f"  Filtered dataset: {len(df_filtered):,} documents")

    # Initialize partitions
    df_filtered['partition'] = 'unassigned'

    # Get unique groups
    groups = df_filtered['duplicate_group'].unique()
    print(f"  Total groups to assign: {len(groups):,}")

    # For groups with multiple documents, count label distribution
    group_labels = {}
    for group in groups:
        group_df = df_filtered[df_filtered['duplicate_group'] == group]
        label_counts = group_df[label_col].value_counts()
        # Assign the dominant label to the group
        group_labels[group] = label_counts.index[0]

    # Create group-level dataframe
    group_df = pd.DataFrame({
        'group': groups,
        'label': [group_labels[g] for g in groups],
        'count': [df_filtered[df_filtered['duplicate_group'] == g].shape[0] for g in groups]
    })

    print(f"  Group label distribution:")
    for label in group_df['label'].unique():
        count = (group_df['label'] == label).sum()
        print(f"    {label}: {count} groups")

    # First split: train vs (val + test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=RANDOM_SEED)
    train_idx, valtest_idx = next(gss1.split(group_df, groups=group_df['group']))

    train_groups = group_df.iloc[train_idx]['group'].values
    valtest_groups = group_df.iloc[valtest_idx]['group'].values

    # Second split: val vs test from valtest
    gss2 = GroupShuffleSplit(n_splits=1, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=RANDOM_SEED)
    val_idx, test_idx = next(gss2.split(group_df.iloc[valtest_idx], groups=group_df.iloc[valtest_idx]['group']))

    val_groups = group_df.iloc[valtest_idx[val_idx]]['group'].values
    test_groups = group_df.iloc[valtest_idx[test_idx]]['group'].values

    # Assign partitions
    df_filtered.loc[df_filtered['duplicate_group'].isin(train_groups), 'partition'] = 'train'
    df_filtered.loc[df_filtered['duplicate_group'].isin(val_groups), 'partition'] = 'val'
    df_filtered.loc[df_filtered['duplicate_group'].isin(test_groups), 'partition'] = 'test'

    # Verify no group crosses partitions
    violations = 0
    for group in groups:
        partitions = df_filtered[df_filtered['duplicate_group'] == group]['partition'].unique()
        if len(partitions) > 1:
            violations += 1
            if violations <= 5:
                print(f"  WARNING: Group {group} spans multiple partitions: {partitions.tolist()}")

    if violations > 0:
        print(f"  Total partition violations: {violations}")
    else:
        print(f"  No partition violations detected")

    # Statistics
    train_count = (df_filtered['partition'] == 'train').sum()
    val_count = (df_filtered['partition'] == 'val').sum()
    test_count = (df_filtered['partition'] == 'test').sum()

    print(f"\n  Partition sizes:")
    print(f"    Train: {train_count:,} ({train_count/len(df_filtered)*100:.2f}%)")
    print(f"    Val: {val_count:,} ({val_count/len(df_filtered)*100:.2f}%)")
    print(f"    Test: {test_count:,} ({test_count/len(df_filtered)*100:.2f}%)")

    # Label distribution per partition
    print(f"\n  Label distribution by partition:")
    for partition in ['train', 'val', 'test']:
        part_df = df_filtered[df_filtered['partition'] == partition]
        print(f"    {partition}:")
        for label in sorted(df_filtered[label_col].unique()):
            count = (part_df[label_col] == label).sum()
            print(f"      {label}: {count:,} ({count/len(part_df)*100:.2f}%)")

    return df_filtered


def save_partitions(df, task_name, output_dir):
    """Save partitions to parquet files."""
    print(f"\n[3] Saving {task_name} partitions...")

    output_dir.mkdir(parents=True, exist_ok=True)

    for partition in ['train', 'val', 'test']:
        part_df = df[df['partition'] == partition].copy()
        output_path = output_dir / f"{partition}.parquet"
        part_df.to_parquet(output_path, index=False)
        print(f"  Saved {partition}: {len(part_df):,} documents to {output_path}")

    return df


def generate_partition_statistics(binary_df, three_class_df):
    """Generate partition statistics table."""
    print("\n[4] Generating partition statistics...")

    stats = []

    # Binary task statistics
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

    # Three-class task statistics
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
    print(f"  Saved partition statistics: {stats_path}")

    return stats_df


def generate_label_distribution(binary_df, three_class_df):
    """Generate label distribution table."""
    print("\n[5] Generating label distribution table...")

    dist = []

    # Binary task
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

    # Three-class task
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
    print(f"  Saved label distribution: {dist_path}")

    return dist_df


def main():
    print("=" * 80)
    print("PHASE 18C: JUD-IPL GROUP-AWARE DATASET PARTITIONING")
    print("=" * 80)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[0] Loading Jud-IPL data...")
    df = pd.read_parquet(DOCS_PATH)
    print(f"  Loaded {len(df):,} documents")

    # Create duplicate groups
    df = create_duplicate_groups(df)

    # Binary task split (Accepted/Rejected only)
    print("\n" + "=" * 80)
    print("BINARY TASK: Accepted vs Rejected")
    print("=" * 80)
    binary_df = df[df['label'].isin(['Accepted', 'Rejected'])].copy()
    binary_df = group_aware_stratified_split(binary_df, 'label')

    binary_output_dir = OUTPUT_DIR / "binary"
    save_partitions(binary_df, 'binary', binary_output_dir)

    # Three-class task split (Accepted/Rejected/Other)
    print("\n" + "=" * 80)
    print("THREE-CLASS TASK: Accepted vs Rejected vs Other")
    print("=" * 80)
    three_class_df = df[df['label'].isin(['Accepted', 'Rejected', 'Other'])].copy()
    three_class_df = group_aware_stratified_split(three_class_df, 'label')

    three_class_output_dir = OUTPUT_DIR / "three_class"
    save_partitions(three_class_df, 'three_class', three_class_output_dir)

    # Generate statistics
    stats_df = generate_partition_statistics(binary_df, three_class_df)
    dist_df = generate_label_distribution(binary_df, three_class_df)

    print("\n" + "=" * 80)
    print("Group-aware partitioning completed successfully!")
    print("=" * 80)

    return binary_df, three_class_df, stats_df, dist_df


if __name__ == "__main__":
    main()
