"""
Phase 21B-2: Cross-Split Duplicate Resolution Script.

Audits exact masked-text hash overlaps between splits and generates:
1. results/reports/phase21b_duplicate_exclusion_manifest.json
2. results/tables/phase21b_duplicate_exclusion_audit.csv

Identifies cross-partition duplicate pairs without accessing test labels,
and creates the deterministic exclusion manifest for test evaluation.
"""

import sys
import os
import json
import hashlib
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c" / "three_class"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def resolve_cross_split_duplicates():
    print("=" * 80)
    print("PHASE 21B-2: CROSS-SPLIT DUPLICATE RESOLUTION & EXCLUSION AUDIT")
    print("=" * 80)

    tr_df = pd.read_parquet(DATA_DIR / "train_masked.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val_masked.parquet")
    te_df = pd.read_parquet(DATA_DIR / "test_masked.parquet")

    print(f"Loaded partitions: Train={len(tr_df):,}, Val={len(val_df):,}, Test={len(te_df):,}")

    # Compute MD5 hashes of masked_text
    def get_hash_map(df, split_name):
        hmap = {}
        for idx, row in df.iterrows():
            h = hashlib.md5(row["masked_text"].encode("utf-8")).hexdigest()
            hmap[row["id"]] = {
                "id": str(row["id"]),
                "raw_id": str(row.get("raw_id", "")),
                "duplicate_group": str(row.get("duplicate_group", "")),
                "hash": h,
                "split": split_name,
            }
        return hmap

    tr_map = get_hash_map(tr_df, "Train")
    val_map = get_hash_map(val_df, "Validation")
    te_map = get_hash_map(te_df, "Test")

    # Inverted map: hash -> list of doc records
    hash_to_docs = {}
    for d in list(tr_map.values()) + list(val_map.values()) + list(te_map.values()):
        h = d["hash"]
        if h not in hash_to_docs:
            hash_to_docs[h] = []
        hash_to_docs[h].append(d)

    # Find cross-split collisions
    collisions = []
    excluded_test_ids = set()
    audit_records = []

    for h, doc_list in hash_to_docs.items():
        splits_present = set(d["split"] for d in doc_list)
        if len(splits_present) > 1:
            collisions.append({
                "hash": h,
                "splits": list(splits_present),
                "documents": doc_list,
            })
            # If present in Test, flag for exclusion
            for d in doc_list:
                if d["split"] == "Test":
                    excluded_test_ids.add(d["id"])

                audit_records.append({
                    "hash": h,
                    "split": d["split"],
                    "document_id": d["id"],
                    "raw_id": d["raw_id"],
                    "duplicate_group": d["duplicate_group"],
                    "action": "EXCLUDE_FROM_TEST" if d["split"] == "Test" else "LOGGED_PARTNER",
                })

    original_test_count = len(te_df)
    excluded_count = len(excluded_test_ids)
    eligible_test_count = original_test_count - excluded_count

    print(f"\nCross-Split Collisions Found: {len(collisions)}")
    for c in collisions:
        print(f"  Hash {c['hash']} across splits: {c['splits']}")
        for d in c["documents"]:
            print(f"    - Split: {d['split']:<10} ID: {d['id']:<12} raw_id: {d['raw_id']:<10} group: {d['duplicate_group']}")

    print(f"\nOriginal Test Size: {original_test_count:,}")
    print(f"Test Cases Excluded: {excluded_count:,} ({list(excluded_test_ids)})")
    print(f"Eligible Test Size: {eligible_test_count:,}")

    # Build Manifest JSON
    manifest = {
        "audit_phase": "Phase 21B-2",
        "description": "Deterministic cross-split duplicate exclusion manifest",
        "original_test_size": original_test_count,
        "excluded_test_count": excluded_count,
        "eligible_test_size": eligible_test_count,
        "excluded_test_ids": sorted(list(excluded_test_ids)),
        "collision_details": collisions,
    }

    manifest_path = REPORTS_DIR / "phase21b_duplicate_exclusion_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved Exclusion Manifest to {manifest_path}")

    # Build Audit CSV Table
    audit_df = pd.DataFrame(audit_records)
    audit_csv = TABLES_DIR / "phase21b_duplicate_exclusion_audit.csv"
    audit_df.to_csv(audit_csv, index=False)
    print(f"Saved Duplicate Exclusion Audit Table to {audit_csv}")

    return manifest


if __name__ == "__main__":
    resolve_cross_split_duplicates()

