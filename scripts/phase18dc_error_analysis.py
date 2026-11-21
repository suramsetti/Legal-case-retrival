"""
Phase 18D-C: Standalone Error Analysis Script.

Analyzes misclassified judgments on the held-out test set across:
1. Case outcome (False Positives vs False Negatives)
2. Case categories (Civil, Criminal, etc.)
3. Document length quartiles
4. Truncation flags (>=32,760 chars)
5. Missing metadata attributes
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def run_error_analysis():
    print("=" * 80, flush=True)
    print("PHASE 18D-C: STANDALONE ERROR ANALYSIS", flush=True)
    print("=" * 80, flush=True)

    err_path = TABLES_DIR / "phase18dc_error_analysis.csv"
    if not err_path.exists():
        print(f"Error analysis table not found at {err_path}. Running generator...", flush=True)
        from scripts.phase18dc_final_evaluation import run_pipeline
        run_pipeline()

    df = pd.read_csv(err_path)
    print("\nError Rate Breakdown by Dimension:")
    print("-" * 60)
    for dim, group in df.groupby("dimension"):
        print(f"\n[{dim}]")
        for _, row in group.iterrows():
            print(f"  {str(row['stratum']):<25} Cases: {int(row['count']):<6} Acc: {row['accuracy']*100:.2f}%  Err: {row['error_rate']*100:.2f}%")


if __name__ == "__main__":
    run_error_analysis()

