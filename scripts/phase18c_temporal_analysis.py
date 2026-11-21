"""
Phase 18C: Temporal Split Analysis

Analyzes the feasibility of chronological partitioning given that only 40.1% of documents have known judgment dates.

Output:
- results/tables/phase18c_temporal_split_analysis.csv
- results/reports/phase18c_temporal_split_analysis.md
"""

import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

DOCS_PATH = PROJECT_ROOT / "data" / "processed" / "judipl" / "documents.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"


def parse_date(date_str):
    """Parse date string in DD/MM/YYYY format."""
    if pd.isna(date_str) or date_str == "":
        return None
    try:
        parts = date_str.split('/')
        if len(parts) == 3:
            day, month, year = parts
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    except:
        pass
    return None


def main():
    print("=" * 80)
    print("PHASE 18C: TEMPORAL SPLIT ANALYSIS")
    print("=" * 80)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[0] Loading Jud-IPL data...")
    df = pd.read_parquet(DOCS_PATH)
    print(f"  Loaded {len(df):,} documents")

    # Parse dates
    print("\n[1] Parsing judgment dates...")
    df['parsed_date'] = df['date_of_judgment'].apply(parse_date)
    df['parsed_date'] = pd.to_datetime(df['parsed_date'], errors='coerce')

    has_date = df['parsed_date'].notna()
    missing_date = df['parsed_date'].isna()

    print(f"  Documents with valid dates: {has_date.sum():,} ({has_date.mean()*100:.1f}%)")
    print(f"  Documents with missing dates: {missing_date.sum():,} ({missing_date.mean()*100:.1f}%)")

    # Date range analysis
    if has_date.sum() > 0:
        dated_df = df[has_date].copy()
        min_date = dated_df['parsed_date'].min()
        max_date = dated_df['parsed_date'].max()
        date_range_years = (max_date - min_date).days / 365.25

        print(f"\n[2] Date range analysis...")
        print(f"  Earliest judgment: {min_date.strftime('%Y-%m-%d')}")
        print(f"  Latest judgment: {max_date.strftime('%Y-%m-%d')}")
        print(f"  Date range: {date_range_years:.1f} years")

        # Decade distribution
        dated_df['decade'] = (dated_df['parsed_date'].dt.year // 10) * 10
        decade_dist = dated_df['decade'].value_counts().sort_index()

        print(f"\n[3] Decade distribution...")
        for decade, count in decade_dist.items():
            print(f"  {decade}s: {count:,} ({count/len(dated_df)*100:.1f}%)")

        # Label distribution by decade
        print(f"\n[4] Label distribution by decade...")
        label_by_decade = dated_df.groupby(['decade', 'label']).size().unstack(fill_value=0)
        print(label_by_decade)

    # Temporal split feasibility
    print(f"\n[5] Temporal split feasibility assessment...")

    if has_date.sum() < len(df) * 0.8:
        print(f"  STATUS: NOT RECOMMENDED")
        print(f"  REASON: Only {has_date.mean()*100:.1f}% of documents have dates")
        print(f"  IMPACT: Temporal split would exclude {missing_date.sum():,} documents")
        print(f"  RECOMMENDATION: Use group-aware random split as primary method")
        print(f"  ALTERNATIVE: Conduct temporal sensitivity analysis on dated subset only")
    else:
        print(f"  STATUS: FEASIBLE")
        print(f"  REASON: {has_date.mean()*100:.1f}% of documents have dates")
        print(f"  RECOMMENDATION: Can use chronological split (e.g., pre-2010 train, 2010-2015 val, post-2015 test)")

    # Generate analysis table
    analysis_data = {
        'metric': [
            'total_documents',
            'documents_with_dates',
            'documents_without_dates',
            'date_coverage_percentage',
            'earliest_judgment_date',
            'latest_judgment_date',
            'date_range_years',
            'temporal_split_feasible'
        ],
        'value': [
            len(df),
            has_date.sum(),
            missing_date.sum(),
            has_date.mean() * 100,
            str(min_date) if has_date.sum() > 0 else 'N/A',
            str(max_date) if has_date.sum() > 0 else 'N/A',
            date_range_years if has_date.sum() > 0 else 0,
            'NO' if has_date.mean() < 0.8 else 'YES'
        ]
    }

    analysis_df = pd.DataFrame(analysis_data)
    analysis_path = TABLES_DIR / "phase18c_temporal_split_analysis.csv"
    analysis_df.to_csv(analysis_path, index=False)
    print(f"\n  Saved analysis: {analysis_path}")

    # Generate report
    report_path = REPORTS_DIR / "phase18c_temporal_split_analysis.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 18C: Temporal Split Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Documents:** {len(df):,}\n")
        f.write(f"- **Documents with Dates:** {has_date.sum():,} ({has_date.mean()*100:.1f}%)\n")
        f.write(f"- **Documents without Dates:** {missing_date.sum():,} ({missing_date.mean()*100:.1f}%)\n")
        f.write(f"- **Temporal Split Feasible:** NO\n\n")

        f.write("## Date Coverage\n\n")
        f.write(f"Only {has_date.mean()*100:.1f}% of documents have judgment dates. ")
        f.write("This is insufficient for a robust temporal split, as it would exclude ")
        f.write(f"{missing_date.sum():,} documents ({missing_date.mean()*100:.1f}% of the dataset).\n\n")

        if has_date.sum() > 0:
            f.write("## Date Range\n\n")
            f.write(f"- **Earliest Judgment:** {min_date.strftime('%Y-%m-%d')}\n")
            f.write(f"- **Latest Judgment:** {max_date.strftime('%Y-%m-%d')}\n")
            f.write(f"- **Date Range:** {date_range_years:.1f} years\n\n")

            f.write("## Decade Distribution\n\n")
            f.write("| Decade | Count | Percentage |\n")
            f.write("|--------|-------|------------|\n")
            for decade, count in decade_dist.items():
                f.write(f"| {decade}s | {count:,} | {count/len(dated_df)*100:.1f}% |\n")
            f.write("\n")

        f.write("## Recommendations\n\n")
        f.write("1. **Primary Split:** Use group-aware random split (70/15/15) as implemented in phase18c_grouped_split\n")
        f.write("2. **Temporal Sensitivity Analysis:** Can be conducted on the dated subset ({has_date.sum():,} documents) separately\n")
        f.write("3. **Missing Date Policy:** Do not silently discard undated records. Document exclusion policy explicitly.\n")
        f.write("4. **Future Work:** Consider collecting or inferring missing dates if temporal analysis becomes critical.\n\n")

        f.write("## Conclusion\n\n")
        f.write("Temporal partitioning is not feasible as the primary split method due to insufficient date coverage. ")
        f.write("The group-aware random split ensures all documents are utilized while preventing duplicate leakage. ")
        f.write("Temporal sensitivity can be analyzed as a secondary investigation on the dated subset only.\n\n")

    print(f"  Saved report: {report_path}")

    print("\n" + "=" * 80)
    print("Temporal split analysis completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
