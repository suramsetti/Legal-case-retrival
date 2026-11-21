"""
Phase 21B-5: Comparative Statistical Analysis Script.

Performs paired statistical analysis between M8 and M4 on identical eligible test cases (N = 6,311):
1. Point estimate differences for Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1, and Class-Wise F1
2. Paired bootstrap confidence intervals (B = 1,000, fixed seed 42)
3. McNemar's test for paired classification errors
4. Saves results/tables/phase21b_statistical_comparison.csv and results/reports/phase21b_statistical_analysis.md
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.phase20_multiclass_models import CANONICAL_CLASSES

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c" / "three_class"
PRED_DIR = PROJECT_ROOT / "results" / "predictions" / "phase21b"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_statistical_comparison():
    print("=" * 80)
    print("PHASE 21B-5: COMPARATIVE STATISTICAL ANALYSIS (M8 vs M4)")
    print("=" * 80)

    te_df = pd.read_parquet(DATA_DIR / "test_masked.parquet")
    df_m4 = pd.read_csv(PRED_DIR / "m4_test_predictions.csv", dtype={"id": str})
    df_m8 = pd.read_csv(PRED_DIR / "m8_test_predictions.csv", dtype={"id": str})

    merged = pd.merge(df_m4[["id", "predicted_label"]], te_df[["id", "label"]], on="id")
    merged = pd.merge(merged, df_m8[["id", "predicted_label"]], on="id", suffixes=("_m4", "_m8"))

    y_true = merged["label"].values
    y_m4 = merged["predicted_label_m4"].values
    y_m8 = merged["predicted_label_m8"].values
    n = len(y_true)

    # Point estimates
    acc_m4 = accuracy_score(y_true, y_m4)
    acc_m8 = accuracy_score(y_true, y_m8)
    bal_m4 = balanced_accuracy_score(y_true, y_m4)
    bal_m8 = balanced_accuracy_score(y_true, y_m8)
    f1_m4 = f1_score(y_true, y_m4, average="macro")
    f1_m8 = f1_score(y_true, y_m8, average="macro")
    wf1_m4 = f1_score(y_true, y_m4, average="weighted")
    wf1_m8 = f1_score(y_true, y_m8, average="weighted")

    delta_acc = acc_m8 - acc_m4
    delta_bal = bal_m8 - bal_m4
    delta_f1 = f1_m8 - f1_m4
    delta_wf1 = wf1_m8 - wf1_m4

    # Class-wise point estimates
    class_deltas = {}
    for c in CANONICAL_CLASSES:
        f_m4 = f1_score(y_true == c, y_m4 == c, zero_division=0)
        f_m8 = f1_score(y_true == c, y_m8 == c, zero_division=0)
        class_deltas[c] = {
            "m4": round(float(f_m4), 4),
            "m8": round(float(f_m8), 4),
            "delta": round(float(f_m8 - f_m4), 4),
        }

    # McNemar's Test
    correct_m4 = (y_m4 == y_true)
    correct_m8 = (y_m8 == y_true)

    n00 = int((~correct_m4 & ~correct_m8).sum())
    n01 = int((~correct_m4 & correct_m8).sum())   # M4 wrong, M8 right
    n10 = int((correct_m4 & ~correct_m8).sum())   # M4 right, M8 wrong
    n11 = int((correct_m4 & correct_m8).sum())

    # Continuity-corrected chi-square: (|n01 - n10| - 1)^2 / (n01 + n10)
    if (n01 + n10) > 0:
        chi2_stat = ((abs(n01 - n10) - 1.0) ** 2) / (n01 + n10)
        p_val_mcnemar = 1.0 - stats.chi2.cdf(chi2_stat, df=1)
    else:
        chi2_stat = 0.0
        p_val_mcnemar = 1.0

    # Paired Bootstrap (B = 1,000, seed = 42)
    print("Running 1,000 paired bootstrap iterations...")
    np.random.seed(42)
    B = 1000
    boot_delta_f1 = np.zeros(B)
    boot_delta_acc = np.zeros(B)
    boot_delta_bal = np.zeros(B)

    for b in range(B):
        idx = np.random.randint(0, n, size=n)
        b_true = y_true[idx]
        b_m4 = y_m4[idx]
        b_m8 = y_m8[idx]

        b_f1_m4 = f1_score(b_true, b_m4, average="macro", zero_division=0)
        b_f1_m8 = f1_score(b_true, b_m8, average="macro", zero_division=0)
        boot_delta_f1[b] = b_f1_m8 - b_f1_m4

        b_acc_m4 = accuracy_score(b_true, b_m4)
        b_acc_m8 = accuracy_score(b_true, b_m8)
        boot_delta_acc[b] = b_acc_m8 - b_acc_m4

        b_bal_m4 = balanced_accuracy_score(b_true, b_m4)
        b_bal_m8 = balanced_accuracy_score(b_true, b_m8)
        boot_delta_bal[b] = b_bal_m8 - b_bal_m4

    ci_f1_lower, ci_f1_upper = np.percentile(boot_delta_f1, [2.5, 97.5])
    ci_acc_lower, ci_acc_upper = np.percentile(boot_delta_acc, [2.5, 97.5])
    ci_bal_lower, ci_bal_upper = np.percentile(boot_delta_bal, [2.5, 97.5])

    p_boot_f1 = (boot_delta_f1 <= 0).mean()
    p_boot_acc = (boot_delta_acc <= 0).mean()

    # Compile Comparison Records
    comp_records = [
        {
            "metric": "Accuracy",
            "m4_score": round(float(acc_m4), 4),
            "m8_score": round(float(acc_m8), 4),
            "delta_m8_minus_m4": round(float(delta_acc), 4),
            "ci_95_lower": round(float(ci_acc_lower), 4),
            "ci_95_upper": round(float(ci_acc_upper), 4),
            "p_value": round(float(p_val_mcnemar), 6),
            "test_type": "McNemar Chi2",
            "stat_significant": bool(p_val_mcnemar < 0.05),
        },
        {
            "metric": "Macro-F1",
            "m4_score": round(float(f1_m4), 4),
            "m8_score": round(float(f1_m8), 4),
            "delta_m8_minus_m4": round(float(delta_f1), 4),
            "ci_95_lower": round(float(ci_f1_lower), 4),
            "ci_95_upper": round(float(ci_f1_upper), 4),
            "p_value": round(float(p_boot_f1), 6),
            "test_type": "Paired Bootstrap (B=1000)",
            "stat_significant": bool(ci_f1_lower > 0),
        },
        {
            "metric": "Balanced Accuracy",
            "m4_score": round(float(bal_m4), 4),
            "m8_score": round(float(bal_m8), 4),
            "delta_m8_minus_m4": round(float(delta_bal), 4),
            "ci_95_lower": round(float(ci_bal_lower), 4),
            "ci_95_upper": round(float(ci_bal_upper), 4),
            "p_value": round(float((boot_delta_bal <= 0).mean()), 6),
            "test_type": "Paired Bootstrap (B=1000)",
            "stat_significant": bool(ci_bal_lower > 0),
        },
        {
            "metric": "Weighted-F1",
            "m4_score": round(float(wf1_m4), 4),
            "m8_score": round(float(wf1_m8), 4),
            "delta_m8_minus_m4": round(float(delta_wf1), 4),
            "ci_95_lower": "N/A",
            "ci_95_upper": "N/A",
            "p_value": "N/A",
            "test_type": "Descriptive",
            "stat_significant": False,
        },
    ]

    for c in CANONICAL_CLASSES:
        comp_records.append({
            "metric": f"F1-{c}",
            "m4_score": class_deltas[c]["m4"],
            "m8_score": class_deltas[c]["m8"],
            "delta_m8_minus_m4": class_deltas[c]["delta"],
            "ci_95_lower": "N/A",
            "ci_95_upper": "N/A",
            "p_value": "N/A",
            "test_type": "Per-Class Point Diff",
            "stat_significant": False,
        })

    comp_df = pd.DataFrame(comp_records)
    out_table = TABLES_DIR / "phase21b_statistical_comparison.csv"
    comp_df.to_csv(out_table, index=False)
    print(f"Saved Statistical Comparison Table to {out_table}")

    # Build Markdown Report
    report_path = REPORTS_DIR / "phase21b_statistical_analysis.md"
    content = f"""# Phase 21B-5: Comparative Statistical Analysis Report (M8 vs M4)

**Evaluation Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Project:** Jud-IPL Standalone Legal Intelligence Research System
**Evaluation Corpus:** Held-out Three-Class Test Partition ($N = 6,311$ eligible cases)

---

## 1. Executive Summary

A paired statistical comparison was conducted between **Model M8 (Hybrid Multi-Class Fusion Ensemble)** and **Model M4 (Section-Aware TF-IDF Classifier)** across identical eligible test documents.

### Key Comparative Metrics:

| Metric | M4 (Section-Aware) | M8 (Hybrid Fusion) | Difference ($\Delta$) | 95% Bootstrap CI | Statistical Significance ($p < 0.05$) |
|---|---|---|---|---|---|
| **Accuracy** | {acc_m4*100:.2f}% | {acc_m8*100:.2f}% | **{delta_acc*100:+.2f}%** | [{ci_acc_lower*100:+.2f}%, {ci_acc_upper*100:+.2f}%] | {'Significant' if p_val_mcnemar < 0.05 else 'Not Significant'} ($p = {p_val_mcnemar:.4f}$) |
| **Macro-F1** | {f1_m4:.4f} | {f1_m8:.4f} | **{delta_f1:+.4f}** | [{ci_f1_lower:+.4f}, {ci_f1_upper:+.4f}] | {'Significant' if ci_f1_lower > 0 else 'Inconclusive'} ($p = {p_boot_f1:.4f}$) |
| **Balanced Acc** | {bal_m4*100:.2f}% | {bal_m8*100:.2f}% | **{delta_bal*100:+.2f}%** | [{ci_bal_lower*100:+.2f}%, {ci_bal_upper*100:+.2f}%] | {'Significant' if ci_bal_lower > 0 else 'Inconclusive'} |
| **Weighted-F1** | {wf1_m4:.4f} | {wf1_m8:.4f} | **{delta_wf1:+.4f}** | N/A | N/A |

---

## 2. McNemar's Paired Error Analysis

Contingency table of paired predictions across all 6,311 eligible documents:

| Contingency Cell | Definition | Count |
|---|---|---|
| **$n_{{11}}$** | Both models correct | {n11:,} ({n11/n*100:.2f}%) |
| **$n_{{00}}$** | Both models incorrect | {n00:,} ({n00/n*100:.2f}%) |
| **$n_{{01}}$** | M4 incorrect, M8 correct | **{n01:,}** ({n01/n*100:.2f}%) |
| **$n_{{10}}$** | M4 correct, M8 incorrect | **{n10:,}** ({n10/n*100:.2f}%) |

- **Net Error Reduction by M8:** $n_{{01}} - n_{{10}} = {n01 - n10:+,d}$ judgments.
- **McNemar Chi-Square Statistic ($\chi^2$):** {chi2_stat:.4f} ($df = 1$)
- **P-Value:** {p_val_mcnemar:.6f}

---

## 3. Class-Wise F1 Comparison

| Class | M4 F1-Score | M8 F1-Score | F1 Delta ($\Delta$) | Support ($N$) |
|---|---|---|---|---|
| **Accepted** | {class_deltas['Accepted']['m4']:.4f} | {class_deltas['Accepted']['m8']:.4f} | **{class_deltas['Accepted']['delta']:+.4f}** | {(y_true == 'Accepted').sum():,} |
| **Other** | {class_deltas['Other']['m4']:.4f} | {class_deltas['Other']['m8']:.4f} | **{class_deltas['Other']['delta']:+.4f}** | {(y_true == 'Other').sum():,} |
| **Rejected** | {class_deltas['Rejected']['m4']:.4f} | {class_deltas['Rejected']['m8']:.4f} | **{class_deltas['Rejected']['delta']:+.4f}** | {(y_true == 'Rejected').sum():,} |

---

## 4. Statistical Conclusion

- Model M8 achieves a **{delta_acc*100:+.2f}%** gain in overall test accuracy and **{delta_f1:+.4f}** gain in Macro-F1 over M4.
- The 95% paired bootstrap confidence interval for Macro-F1 difference is **[{ci_f1_lower:+.4f}, {ci_f1_upper:+.4f}]**.
- McNemar's test yielded $p = {p_val_mcnemar:.6f}$.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved Statistical Analysis Report to {report_path}")


if __name__ == "__main__":
    run_statistical_comparison()

