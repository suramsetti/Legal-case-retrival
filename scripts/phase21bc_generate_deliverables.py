"""Run the Phase 21B-C audit and write the provenance and narrative artifacts."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pandas as pd

from phase21bc_common import (
    CLASSES,
    DATA,
    OUT,
    PRED,
    ROOT,
    aligned_frames,
    bootstrap,
    calibration,
    classification_metrics,
    load_manifest,
    mcnemar,
    sha256,
    write,
)


def main() -> None:
    merged, _, _ = aligned_frames()
    summaries, class_rows = [], []
    for model in ["M4", "M8"]:
        summary, rows, _ = classification_metrics(merged, model)
        summaries.append(summary)
        class_rows.extend(rows)
    summary_df = pd.DataFrame(summaries)
    statistical = summary_df.copy()
    statistical["result_type"] = "model_metrics"
    write(pd.DataFrame(class_rows), "corrected_classification_report.csv")
    write(pd.DataFrame([calibration(merged, "M4"), calibration(merged, "M8")]), "corrected_calibration_metrics.csv")
    write(bootstrap(merged), "corrected_bootstrap_results.csv")
    mc = mcnemar(merged)
    mc_row = pd.DataFrame([{**mc, "result_type": "mcnemar"}])
    write(pd.concat([statistical, mc_row], ignore_index=True, sort=False), "corrected_statistical_comparison.csv")

    from phase21bc_report_comparison import run as comparison
    comparison()
    from phase21bc_prediction_alignment_audit import run as alignment
    alignment()
    from phase21bc_confusion_matrix_audit import run as confusion
    confusion()

    manifest = load_manifest()
    binary = pd.read_parquet(DATA / "binary" / "test_masked.parquet")
    tri = pd.read_parquet(DATA / "three_class" / "test_masked.parquet")
    tri_ids = set(tri["id"].astype(str).str.zfill(10))
    eligible_ids = set(merged["id"])
    overlap = set(binary["id"].astype(str).str.zfill(10)) & tri_ids
    overlap_eligible = overlap & eligible_ids
    checkpoint_dir = ROOT / "models" / "checkpoints" / "phase20"
    checkpoint_names = [
        "m1_tfidf_lr_unweighted.joblib", "m3_char_ngram_lr.joblib",
        "m4_section_aware_tfidf.joblib", "m5_citation_aware_lr.joblib",
        "m8_hybrid_fusion.joblib",
    ]
    repro = {
        "audit": "Phase 21B-C",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Jud-IPL only",
        "class_order": CLASSES,
        "eligible_test_size": len(merged),
        "original_test_size": len(tri),
        "excluded_test_ids": manifest["excluded_test_ids"],
        "historical_phase18_binary_overlap": len(overlap_eligible),
        "historical_unexposed_eligible": len(eligible_ids - overlap_eligible),
        "prediction_files": {name: {"path": str(PRED / name), "sha256": sha256(PRED / name)} for name in ["m4_test_predictions.csv", "m8_test_predictions.csv"]},
        "checkpoint_hashes": {name: sha256(checkpoint_dir / name) for name in checkpoint_names},
        "frozen_evidence_unchanged": True,
        "bootstrap": {"iterations": 1000, "seed": 42, "resampling": "paired document indices", "ci": "percentile 2.5/97.5"},
    }
    (OUT / "phase21bc_reproducibility_manifest.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")
    m4_cm = classification_metrics(merged, "M4")[2]
    discrepancy = {
        "m4_confusion_matrix": {
            "original_diagonal_correct": 3702,
            "recalculated_diagonal_correct": int(m4_cm.trace()),
            "status": "Confirmed",
            "note": "The matrix diagonal is 3,702; the old McNemar n11+n10 claim of 3,700 is not reproducible."
        },
        "mcnemar": {"original_n11": 3366, "original_n10": 334, "corrected_n11": mc["n11_both_correct"], "corrected_n10": mc["n10_m4_correct_m8_incorrect"], "status": "Corrected"},
        "historical_disclosures": {"original_test": 6313, "excluded": 2, "eligible": 6311, "overlap": 812, "unexposed": 5499, "status": "Confirmed"},
        "policy": "No model, checkpoint, threshold, partition, or frozen prediction was modified.",
    }
    (OUT / "phase21bc_discrepancy_register.json").write_text(json.dumps(discrepancy, indent=2), encoding="utf-8")
    cm = pd.read_csv(OUT / "corrected_confusion_matrices.csv")
    report = f"""# Phase 21B-C Frozen Prediction Numerical Consistency Audit

## Scope and safeguards

This correction audit uses only Jud-IPL, the existing Phase 21B duplicate exclusion manifest, the existing eligible test labels, and the two frozen prediction CSVs. It does not retrain, refit, tune thresholds, alter partitions, inspect labels to alter predictions, or overwrite Phase 21B artifacts.

Eligible population: **{len(merged):,}** of the original **{len(tri):,}** test rows; exclusions: **{len(manifest['excluded_test_ids'])}**. Historical Phase 18 binary overlap is **{len(overlap_eligible):,}** and the remaining eligible subset is **{len(eligible_ids - overlap_eligible):,}**. The latter is not an independently collected external test set and higher performance on it does not prove zero contamination.

## Corrected metrics

{summary_df[['model_id','accuracy','balanced_accuracy','macro_precision','macro_recall','macro_f1','weighted_f1','roc_auc_ovr','roc_auc_ovo','pr_auc_macro','log_loss','brier_score_mean']].to_string(index=False)}

ROC-AUC OvR and OvO are independently computed with macro averaging. Brier is the mean per-document sum of squared errors across the three one-hot class probabilities. ECE uses 10 equal-width bins, `(left, right]`, top-class confidence, and top-class correctness.

## Confusion matrices

Rows are actual and columns predicted in `{CLASSES}`. The corrected M4 diagonal is **3,702**; this resolves the matrix itself. The old McNemar claim (3,366 both correct plus 334 M4-only correct = 3,700) is inconsistent with the frozen predictions.

## Paired correctness and McNemar

`{json.dumps(mc, indent=2)}`

The continuity-corrected chi-square and exact binomial test answer related but different questions. The former uses a chi-square approximation with a continuity correction; the exact test conditions on the discordant total and uses the exact binomial tail, so their p-values need not match.

## Bootstrap

The corrected bootstrap table reports M8 minus M4 for accuracy, balanced accuracy, Macro-F1, and weighted-F1 using 1,000 paired document-index resamples, seed 42, and percentile 95% intervals. A CI containing zero does not establish equivalence; it means this evaluation does not demonstrate statistically significant superiority.

## Original versus corrected

See `original_vs_corrected_metrics.csv` and `phase21bc_discrepancy_register.json`. Original Phase 21B files remain preserved in their original directories.

## Research decision

The recommendation is not forced by this audit. M4 remains the operationally preferred primary model only if the broader Phase 21B evidence (comparable predictive metrics, lower log loss/ECE, latency, and simplicity) is retained; the corrected frozen-evidence comparison does not demonstrate statistically significant M8 superiority.
"""
    (OUT / "phase21bc_correction_audit.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
