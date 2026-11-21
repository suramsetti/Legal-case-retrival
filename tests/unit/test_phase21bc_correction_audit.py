import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from phase21bc_common import (  # noqa: E402
    CLASSES,
    PROB_COLUMNS,
    aligned_frames,
    bootstrap,
    classification_metrics,
    mcnemar,
    validate_prediction_frame,
)


def test_prediction_id_alignment_and_probability_validation():
    merged, m4, m8 = aligned_frames()
    expected = set(merged.id)
    assert set(m4.id) == set(m8.id) == expected
    assert not m4.id.duplicated().any()
    assert not m8.id.duplicated().any()
    assert validate_prediction_frame(m4, "M4", expected) == []
    assert validate_prediction_frame(m8, "M8", expected) == []


def test_duplicate_id_detection():
    merged, m4, _ = aligned_frames()
    duplicate = pd.concat([m4, m4.iloc[[0]]], ignore_index=True)
    errors = validate_prediction_frame(duplicate, "M4", set(merged.id))
    assert any(e["check"] == "duplicate_ids" for e in errors)


def test_confusion_totals_accuracy_and_per_class_metrics():
    merged, _, _ = aligned_frames()
    for model in CLASSES[:0] or ["M4", "M8"]:
        summary, rows, cm = classification_metrics(merged, model)
        assert cm.sum() == 6311
        assert np.isclose(summary["accuracy"], np.trace(cm) / 6311)
        assert sum(r["support"] for r in rows) == 6311
        assert all(0 <= r["precision"] <= 1 and 0 <= r["recall"] <= 1 for r in rows)


def test_mcnemar_count_identities():
    merged, _, _ = aligned_frames()
    result = mcnemar(merged)
    assert result["n11_both_correct"] + result["n00_both_incorrect"] + result["n01_m4_incorrect_m8_correct"] + result["n10_m4_correct_m8_incorrect"] == 6311
    assert result["m4_correct"] == result["n11_both_correct"] + result["n10_m4_correct_m8_incorrect"]
    assert result["m8_correct"] == result["n11_both_correct"] + result["n01_m4_incorrect_m8_correct"]


def test_bootstrap_reproducibility_and_ovr_ovo_independence():
    merged, _, _ = aligned_frames()
    first = bootstrap(merged)
    second = bootstrap(merged)
    pd.testing.assert_frame_equal(first, second)
    m4 = classification_metrics(merged, "M4")[0]
    assert np.isfinite(m4["roc_auc_ovr"]) and np.isfinite(m4["roc_auc_ovo"])
    assert m4["roc_auc_ovr"] != m4["roc_auc_ovo"] or np.isclose(m4["roc_auc_ovr"], m4["roc_auc_ovo"])


def test_original_phase21b_files_preserved():
    for name in ["m4_test_predictions.csv", "m8_test_predictions.csv"]:
        path = ROOT / "results" / "predictions" / "phase21b" / name
        assert path.exists()
    assert (ROOT / "results" / "tables" / "phase21b_test_results.csv").exists()
