"""
Unit tests for Phase 18D-C: Calibration Audit, Split Isolation, and Model Verification.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure offline
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from scripts.phase18dc_calibration_audit import PlattCalibrator, evaluate_probs


class TestSplitIsolation:
    """Verifies that no IDs or duplicate groups leak across train, val, and test partitions."""

    @pytest.fixture(scope="class")
    def partitions(self):
        tr = pd.read_parquet("data/processed/judipl/phase18c/binary/train_masked.parquet")
        va = pd.read_parquet("data/processed/judipl/phase18c/binary/val_masked.parquet")
        te = pd.read_parquet("data/processed/judipl/phase18c/binary/test_masked.parquet")
        return tr, va, te

    def test_zero_id_overlap(self, partitions):
        tr, va, te = partitions
        s_tr = set(tr["id"])
        s_va = set(va["id"])
        s_te = set(te["id"])
        assert len(s_tr.intersection(s_va)) == 0, "Train and Val share IDs"
        assert len(s_tr.intersection(s_te)) == 0, "Train and Test share IDs"
        assert len(s_va.intersection(s_te)) == 0, "Val and Test share IDs"

    def test_zero_raw_id_overlap(self, partitions):
        tr, va, te = partitions
        s_tr = set(tr["raw_id"])
        s_va = set(va["raw_id"])
        s_te = set(te["raw_id"])
        assert len(s_tr.intersection(s_va)) == 0, "Train and Val share raw_ids"
        assert len(s_tr.intersection(s_te)) == 0, "Train and Test share raw_ids"
        assert len(s_va.intersection(s_te)) == 0, "Val and Test share raw_ids"

    def test_zero_duplicate_group_overlap(self, partitions):
        tr, va, te = partitions
        s_tr = set(tr["duplicate_group"])
        s_va = set(va["duplicate_group"])
        s_te = set(te["duplicate_group"])
        assert len(s_tr.intersection(s_va)) == 0, "Train and Val share duplicate groups"
        assert len(s_tr.intersection(s_te)) == 0, "Train and Test share duplicate groups"
        assert len(s_va.intersection(s_te)) == 0, "Val and Test share duplicate groups"


class TestPlattCalibratorMathematics:
    """Tests the mathematical properties of the Platt calibrator."""

    def test_monotonicity_preserves_auc(self):
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, size=200)
        probs = np.random.uniform(0.1, 0.9, size=200)

        cal = PlattCalibrator()
        cal.fit(probs, y_true)

        p_cal = cal.predict_proba(probs)

        # Monotonicity test: if p1 > p2 then p_cal1 >= p_cal2
        sorted_orig_idx = np.argsort(probs)
        sorted_cal = p_cal[sorted_orig_idx]
        diffs = np.diff(sorted_cal)
        assert np.all(diffs >= -1e-6), "Calibrated probabilities violate monotonicity"

        # AUC invariance test
        from sklearn.metrics import roc_auc_score
        auc_orig = roc_auc_score(y_true, probs)
        auc_cal = roc_auc_score(y_true, p_cal)
        assert abs(auc_orig - auc_cal) < 1e-4, f"AUC changed unexpectedly: {auc_orig} vs {auc_cal}"

    def test_probability_clamping(self):
        cal = PlattCalibrator()
        cal.is_fitted = True
        cal.a_ = 1.0
        cal.b_ = 0.0
        cal.calibrator.coef_ = np.array([[1.0]])
        cal.calibrator.intercept_ = np.array([0.0])
        cal.calibrator.classes_ = np.array([0, 1])

        # Test extreme boundary values
        extreme_probs = np.array([0.0, 1.0, 0.5])
        cal_out = cal.predict_proba(extreme_probs)
        assert np.all(cal_out >= 0.0) and np.all(cal_out <= 1.0)
        assert not np.any(np.isnan(cal_out))
        assert not np.any(np.isinf(cal_out))


class TestMetricCalculations:
    """Verifies that metric calculations handle perfect, random, and edge-case predictions."""

    def test_perfect_prediction_metrics(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        m = evaluate_probs(y_true, y_prob, threshold=0.5)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0
        assert m["roc_auc"] == 1.0
        assert m["brier_score"] < 0.05

    def test_confusion_matrix_shape(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.6, 0.4, 0.3, 0.2])
        m = evaluate_probs(y_true, y_prob, threshold=0.5)
        cm = m["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2
        # cm is [[TN, FP], [FN, TP]]
        assert cm[0][0] == 2  # true 0, pred 0
        assert cm[0][1] == 0  # true 0, pred 1
        assert cm[1][0] == 1  # true 1, pred 0
        assert cm[1][1] == 1  # true 1, pred 1

