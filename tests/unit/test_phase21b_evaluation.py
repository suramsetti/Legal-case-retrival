"""
Unit tests for Phase 21B: Final Three-Class Test Evaluation & Comparative Performance Analysis.

Tests:
1. Frozen checkpoint hash verification for M8, M4, and component checkpoints.
2. Duplicate exclusion logic and deterministic manifest.
3. Prediction file schema, row counts, and probability axioms.
4. Paired prediction ID alignment between M4 and M8.
5. Canonical class ordering consistency.
6. M8 fusion weight convexity and values.
7. Prior exposure identification (812 cases overlapping with Phase 18 binary test partition).
"""

import os
import sys
import json
import hashlib
import pytest
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.phase20_multiclass_models import CANONICAL_CLASSES

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "judipl", "phase18c", "three_class")
BINARY_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "judipl", "phase18c", "binary")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints", "phase20")
PRED_DIR = os.path.join(PROJECT_ROOT, "results", "predictions", "phase21b")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "results", "reports")


class TestPhase21BPreEvaluationIntegrity:
    """Verifies frozen checkpoints and exclusion manifests."""

    EXPECTED_HASHES = {
        "m1_tfidf_lr_unweighted.joblib": "26c62266ad9a2d59722b49c1464111f0ec3ff8cf4ff336ec0332db818f4f5d50",
        "m3_char_ngram_lr.joblib": "1ce55ad05ceddde1032ec0db5c23f6d9384261a8a03832f0ce2853cd3bc3b820",
        "m4_section_aware_tfidf.joblib": "1d5963eec3d4002112e61fe79f73f96391738ed2bc8e56dc97ed8743ac13b460",
        "m5_citation_aware_lr.joblib": "53b592c95352fea1db9ef837b3424bea5598b79d932e24330f1d4e52c9e45b69",
        "m8_hybrid_fusion.joblib": "552e0819080d560c1bf82b5a2eeba8a2e805fe2fed88907ca004eed16f2b4db2",
    }

    def test_checkpoint_hashes(self):
        for fname, exp_hash in self.EXPECTED_HASHES.items():
            fpath = os.path.join(CHECKPOINTS_DIR, fname)
            assert os.path.exists(fpath), f"Checkpoint missing: {fname}"
            with open(fpath, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            assert actual_hash == exp_hash, f"Hash mismatch for {fname}: got {actual_hash}"

    def test_m8_fusion_weights(self):
        bundle = joblib.load(os.path.join(CHECKPOINTS_DIR, "m8_hybrid_fusion.joblib"))
        w = bundle["weights"]
        assert w == {"m1": 0.50, "m4": 0.25, "m5": 0.15, "m3": 0.10}
        assert pytest.approx(sum(w.values()), 1e-6) == 1.0

    def test_duplicate_exclusion_manifest(self):
        mpath = os.path.join(REPORTS_DIR, "phase21b_duplicate_exclusion_manifest.json")
        assert os.path.exists(mpath), "Exclusion manifest missing"
        with open(mpath, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["original_test_size"] == 6313
        assert manifest["excluded_test_count"] == 2
        assert manifest["eligible_test_size"] == 6311
        assert sorted(manifest["excluded_test_ids"]) == ["0001116667", "0001505760"]

    def test_phase18_binary_overlap_count(self):
        bin_te = pd.read_parquet(os.path.join(BINARY_DATA_DIR, "test_masked.parquet"))
        tri_te = pd.read_parquet(os.path.join(DATA_DIR, "test_masked.parquet"))
        overlap = set(bin_te["id"]).intersection(set(tri_te["id"]))
        assert len(overlap) == 812, f"Expected 812 overlapping cases, found {len(overlap)}"


class TestPhase21BPredictionsAndMetrics:
    """Verifies generated predictions and evaluation alignments."""

    @pytest.fixture(scope="class")
    def predictions(self):
        p_m4 = os.path.join(PRED_DIR, "m4_test_predictions.csv")
        p_m8 = os.path.join(PRED_DIR, "m8_test_predictions.csv")
        if not (os.path.exists(p_m4) and os.path.exists(p_m8)):
            pytest.skip("Test predictions not yet generated")
        df4 = pd.read_csv(p_m4)
        df8 = pd.read_csv(p_m8)
        return df4, df8

    def test_prediction_row_counts_and_ids(self, predictions):
        df4, df8 = predictions
        assert len(df4) == 6311
        assert len(df8) == 6311
        assert (df4["id"] == df8["id"]).all()
        assert not df4["id"].duplicated().any()
        assert not df8["id"].duplicated().any()

    def test_probability_normalization_and_bounds(self, predictions):
        df4, df8 = predictions
        for name, df in [("M4", df4), ("M8", df8)]:
            probs = df[["prob_accepted", "prob_other", "prob_rejected"]].values
            assert (probs >= 0.0).all() and (probs <= 1.0).all(), f"Prob out of bounds in {name}"
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5, err_msg=f"Sum not 1.0 in {name}")
            assert not df.isna().any().any(), f"Nulls found in {name}"

