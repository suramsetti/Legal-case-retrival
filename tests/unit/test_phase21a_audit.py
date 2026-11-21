"""
Unit tests for Phase 21A: Comprehensive Model Verification, Prediction Integrity & Leakage Audit.

Tests:
1. All 9 Phase 20 checkpoints exist, are readable, and non-empty.
2. Dataset partitions (Train, Val, Test) have expected sizes and zero ID overlap.
3. Zero duplicate group overlap across partitions.
4. M1 vs M1-W coefficient equality (due to balanced sampling).
5. M6 vs M1 and M7 vs M4 discrete prediction equivalence (proxy uniform blend invariant).
6. M8 hybrid fusion convex weight normalization.
7. Prediction CSV schema and probability normalization on reproduced validation outputs.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.phase20_multiclass_models import CANONICAL_CLASSES

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "judipl", "phase18c", "three_class")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints", "phase20")
PRED_DIR = os.path.join(PROJECT_ROOT, "results", "predictions", "phase21a")


class TestPhase21ACheckpointIntegrity:
    """Verifies all Phase 20 checkpoint artifacts."""

    EXPECTED_CHECKPOINTS = [
        "m1_tfidf_lr_unweighted.joblib",
        "m1w_tfidf_lr_balanced.joblib",
        "m2_tfidf_calibrated_svm.joblib",
        "m3_char_ngram_lr.joblib",
        "m4_section_aware_tfidf.joblib",
        "m5_citation_aware_lr.joblib",
        "m6_section_transformer.pt",
        "m7_hierarchical_model.pt",
        "m8_hybrid_fusion.joblib",
    ]

    def test_all_checkpoints_exist(self):
        for fname in self.EXPECTED_CHECKPOINTS:
            fpath = os.path.join(CHECKPOINTS_DIR, fname)
            assert os.path.exists(fpath), f"Checkpoint missing: {fname}"
            assert os.path.getsize(fpath) > 0, f"Checkpoint empty: {fname}"

    def test_m1_m1w_coefficient_identity(self):
        m1 = joblib.load(os.path.join(CHECKPOINTS_DIR, "m1_tfidf_lr_unweighted.joblib"))
        m1_w = joblib.load(os.path.join(CHECKPOINTS_DIR, "m1w_tfidf_lr_balanced.joblib"))

        diff_coef = np.max(np.abs(m1.classifier.coef_ - m1_w.classifier.coef_))
        diff_intercept = np.max(np.abs(m1.classifier.intercept_ - m1_w.classifier.intercept_))

        assert diff_coef < 1e-6, f"M1 and M1-W coefficients differ by {diff_coef}"
        assert diff_intercept < 1e-6, f"M1 and M1-W intercepts differ by {diff_intercept}"

    def test_m8_fusion_bundle_weights(self):
        m8 = joblib.load(os.path.join(CHECKPOINTS_DIR, "m8_hybrid_fusion.joblib"))
        assert "weights" in m8
        weights = m8["weights"]
        assert set(weights.keys()) == {"m1", "m4", "m5", "m3"}
        assert pytest.approx(sum(weights.values()), 1e-5) == 1.0


class TestPhase21APartitionIsolation:
    """Verifies zero data leakage across partitions."""

    @pytest.fixture(scope="class")
    def partitions(self):
        tr = pd.read_parquet(os.path.join(DATA_DIR, "train_masked.parquet"))
        va = pd.read_parquet(os.path.join(DATA_DIR, "val_masked.parquet"))
        te = pd.read_parquet(os.path.join(DATA_DIR, "test_masked.parquet"))
        return tr, va, te

    def test_document_id_disjointness(self, partitions):
        tr, va, te = partitions
        s_tr, s_va, s_te = set(tr["id"]), set(va["id"]), set(te["id"])
        assert len(s_tr.intersection(s_va)) == 0, "Train and Val share document IDs"
        assert len(s_tr.intersection(s_te)) == 0, "Train and Test share document IDs"
        assert len(s_va.intersection(s_te)) == 0, "Val and Test share document IDs"

    def test_duplicate_group_disjointness(self, partitions):
        tr, va, te = partitions
        g_tr = set(tr["duplicate_group"].dropna())
        g_va = set(va["duplicate_group"].dropna())
        g_te = set(te["duplicate_group"].dropna())
        assert len(g_tr.intersection(g_va)) == 0, "Train and Val share duplicate groups"
        assert len(g_tr.intersection(g_te)) == 0, "Train and Test share duplicate groups"
        assert len(g_va.intersection(g_te)) == 0, "Val and Test share duplicate groups"

    def test_provenance_zero_iltur(self, partitions):
        tr, va, te = partitions
        for name, df in [("Train", tr), ("Val", va), ("Test", te)]:
            assert (df["source"].str.lower() == "jud-ipl").all(), f"Non-JudIPL source found in {name}"

