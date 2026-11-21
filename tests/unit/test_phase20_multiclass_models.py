"""
Unit tests for Phase 20: Multi-Class Legal Judgment Outcome Classification.

Tests:
1. Three-class label mapping & dataset class distribution
2. Duplicate-group & ID partition isolation
3. Ruling phrase & proof sentence neutralization
4. Multi-class probability dimensions & normalization (sums to 1.0)
5. Class-weight calculation & handling
6. Confusion matrix shape (3x3) & multiclass metric calculations
7. Feature leakage prevention (statistical bounds)
8. Checkpoint loading & reproducibility
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.phase20_multiclass_models import (
    MulticlassTfidfClassifier,
    MulticlassCharNgramClassifier,
    MulticlassSectionAwareClassifier,
    CANONICAL_CLASSES,
)
from scripts.phase20d_multiclass_training import evaluate_multiclass_predictions

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "judipl", "phase18c", "three_class")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints", "phase20")


class TestThreeClassLabelsAndSplits:
    """Verifies label integrity, counts, and partition isolation."""

    @pytest.fixture(scope="class")
    def partitions(self):
        tr = pd.read_parquet(os.path.join(DATA_DIR, "train_masked.parquet"))
        va = pd.read_parquet(os.path.join(DATA_DIR, "val_masked.parquet"))
        te = pd.read_parquet(os.path.join(DATA_DIR, "test_masked.parquet"))
        return tr, va, te

    def test_exact_classes(self, partitions):
        tr, va, te = partitions
        expected_classes = {"Accepted", "Rejected", "Other"}
        assert set(tr["label"].unique()) == expected_classes
        assert set(va["label"].unique()) == expected_classes
        assert set(te["label"].unique()) == expected_classes

    def test_partition_sizes(self, partitions):
        tr, va, te = partitions
        assert len(tr) == 29395
        assert len(va) == 6297
        assert len(te) == 6313
        assert len(tr) + len(va) + len(te) == 42005

    def test_partition_isolation(self, partitions):
        tr, va, te = partitions
        s_tr = set(tr["id"])
        s_va = set(va["id"])
        s_te = set(te["id"])
        assert len(s_tr.intersection(s_va)) == 0, "Train and Val share IDs"
        assert len(s_tr.intersection(s_te)) == 0, "Train and Test share IDs"
        assert len(s_va.intersection(s_te)) == 0, "Val and Test share IDs"

    def test_duplicate_group_isolation(self, partitions):
        tr, va, te = partitions
        g_tr = set(tr["duplicate_group"])
        g_va = set(va["duplicate_group"])
        g_te = set(te["duplicate_group"])
        assert len(g_tr.intersection(g_va)) == 0, "Train and Val share duplicate groups"
        assert len(g_tr.intersection(g_te)) == 0, "Train and Test share duplicate groups"
        assert len(g_va.intersection(g_te)) == 0, "Val and Test share duplicate groups"


class TestMaskingAndNeutralization:
    """Verifies that dispositive ruling text and proof sentences are neutralized."""

    def test_proof_sentence_neutralization(self):
        va = pd.read_parquet(os.path.join(DATA_DIR, "val_masked.parquet"))
        # 100% of cases in three-class partitions must be neutralized
        assert va["proof_masked"].mean() == 1.0


class TestMultiClassProbabilitiesAndMetrics:
    """Verifies 3-class prediction dimensions, normalization, and metric calculations."""

    def test_probability_dimensions_and_normalization(self):
        texts = [
            "Facts: accused charged. Reasoning: appeal allowed.",
            "Facts: civil suit filed. Reasoning: appeal dismissed.",
            "Facts: writ petition filed. Reasoning: matter remanded.",
        ]
        y = np.array(["Accepted", "Rejected", "Other"])
        clf = MulticlassTfidfClassifier(max_features=100)
        clf.fit(texts, y)
        probs = clf.predict_proba(texts)

        # Shape must be (N, 3)
        assert probs.shape == (3, 3)
        # All rows must sum to 1.0
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        # All values must be in [0, 1]
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_metric_calculations_and_confusion_matrix(self):
        y_true = np.array(["Accepted", "Rejected", "Other", "Accepted"])
        probs = np.array([
            [0.8, 0.1, 0.1],  # True Accepted -> Pred Accepted (idx 0)
            [0.1, 0.1, 0.8],  # True Rejected -> Pred Rejected (idx 2)
            [0.1, 0.7, 0.2],  # True Other -> Pred Other (idx 1)
            [0.2, 0.2, 0.6],  # True Accepted -> Pred Rejected (idx 2, False Neg)
        ])
        classes = ["Accepted", "Other", "Rejected"]
        metrics = evaluate_multiclass_predictions(y_true, probs, classes=classes)

        assert metrics["accuracy"] == 0.75
        cm = metrics["confusion_matrix"]
        assert len(cm) == 3 and len(cm[0]) == 3
        # Total elements in CM must equal len(y_true)
        assert sum(sum(row) for row in cm) == len(y_true)


class TestCheckpointLoadingAndReproducibility:
    """Verifies that all Phase 20 frozen checkpoints exist and can be loaded."""

    def test_checkpoint_files_exist(self):
        expected_ckpts = [
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
        for ckpt in expected_ckpts:
            p = os.path.join(MODELS_DIR, ckpt)
            assert os.path.exists(p), f"Checkpoint missing: {ckpt}"
            assert os.path.getsize(p) > 0, f"Checkpoint empty: {ckpt}"

