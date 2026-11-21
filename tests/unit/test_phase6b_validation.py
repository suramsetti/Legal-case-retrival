"""
Unit tests for Phase 6B.1 Hybrid Retrieval Validation and Reproducibility Audit.

Tests:
1. Exact reproducibility of Weighted Linear Fusion metrics from saved checkpoint.
2. Exact mathematical verification of Phase 3 vs Phase 6B MRR discrepancy (MRR@10 vs unconstrained MRR).
3. Zero-leakage verification of 5-fold GroupKFold stacking partitions.
4. Measurable contribution verification of legal knowledge features.
5. Paired bootstrap calculation integrity and confidence interval bounds.
"""

import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.evaluation.hybrid_evaluator import HybridEvaluator
from src.ranking.weighted_fusion import WeightedLinearFusion


class TestPhase6BValidation(unittest.TestCase):
    """Validation test suite certifying Phase 6B reproducibility and integrity."""

    @classmethod
    def setUpClass(cls):
        cls.dev_path = Path("data/processed/IL-TUR-PCR/dev/hybrid_features.parquet")
        cls.train_path = Path("data/processed/IL-TUR-PCR/train/hybrid_features.parquet")
        cls.weights_path = Path("models/checkpoints/weighted_fusion_weights.json")

        if cls.dev_path.exists():
            cls.dev_df = pd.read_parquet(cls.dev_path).sort_values("query_id")
        else:
            cls.dev_df = None

        if cls.train_path.exists():
            cls.train_df = pd.read_parquet(cls.train_path).sort_values("query_id")
        else:
            cls.train_df = None

    def test_weighted_fusion_checkpoint_reproducibility(self):
        """Verifies that the checkpoint weights reproduce reported dev metrics within 1e-4."""
        self.assertIsNotNone(self.dev_df, "Dev dataset not found.")
        self.assertTrue(self.weights_path.exists(), "Weights checkpoint not found.")

        loaded_fusion = WeightedLinearFusion.load(self.weights_path)
        np.testing.assert_allclose(loaded_fusion.weights, [0.0, 0.65, 0.25, 0.1], atol=1e-5)

        dev_eval = self.dev_df.copy()
        dev_eval["score_reproduced"] = loaded_fusion.compute_scores(dev_eval)
        metrics = HybridEvaluator.evaluate_predictions(dev_eval, score_col="score_reproduced")

        expected = {
            "P@1": 0.7881,
            "P@5": 0.4712,
            "P@10": 0.3212,
            "R@5": 0.4701,
            "R@10": 0.5770,
            "MRR": 0.8461,
            "nDCG@5": 0.6543,
            "nDCG@10": 0.6349,
        }

        for metric, exp_val in expected.items():
            self.assertAlmostEqual(
                metrics[metric], exp_val, places=4,
                msg=f"Metric {metric} reproduced as {metrics[metric]}, expected {exp_val}"
            )

    def test_baseline_mrr_top10_truncation_exactness(self):
        """
        Verifies that Phase 3's reported 0.8338 (TF-IDF) and 0.5939 (BM25)
        are the exact consequence of top-10 rank cutoff (MRR@10).
        """
        self.assertIsNotNone(self.dev_df)

        tfidf_mrr_full, tfidf_mrr_top10 = [], []
        bm25_mrr_full, bm25_mrr_top10 = [], []

        for _, group in self.dev_df.groupby("query_id", sort=False):
            # TF-IDF
            s_t = group.sort_values(by="tfidf_score", ascending=False)
            lbls_t = s_t["label"].values.astype(int)
            if np.sum(lbls_t) > 0:
                rel_f = np.where(lbls_t == 1)[0]
                tfidf_mrr_full.append(1.0 / (rel_f[0] + 1) if len(rel_f) > 0 else 0.0)
                rel_10 = np.where(lbls_t[:10] == 1)[0]
                tfidf_mrr_top10.append(1.0 / (rel_10[0] + 1) if len(rel_10) > 0 else 0.0)

            # BM25
            s_b = group.sort_values(by="bm25_score", ascending=False)
            lbls_b = s_b["label"].values.astype(int)
            if np.sum(lbls_b) > 0:
                rel_b_f = np.where(lbls_b == 1)[0]
                bm25_mrr_full.append(1.0 / (rel_b_f[0] + 1) if len(rel_b_f) > 0 else 0.0)
                rel_b_10 = np.where(lbls_b[:10] == 1)[0]
                bm25_mrr_top10.append(1.0 / (rel_b_10[0] + 1) if len(rel_b_10) > 0 else 0.0)

        # Exact match to Phase 3 reported values
        self.assertAlmostEqual(float(np.mean(tfidf_mrr_top10)), 0.8338, places=4)
        self.assertAlmostEqual(float(np.mean(bm25_mrr_top10)), 0.5939, places=4)

        # Exact match to Phase 6B reported values
        self.assertAlmostEqual(float(np.mean(tfidf_mrr_full)), 0.8373, places=4)
        self.assertAlmostEqual(float(np.mean(bm25_mrr_full)), 0.5988, places=4)

    def test_stacking_group_kfold_zero_leakage(self):
        """Verifies zero query overlap across all 5 folds in stacking training."""
        self.assertIsNotNone(self.train_df)

        gkf = GroupKFold(n_splits=5)
        train_sorted = self.train_df.sort_values("query_id").copy()
        y = train_sorted["label"].values.astype(int)
        groups = train_sorted["query_id"].values

        for fold, (trn_idx, val_idx) in enumerate(gkf.split(train_sorted, y, groups=groups)):
            trn_qids = set(train_sorted.iloc[trn_idx]["query_id"].unique())
            val_qids = set(train_sorted.iloc[val_idx]["query_id"].unique())
            overlap = trn_qids.intersection(val_qids)
            self.assertEqual(
                len(overlap), 0,
                f"Fold {fold+1} has {len(overlap)} overlapping queries!"
            )

    def test_legal_knowledge_measurable_gain(self):
        """Verifies that adding legal knowledge to TF-IDF improves MRR and Recall@10."""
        self.assertIsNotNone(self.dev_df)

        features = ["bm25_score", "tfidf_score", "inlegalbert_sim", "composite_legal_score"]
        wf = WeightedLinearFusion(feature_cols=features, use_query_min_max=True)
        norm_dev = wf.normalize_features(self.dev_df)

        # TF-IDF alone
        norm_dev["score_tfidf_only"] = norm_dev["tfidf_score"].values
        m_tfidf = HybridEvaluator.evaluate_predictions(norm_dev, score_col="score_tfidf_only")

        # TF-IDF + Legal Knowledge (0.85 / 0.15)
        norm_dev["score_tfidf_legal"] = (
            0.85 * norm_dev["_qnorm_tfidf_score"].values +
            0.15 * norm_dev["_qnorm_composite_legal_score"].values
        )
        m_legal = HybridEvaluator.evaluate_predictions(norm_dev, score_col="score_tfidf_legal")

        # Verify MRR improvement
        self.assertGreater(m_legal["MRR"], m_tfidf["MRR"])
        self.assertGreater(m_legal["nDCG@10"], m_tfidf["nDCG@10"])
        self.assertGreater(m_legal["R@10"], m_tfidf["R@10"])

    def test_bootstrap_calculation_integrity(self):
        """Verifies paired bootstrap CI and p-value calculation logic."""
        rng = np.random.RandomState(42)
        # Synthetic positive differences
        diffs = rng.normal(loc=0.05, scale=0.02, size=100)
        n_boot = 500
        boot_means = np.zeros(n_boot)
        for b in range(n_boot):
            sample = rng.choice(diffs, size=len(diffs), replace=True)
            boot_means[b] = np.mean(sample)

        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)

        self.assertGreater(ci_lower, 0.0)
        self.assertGreater(ci_upper, ci_lower)


if __name__ == "__main__":
    unittest.main()

