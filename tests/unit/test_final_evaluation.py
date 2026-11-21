"""
Unit tests for Phase 8 final evaluation routines and metrics verification.
"""

import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_final_test_evaluation import (
    compute_query_level_metrics,
    run_paired_bootstrap,
)
from src.evaluation.hybrid_evaluator import HybridEvaluator


class TestFinalEvaluation(unittest.TestCase):
    """Test suite for Phase 8 final evaluation procedures and bootstrap analysis."""

    def setUp(self):
        # Create synthetic evaluation dataset with 3 queries (1 zero-relevance)
        self.df = pd.DataFrame([
            {"query_id": "q1", "candidate_id": "c1", "score": 0.9, "label": 1},
            {"query_id": "q1", "candidate_id": "c2", "score": 0.8, "label": 0},
            {"query_id": "q1", "candidate_id": "c3", "score": 0.1, "label": 0},
            {"query_id": "q2", "candidate_id": "c1", "score": 0.2, "label": 0},
            {"query_id": "q2", "candidate_id": "c2", "score": 0.7, "label": 1},
            {"query_id": "q2", "candidate_id": "c3", "score": 0.4, "label": 0},
            # Zero-relevance query:
            {"query_id": "q3", "candidate_id": "c1", "score": 0.6, "label": 0},
            {"query_id": "q3", "candidate_id": "c2", "score": 0.5, "label": 0},
        ])

    def test_zero_relevance_exclusion_in_query_metrics(self):
        """Verifies zero-relevance queries are strictly excluded from query-level metrics."""
        qm = compute_query_level_metrics(self.df, score_col="score")
        self.assertIn("q1", qm)
        self.assertIn("q2", qm)
        self.assertNotIn("q3", qm)  # q3 has 0 positives, must be excluded

        # Check values
        self.assertEqual(qm["q1"]["P@1"], 1.0)
        self.assertEqual(qm["q1"]["MRR"], 1.0)
        self.assertEqual(qm["q2"]["P@1"], 1.0)  # c2 has score 0.7 and label 1, so rank 1
        self.assertEqual(qm["q2"]["MRR"], 1.0)

    def test_paired_bootstrap_computation(self):
        """Tests that paired bootstrap returns valid confidence intervals and p-values."""
        # Setup synthetic model A and model B per-query metrics
        metrics_a = {
            f"q{i}": {"MRR": 0.5 + 0.1 * (i % 3), "nDCG@10": 0.4 + 0.05 * (i % 3)}
            for i in range(50)
        }
        # Model B is strictly better by 0.15
        metrics_b = {
            f"q{i}": {"MRR": metrics_a[f"q{i}"]["MRR"] + 0.15, "nDCG@10": metrics_a[f"q{i}"]["nDCG@10"] + 0.12}
            for i in range(50)
        }

        boot_res = run_paired_bootstrap(
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            metric_names=["MRR", "nDCG@10"],
            n_resamples=500,
            random_state=42,
        )

        self.assertIn("MRR", boot_res)
        self.assertAlmostEqual(boot_res["MRR"]["mean_diff"], 0.15, places=4)
        self.assertGreater(boot_res["MRR"]["ci_lower"], 0.0)
        self.assertTrue(boot_res["MRR"]["is_significant"])
        self.assertLess(boot_res["MRR"]["p_value"], 0.05)

    def test_final_metrics_json_integrity(self):
        """Verifies structure and key metrics in final_test_metrics.json."""
        metrics_path = PROJECT_ROOT / "results/evaluation/final_test_metrics.json"
        self.assertTrue(metrics_path.exists())

        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data["metadata"]["dataset_configuration"]
        self.assertEqual(meta["total_queries"], 237)
        self.assertEqual(meta["eligible_queries"], 234)
        self.assertEqual(len(meta["zero_relevance_queries"]), 3)
        self.assertEqual(meta["total_unique_candidates"], 1727)
        self.assertEqual(meta["total_candidate_pairs"], 409062)
        self.assertEqual(meta["candidate_pool_per_query"], 1726)

        # Check that Weighted Linear Fusion beats BM25 and InLegalBERT in MRR
        models = data["model_comparison"]
        self.assertIn("Weighted_Linear_Fusion", models)
        self.assertGreater(models["Weighted_Linear_Fusion"]["MRR"], models["BM25_Baseline"]["MRR"])
        self.assertGreater(models["Weighted_Linear_Fusion"]["MRR"], models["InLegalBERT_Baseline"]["MRR"])


if __name__ == "__main__":
    unittest.main()
