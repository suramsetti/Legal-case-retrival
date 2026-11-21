"""
Unit tests for Phase 6B Hybrid Retrieval Fusion and Learning-to-Rank modules:
- RRFFusion
- WeightedLinearFusion
- LightGBMRanker
- XGBoostRanker
- LeakageSafeLogisticStacking
- HybridEvaluator
"""

import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from src.evaluation.hybrid_evaluator import HybridEvaluator
from src.ranking.rrf_fusion import RRFFusion
from src.ranking.weighted_fusion import WeightedLinearFusion
from src.ranking.lightgbm_ranker import LightGBMRanker
from src.ranking.xgboost_ranker import XGBoostRanker
from src.ranking.logistic_stacking import LeakageSafeLogisticStacking


class TestHybridEvaluator(unittest.TestCase):
    """Tests evaluation metrics calculation and zero-positive safety."""

    def setUp(self):
        # 2 queries: q1 has 2 positives at rank 1 and 3; q2 has 1 positive at rank 2
        self.df = pd.DataFrame({
            "query_id": ["q1", "q1", "q1", "q1", "q2", "q2", "q2", "q3"],
            "candidate_id": ["c1", "c2", "c3", "c4", "c1", "c2", "c3", "c1"],
            "score": [10.0, 8.0, 5.0, 2.0, 9.0, 7.0, 3.0, 5.0],
            "label": [1, 0, 1, 0, 0, 1, 0, 0],  # q3 has 0 positives
        })

    def test_evaluate_predictions_basic(self):
        metrics = HybridEvaluator.evaluate_predictions(self.df, score_col="score")
        self.assertEqual(metrics["evaluated_queries"], 2)  # q3 skipped (0 positives)
        self.assertIn("P@1", metrics)
        self.assertIn("P@5", metrics)
        self.assertIn("MRR", metrics)
        self.assertIn("nDCG@5", metrics)
        self.assertIn("nDCG@10", metrics)

        # q1: rank 1 is rel (1), rank 2 (0), rank 3 (1). P@1=1.0, MRR=1.0
        # q2: rank 1 is non-rel (0), rank 2 (1). P@1=0.0, MRR=0.5
        # Avg P@1 = (1.0 + 0.0) / 2 = 0.5
        self.assertAlmostEqual(metrics["P@1"], 0.5, places=2)
        # Avg MRR = (1.0 + 0.5) / 2 = 0.75
        self.assertAlmostEqual(metrics["MRR"], 0.75, places=2)

    def test_dcg_and_ndcg_bounds(self):
        # Perfect ranking: [1, 1, 0, 0]
        ndcg_perfect = HybridEvaluator.ndcg_at_k([1, 1, 0, 0], k=4, num_positives=2)
        self.assertAlmostEqual(ndcg_perfect, 1.0, places=4)

        # Worst ranking: [0, 0, 1, 1]
        ndcg_worst = HybridEvaluator.ndcg_at_k([0, 0, 1, 1], k=4, num_positives=2)
        self.assertLess(ndcg_worst, 1.0)
        self.assertGreater(ndcg_worst, 0.0)

        # 0 positives
        ndcg_zero = HybridEvaluator.ndcg_at_k([0, 0, 0], k=3, num_positives=0)
        self.assertEqual(ndcg_zero, 0.0)


class TestRRFFusion(unittest.TestCase):
    """Tests Reciprocal Rank Fusion implementation."""

    def setUp(self):
        self.df = pd.DataFrame({
            "query_id": ["q1", "q1", "q1"],
            "candidate_id": ["c1", "c2", "c3"],
            "bm25_rank": [1, 2, 3],
            "tfidf_rank": [2, 1, 3],
            "inlegalbert_rank": [3, 2, 1],
            "label": [1, 0, 0],
        })

    def test_rrf_formula_computation(self):
        fusion = RRFFusion(k=60, rank_cols=["bm25_rank", "tfidf_rank", "inlegalbert_rank"])
        scores = fusion.compute_scores(self.df)
        self.assertEqual(len(scores), 3)

        # Expected for c1: 1/(60+1) + 1/(60+2) + 1/(60+3)
        expected_c1 = 1.0 / 61.0 + 1.0 / 62.0 + 1.0 / 63.0
        self.assertAlmostEqual(scores[0], expected_c1, places=5)

    def test_rrf_k_grid_evaluation(self):
        results = RRFFusion.evaluate_k_grid(
            self.df,
            k_values=[20, 60],
            rank_cols=["bm25_rank", "tfidf_rank", "inlegalbert_rank"],
        )
        self.assertIn(20, results)
        self.assertIn(60, results)
        self.assertIn("MRR", results[60])

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            RRFFusion(k=-5, rank_cols=["bm25_rank"])


class TestWeightedLinearFusion(unittest.TestCase):
    """Tests Weighted Linear Fusion and weight grid search."""

    def setUp(self):
        np.random.seed(42)
        n = 30
        self.df = pd.DataFrame({
            "query_id": ["q1"] * 10 + ["q2"] * 10 + ["q3"] * 10,
            "candidate_id": [f"c{i}" for i in range(10)] * 3,
            "f1": np.random.uniform(0, 10, n),
            "f2": np.random.uniform(0, 5, n),
            "label": ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 3),
        })

    def test_weight_validation(self):
        # Invalid sum != 1.0
        with self.assertRaises(ValueError):
            WeightedLinearFusion(weights=[0.3, 0.3], feature_cols=["f1", "f2"])
        # Negative weight
        with self.assertRaises(ValueError):
            WeightedLinearFusion(weights=[-0.2, 1.2], feature_cols=["f1", "f2"])
        # Valid weights
        fusion = WeightedLinearFusion(weights=[0.4, 0.6], feature_cols=["f1", "f2"])
        scores = fusion.compute_scores(self.df)
        self.assertEqual(len(scores), len(self.df))

    def test_fit_grid_search_and_save(self):
        fusion = WeightedLinearFusion(feature_cols=["f1", "f2"], use_query_min_max=True)
        res = fusion.fit_grid_search(self.df, step=0.1)
        self.assertIn("best_weights", res)
        self.assertAlmostEqual(sum(res["best_weights"]), 1.0, places=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            fusion.save(path)
            loaded = WeightedLinearFusion.load(path)
            self.assertEqual(len(loaded.weights), 2)
            np.testing.assert_allclose(fusion.weights, loaded.weights)


class TestLightGBMRanker(unittest.TestCase):
    """Tests LightGBM LambdaMART ranker."""

    def setUp(self):
        np.random.seed(42)
        rows = []
        for q in range(15):
            qid = f"query_{q}"
            for c in range(8):
                is_pos = 1 if c == 0 else 0
                rows.append({
                    "query_id": qid,
                    "candidate_id": f"cand_{c}",
                    "feat_a": float(is_pos * 2.0 + np.random.randn() * 0.5),
                    "feat_b": float(is_pos * 1.5 + np.random.randn() * 0.5),
                    "label": is_pos,
                })
        self.df = pd.DataFrame(rows)

    def test_fit_predict_save(self):
        ranker = LightGBMRanker(
            feature_cols=["feat_a", "feat_b"],
            n_estimators=10,
            num_leaves=7,
            random_state=42,
        )
        ranker.fit(self.df)
        scores = ranker.predict(self.df)
        self.assertEqual(len(scores), len(self.df))

        metrics = ranker.evaluate(self.df)
        self.assertIn("nDCG@10", metrics)

        importances = ranker.get_feature_importances()
        self.assertEqual(len(importances), 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lgbm_model.txt"
            ranker.save_model(path)
            self.assertTrue(path.exists())


class TestXGBoostRanker(unittest.TestCase):
    """Tests XGBoost Ranker."""

    def setUp(self):
        np.random.seed(42)
        rows = []
        for q in range(15):
            qid = f"query_{q}"
            for c in range(8):
                is_pos = 1 if c == 0 else 0
                rows.append({
                    "query_id": qid,
                    "candidate_id": f"cand_{c}",
                    "feat_a": float(is_pos * 2.0 + np.random.randn() * 0.5),
                    "feat_b": float(is_pos * 1.5 + np.random.randn() * 0.5),
                    "label": is_pos,
                })
        self.df = pd.DataFrame(rows)

    def test_fit_predict_save(self):
        ranker = XGBoostRanker(
            feature_cols=["feat_a", "feat_b"],
            n_estimators=10,
            max_depth=3,
            random_state=42,
        )
        ranker.fit(self.df)
        scores = ranker.predict(self.df)
        self.assertEqual(len(scores), len(self.df))

        metrics = ranker.evaluate(self.df)
        self.assertIn("MRR", metrics)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "xgb_model.json"
            ranker.save_model(path)
            self.assertTrue(path.exists())


class TestLeakageSafeLogisticStacking(unittest.TestCase):
    """Tests GroupKFold stacking meta-learner."""

    def setUp(self):
        np.random.seed(42)
        rows = []
        for q in range(20):
            qid = f"query_{q}"
            for c in range(6):
                is_pos = 1 if c == 0 else 0
                rows.append({
                    "query_id": qid,
                    "candidate_id": f"cand_{c}",
                    "lex1": float(is_pos * 2.0 + np.random.randn() * 0.2),
                    "dense1": float(is_pos * 1.0 + np.random.randn() * 0.3),
                    "legal1": float(is_pos * 1.5 + np.random.randn() * 0.2),
                    "label": is_pos,
                })
        self.df = pd.DataFrame(rows)

    def test_fit_predict_zero_leakage(self):
        stacker = LeakageSafeLogisticStacking(
            n_splits=4,
            lexical_cols=["lex1"],
            dense_cols=["dense1"],
            legal_cols=["legal1"],
            random_state=42,
        )
        stacker.fit(self.df)

        # OOF shape matches train df length and 3 modalities
        self.assertEqual(stacker.oof_predictions.shape, (len(self.df), 3))
        self.assertIsNotNone(stacker.meta_coefficients)

        scores = stacker.predict(self.df)
        self.assertEqual(len(scores), len(self.df))
        self.assertTrue(np.all(scores >= 0.0) and np.all(scores <= 1.0))

        metrics = stacker.evaluate(self.df)
        self.assertGreater(metrics["P@1"], 0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stacking.joblib"
            stacker.save(path)
            loaded = LeakageSafeLogisticStacking.load(path)
            loaded_scores = loaded.predict(self.df)
            np.testing.assert_allclose(scores, loaded_scores)


if __name__ == "__main__":
    unittest.main()

