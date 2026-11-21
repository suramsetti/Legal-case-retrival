"""
Unit Tests for Phase 10 Audit Logic.

Verifies:
1. No test-label training assertions (training data uses only train split).
2. MRR denominator computation (uses first-relevant-rank, not total positives).
3. P@1 formula (binary hit at rank 1, averaged over queries with ≥1 positive).
4. Alpha tuning only on dev split.
5. Feature normalizer fitted on training data only.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd

from src.evaluation.hybrid_evaluator import HybridEvaluator
from src.ranking.pairwise_reranker import PairwiseNeuralReRanker


# ─── Test fixtures ────────────────────────────────────────────────────────────

def make_df(query_scores: dict) -> pd.DataFrame:
    """Build a minimal evaluator-compatible DataFrame from {qid: [(score, label), ...]}."""
    rows = []
    for qid, pairs in query_scores.items():
        for i, (score, label) in enumerate(pairs):
            rows.append({
                "query_id":    qid,
                "candidate_id": f"c{i}",
                "score":       score,
                "label":       label,
            })
    return pd.DataFrame(rows)


# ─── Test 1: P@1 formula ─────────────────────────────────────────────────────

class TestPrecisionAtOne:
    """P@1 = fraction of queries where rank-1 candidate is relevant."""

    def test_all_hits(self):
        df = make_df({
            "q1": [(1.0, 1), (0.5, 0)],
            "q2": [(0.9, 1), (0.1, 0)],
        })
        m = HybridEvaluator.evaluate_predictions(df, score_col="score")
        assert m["P@1"] == pytest.approx(1.0, abs=1e-6)

    def test_no_hits(self):
        df = make_df({
            "q1": [(1.0, 0), (0.5, 1)],
            "q2": [(0.9, 0), (0.1, 1)],
        })
        m = HybridEvaluator.evaluate_predictions(df, score_col="score")
        assert m["P@1"] == pytest.approx(0.0, abs=1e-6)

    def test_partial_hits(self):
        df = make_df({
            "q1": [(1.0, 1), (0.5, 0)],
            "q2": [(0.9, 0), (0.1, 1)],
        })
        m = HybridEvaluator.evaluate_predictions(df, score_col="score")
        assert m["P@1"] == pytest.approx(0.5, abs=1e-6)

    def test_zero_relevance_query_excluded(self):
        """Queries with 0 positives must be excluded from metric computation."""
        df = make_df({
            "q1": [(1.0, 1), (0.5, 0)],  # eligible
            "q2": [(0.9, 0), (0.1, 0)],  # zero relevance → skipped
        })
        m = HybridEvaluator.evaluate_predictions(df, score_col="score")
        # Only q1 is evaluated
        assert m["evaluated_queries"] == 1
        assert m["P@1"] == pytest.approx(1.0, abs=1e-6)

    def test_p1_is_not_fraction_of_all_candidates(self):
        """P@1 is per-query binary, not per-candidate."""
        df = make_df({
            "q1": [(1.0, 1), (0.8, 1), (0.5, 0)],  # two positives, rank-1 is positive
        })
        m = HybridEvaluator.evaluate_predictions(df, score_col="score")
        # P@1 = 1 (rank-1 is relevant), not 2/3
        assert m["P@1"] == pytest.approx(1.0, abs=1e-6)


# ─── Test 2: MRR denominator ──────────────────────────────────────────────────

class TestMRR:
    """MRR = mean of 1/rank_of_first_relevant over eligible queries."""

    def test_first_rank_relevant(self):
        df = make_df({"q1": [(1.0, 1), (0.5, 0), (0.2, 0)]})
        m = HybridEvaluator.evaluate_predictions(df, score_col="score")
        assert m["MRR"] == pytest.approx(1.0, abs=1e-6)

    def test_second_rank_relevant(self):
        df = make_df({"q1": [(1.0, 0), (0.5, 1), (0.2, 0)]})
        m = HybridEvaluator.evaluate_predictions(df, score_col="score")
        assert m["MRR"] == pytest.approx(0.5, abs=1e-6)

    def test_third_rank_relevant(self):
        df = make_df({"q1": [(1.0, 0), (0.5, 0), (0.2, 1)]})
        m = HybridEvaluator.evaluate_predictions(df, score_col="score")
        # HybridEvaluator rounds to 4 decimal places: 1/3 → 0.3333
        assert m["MRR"] == pytest.approx(1.0 / 3.0, abs=5e-4)

    def test_multiple_queries_average(self):
        """MRR is the mean across queries, not globally the first relevant across all rows."""
        df = make_df({
            "q1": [(1.0, 1), (0.5, 0)],   # MRR = 1.0
            "q2": [(1.0, 0), (0.5, 1)],   # MRR = 0.5
        })
        m = HybridEvaluator.evaluate_predictions(df, score_col="score")
        assert m["MRR"] == pytest.approx(0.75, abs=1e-6)

    def test_mrr_uses_first_relevant_not_total(self):
        """
        If there are 3 positives, MRR is still 1/rank_of_FIRST_relevant,
        not 1/total_positives.
        """
        df = make_df({"q1": [(1.0, 0), (0.9, 1), (0.8, 1), (0.7, 1)]})
        m = HybridEvaluator.evaluate_predictions(df, score_col="score")
        # First relevant at rank 2 → MRR = 0.5, not 1/3
        assert m["MRR"] == pytest.approx(0.5, abs=1e-6)


# ─── Test 3: No test-label training assertions ────────────────────────────────

class TestNoTestLabelTraining:
    """
    The training pipeline (train_pairwise_reranker.py) must never pass test_df to
    reranker.train_epochs() or reranker.prepare_pairwise_training_data() when
    called in the training phase (only train/dev are used).

    We verify this by code inspection (static): check that the training script
    does not use the 'split == test' filter or test_df.label for fitting.
    """

    def test_train_script_does_not_fit_on_test(self):
        """Verify train_pairwise_reranker.py never calls train_epochs with test data."""
        script = PROJECT_ROOT / "scripts/train_pairwise_reranker.py"
        assert script.exists(), "Train script must exist"
        src = script.read_text(encoding="utf-8")

        # The script should not prepare pairwise data from test_df for training
        # (It may load test_df for evaluation, which is fine.)
        # The dangerous pattern would be passing test_df to prepare_pairwise_training_data
        # and then feeding those pairs to train_epochs.
        # Static check: prepare_pairwise_training_data is called with train_df and dev_df only.
        assert "prepare_pairwise_training_data(test_df" not in src, (
            "test_df must never be passed to prepare_pairwise_training_data in the training script"
        )

    def test_train_script_fits_normalizer_on_train_only(self):
        """fit_normalizer is called inside train_epochs on concatenation of pos_train+neg_train."""
        script = PROJECT_ROOT / "scripts/train_pairwise_reranker.py"
        src = script.read_text(encoding="utf-8")
        # train_epochs internally calls fit_normalizer on the provided arrays.
        # The script passes pos_train, neg_train (derived from train_df alone).
        # Verify the script does NOT call fit_normalizer directly on test data.
        assert "fit_normalizer(test" not in src and "fit_normalizer(pos_test" not in src, (
            "Normalizer must not be fitted on test features"
        )

    def test_alpha_tuned_on_dev_not_test(self):
        """Alpha grid search must iterate over dev_reranked, not test_reranked."""
        script = PROJECT_ROOT / "scripts/train_pairwise_reranker.py"
        src = script.read_text(encoding="utf-8")
        # The alpha loop should call evaluate_predictions on dev, not test
        # Check that best_alpha selection block references dev_df
        assert "rerank_query_pool(dev_df" in src, (
            "Alpha must be tuned on dev_df"
        )

    def test_no_split_test_filter_in_feature_builder(self):
        """feature_dataset_builder.py must not filter rows by split=='test' during training."""
        builder_src = (PROJECT_ROOT / "src/ranking/feature_dataset_builder.py").read_text(encoding="utf-8")
        # The builder tags rows with split name but does not filter by it
        assert "split == 'test'" not in builder_src and "split==\"test\"" not in builder_src, (
            "feature_dataset_builder must not filter on test split internally"
        )

    def test_pairwise_training_uses_only_label_column(self):
        """prepare_pairwise_training_data uses df['label'] == 1 / == 0, not derived from test qids."""
        reranker_src = (PROJECT_ROOT / "src/ranking/pairwise_reranker.py").read_text(encoding="utf-8")
        # Confirm it reads labels from the DataFrame column (correct), not from a hardcoded list
        assert "group[\"label\"] == 1" in reranker_src or "group['label'] == 1" in reranker_src, (
            "Training pairs must be selected by the label column, not by hardcoded query IDs"
        )


# ─── Test 4: Feature engineering determinism ─────────────────────────────────

class TestFeatureEngineering:
    """engineer_features should produce identical results when called twice on the same DataFrame."""

    def test_deterministic_query_norm(self):
        df = pd.DataFrame({
            "query_id":    ["q1", "q1", "q1"],
            "candidate_id": ["c1", "c2", "c3"],
            "label":       [1, 0, 0],
            "tfidf_score": [0.8, 0.4, 0.2],
            "inlegalbert_sim": [0.7, 0.5, 0.3],
            "composite_legal_score": [0.6, 0.3, 0.1],
            "bm25_score":  [5.0, 3.0, 1.0],
            "statute_jaccard": [0.1, 0.0, 0.0],
            "statute_containment_query": [0.1, 0.0, 0.0],
            "statute_shared_count": [1.0, 0.0, 0.0],
            "citation_jaccard": [0.0, 0.0, 0.0],
            "citation_shared_count": [0.0, 0.0, 0.0],
            "entity_shared_count": [0.0, 0.0, 0.0],
            "entity_jaccard": [0.0, 0.0, 0.0],
            "judge_overlap": [0.0, 0.0, 0.0],
            "court_overlap": [0.0, 0.0, 0.0],
            "org_overlap":   [0.0, 0.0, 0.0],
            "rhetorical_cosine": [0.5, 0.3, 0.1],
            "rhetorical_js_sim": [0.5, 0.3, 0.1],
            "rhetorical_ratio_diff": [0.1, 0.1, 0.1],
        })
        df1 = PairwiseNeuralReRanker.engineer_features(df.copy())
        df2 = PairwiseNeuralReRanker.engineer_features(df.copy())
        pd.testing.assert_frame_equal(df1, df2)

    def test_stage1_score_formula(self):
        """stage1_score = 0.65*qnorm_tfidf + 0.25*qnorm_dense + 0.10*qnorm_legal."""
        df = pd.DataFrame({
            "query_id":    ["q1", "q1"],
            "candidate_id": ["c1", "c2"],
            "label":       [1, 0],
            "tfidf_score": [1.0, 0.0],
            "inlegalbert_sim": [1.0, 0.0],
            "composite_legal_score": [1.0, 0.0],
            "bm25_score":  [1.0, 0.0],
            "statute_jaccard": [0.0, 0.0],
            "statute_containment_query": [0.0, 0.0],
            "statute_shared_count": [0.0, 0.0],
            "citation_jaccard": [0.0, 0.0],
            "citation_shared_count": [0.0, 0.0],
            "entity_shared_count": [0.0, 0.0],
            "entity_jaccard": [0.0, 0.0],
            "judge_overlap": [0.0, 0.0],
            "court_overlap": [0.0, 0.0],
            "org_overlap":   [0.0, 0.0],
            "rhetorical_cosine": [0.0, 0.0],
            "rhetorical_js_sim": [0.0, 0.0],
            "rhetorical_ratio_diff": [0.0, 0.0],
        })
        out = PairwiseNeuralReRanker.engineer_features(df)
        # c1 has max values → qnorm all 1.0 → stage1 = 0.65+0.25+0.10 = 1.0
        # c2 has min values → qnorm all 0.0 → stage1 = 0.0
        assert out.loc[out["candidate_id"] == "c1", "stage1_score"].values[0] == pytest.approx(1.0, abs=1e-6)
        assert out.loc[out["candidate_id"] == "c2", "stage1_score"].values[0] == pytest.approx(0.0, abs=1e-6)

    def test_feature_normalizer_fitted_on_training_data(self):
        """Normalizer statistics should differ when fit on different data (not cross-contaminated)."""
        train_features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        test_features  = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

        reranker = PairwiseNeuralReRanker.__new__(PairwiseNeuralReRanker)
        reranker.feature_means = None
        reranker.feature_stds  = None

        reranker.fit_normalizer(train_features)
        train_mean = reranker.feature_means.copy()

        reranker.fit_normalizer(test_features)
        test_mean  = reranker.feature_means.copy()

        # Statistics must differ — if normalizer were accidentally fitted on test
        # and then used on train, they would match test_mean, not train_mean.
        assert not np.allclose(train_mean, test_mean), (
            "Normalizer statistics must differ when fitted on different distributions"
        )


# ─── Test 5: Evaluated queries count ─────────────────────────────────────────

class TestEvaluatedQueriesCount:
    """Exactly 234 test queries should be evaluated (237 total minus 3 zero-relevance)."""

    def test_zero_relevance_queries_excluded(self):
        """HybridEvaluator skips queries with 0 positives in the candidate pool."""
        df = make_df({
            "q_good": [(0.9, 1), (0.5, 0)],
            "q_zero": [(0.8, 0), (0.3, 0)],  # no positives
        })
        m = HybridEvaluator.evaluate_predictions(df, score_col="score")
        assert m["evaluated_queries"] == 1

    def test_234_eligible_test_queries(self):
        """Integration check: test split has exactly 234 eligible queries."""
        test_path = PROJECT_ROOT / "data/processed/IL-TUR-PCR/test/hybrid_features.parquet"
        if not test_path.exists():
            pytest.skip("Test parquet not available in this environment")

        df = pd.read_parquet(test_path)
        df = PairwiseNeuralReRanker.engineer_features(df)
        m  = HybridEvaluator.evaluate_predictions(df, score_col="stage1_score")
        assert m["evaluated_queries"] == 234, (
            f"Expected 234 eligible test queries, got {m['evaluated_queries']}"
        )
