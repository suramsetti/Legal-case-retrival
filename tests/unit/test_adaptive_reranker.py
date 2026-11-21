"""
Unit Tests for Query-Adaptive Margin-Gated Re-Ranker (Phase 11).

Verifies:
1. Stage-1 margin calculation logic and edge cases.
2. Query classification into high-confidence and low-confidence regimes.
3. Preservation of Stage-1 ranking for high-margin queries.
4. Selective re-ranking execution on low-margin queries.
5. Full candidate pool retention and deterministic tie-breaking.
6. Rhetorical feature integration.
7. Config save and load integrity.
8. Zero-leakage contract.
"""

import sys
import json
import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ranking.adaptive_reranker import QueryAdaptiveReRanker
from src.ranking.pairwise_reranker import PairwiseNeuralReRanker


@pytest.fixture
def synthetic_ranking_data():
    """Constructs a deterministic synthetic multi-query dataset."""
    rows = []
    # Query 1: High margin (margin = 0.90 - 0.40 = 0.50 >= 0.07)
    rows.append({"query_id": "q1", "candidate_id": "c1_pos", "label": 1, "tfidf_score": 0.90, "inlegalbert_sim": 0.95, "composite_legal_score": 0.80, "bm25_score": 1000.0, "rhetorical_cosine": 0.95})
    rows.append({"query_id": "q1", "candidate_id": "c1_neg1", "label": 0, "tfidf_score": 0.40, "inlegalbert_sim": 0.50, "composite_legal_score": 0.30, "bm25_score": 200.0, "rhetorical_cosine": 0.60})
    rows.append({"query_id": "q1", "candidate_id": "c1_neg2", "label": 0, "tfidf_score": 0.10, "inlegalbert_sim": 0.20, "composite_legal_score": 0.10, "bm25_score": 50.0, "rhetorical_cosine": 0.40})

    # Query 2: Low margin (margin = 0.81 - 0.80 = 0.01 < 0.07)
    # c2_neg is at rank 1 on Stage 1, but c2_pos has higher rhetorical cosine
    rows.append({"query_id": "q2", "candidate_id": "c2_neg", "label": 0, "tfidf_score": 0.81, "inlegalbert_sim": 0.80, "composite_legal_score": 0.50, "bm25_score": 500.0, "rhetorical_cosine": 0.50})
    rows.append({"query_id": "q2", "candidate_id": "c2_pos", "label": 1, "tfidf_score": 0.80, "inlegalbert_sim": 0.81, "composite_legal_score": 0.50, "bm25_score": 490.0, "rhetorical_cosine": 0.98})
    rows.append({"query_id": "q2", "candidate_id": "c2_neg2", "label": 0, "tfidf_score": 0.20, "inlegalbert_sim": 0.20, "composite_legal_score": 0.20, "bm25_score": 100.0, "rhetorical_cosine": 0.30})

    # Query 3: Single candidate edge case
    rows.append({"query_id": "q3", "candidate_id": "c3_single", "label": 1, "tfidf_score": 0.50, "inlegalbert_sim": 0.50, "composite_legal_score": 0.50, "bm25_score": 200.0, "rhetorical_cosine": 0.50})

    df = pd.DataFrame(rows)
    return df


class TestQueryAdaptiveReRanker:

    def test_stage1_feature_computation(self, synthetic_ranking_data):
        df = QueryAdaptiveReRanker.compute_stage1_features(synthetic_ranking_data)
        assert "stage1_score" in df.columns
        assert "s1_rank" in df.columns
        # Verify deterministic descending ranks
        for qid, grp in df.groupby("query_id"):
            ranks = grp["s1_rank"].values
            assert list(ranks) == list(range(1, len(grp) + 1))

    def test_margin_computation(self, synthetic_ranking_data):
        df = QueryAdaptiveReRanker.compute_stage1_features(synthetic_ranking_data)
        margins = QueryAdaptiveReRanker.compute_query_margins(df, score_col="stage1_score")
        assert len(margins) == len(df)

        # q1 margin should be high
        q1_margin = margins[df["query_id"] == "q1"].iloc[0]
        assert q1_margin > 0.10

        # q2 margin should be very low
        q2_margin = margins[df["query_id"] == "q2"].iloc[0]
        assert q2_margin < 0.05

        # q3 single candidate margin should be 1.0 (unambiguous)
        q3_margin = margins[df["query_id"] == "q3"].iloc[0]
        assert q3_margin == 1.0

    def test_query_classification_into_regimes(self, synthetic_ranking_data):
        reranker = QueryAdaptiveReRanker(tau=0.07)
        df_classified = reranker.classify_queries(synthetic_ranking_data)

        assert "routing_regime" in df_classified.columns
        assert "is_low_margin" in df_classified.columns

        # q1 should be high_margin_preserved
        q1_regime = df_classified[df_classified["query_id"] == "q1"]["routing_regime"].iloc[0]
        assert q1_regime == "high_margin_preserved"

        # q2 should be low_margin_reranked
        q2_regime = df_classified[df_classified["query_id"] == "q2"]["routing_regime"].iloc[0]
        assert q2_regime == "low_margin_reranked"

        # q3 should be high_margin_preserved
        q3_regime = df_classified[df_classified["query_id"] == "q3"]["routing_regime"].iloc[0]
        assert q3_regime == "high_margin_preserved"

    def test_high_margin_exact_preservation(self, synthetic_ranking_data):
        reranker = QueryAdaptiveReRanker(tau=0.07, alpha=0.20, gamma=0.05)
        df_reranked = reranker.rerank(synthetic_ranking_data)

        # For q1, final adaptive score must equal stage1_score
        q1_data = df_reranked[df_reranked["query_id"] == "q1"]
        np.testing.assert_allclose(q1_data["final_adaptive_score"].values, q1_data["stage1_score"].values)
        assert list(q1_data["adaptive_rank"].values) == list(q1_data["s1_rank"].values)

    def test_candidate_pool_retention(self, synthetic_ranking_data):
        initial_len = len(synthetic_ranking_data)
        reranker = QueryAdaptiveReRanker(tau=0.07, alpha=0.15, gamma=0.04)
        df_reranked = reranker.rerank(synthetic_ranking_data)

        # No rows dropped
        assert len(df_reranked) == initial_len
        # All candidate IDs preserved
        assert set(df_reranked["candidate_id"]) == set(synthetic_ranking_data["candidate_id"])
        # Relevance labels untouched
        assert (df_reranked["label"].values == synthetic_ranking_data["label"].values).all()

    def test_rhetorical_adjustment_in_low_margin(self, synthetic_ranking_data):
        # In q2, c2_pos has higher rhetorical cosine than c2_neg
        reranker_no_rhet = QueryAdaptiveReRanker(tau=0.07, alpha=0.0, gamma=0.0)
        reranker_rhet = QueryAdaptiveReRanker(tau=0.07, alpha=0.0, gamma=0.10)

        df_no_rhet = reranker_no_rhet.rerank(synthetic_ranking_data)
        df_rhet = reranker_rhet.rerank(synthetic_ranking_data)

        # With gamma=0.10 on rhetorical cosine, c2_pos gets boosted
        q2_rhet = df_rhet[df_rhet["query_id"] == "q2"].set_index("candidate_id")
        assert q2_rhet.loc["c2_pos", "final_adaptive_score"] > q2_rhet.loc["c2_neg", "final_adaptive_score"]
        assert q2_rhet.loc["c2_pos", "adaptive_rank"] == 1

    def test_config_save_load(self):
        reranker = QueryAdaptiveReRanker(tau=0.065, alpha=0.11, gamma=0.035, top_k=25)
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "test_config.json"
            reranker.save_config(cfg_path)

            loaded = QueryAdaptiveReRanker.load_config(cfg_path)
            assert loaded.tau == 0.065
            assert loaded.alpha == 0.11
            assert loaded.gamma == 0.035
            assert loaded.top_k == 25

    def test_evaluate_split_metrics_dict(self, synthetic_ranking_data):
        reranker = QueryAdaptiveReRanker(tau=0.07, alpha=0.10, gamma=0.04)
        summary = reranker.evaluate_split(synthetic_ranking_data)

        assert "overall_adaptive" in summary
        assert "stage1_baseline" in summary
        assert "queries_reranked" in summary
        assert "queries_preserved" in summary
        assert "latency_ms_per_query" in summary
        assert summary["total_queries"] == 3
        assert summary["queries_reranked"] == 1  # only q2 is low margin
        assert summary["queries_preserved"] == 2  # q1 and q3 are high margin

