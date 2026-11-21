"""
Unit Tests for Phase 14 Selective Reranker and Confidence-Gated Overrides.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.ranking.phase14_reranker import Phase14SelectiveReRanker


@pytest.fixture
def mock_pairwise_reranker():
    mock = MagicMock()
    # Return equal neutral scores for slice candidates so s2_norm = 0.5
    mock.predict_scores = MagicMock(side_effect=lambda df: np.ones(len(df)) * 0.5)
    return mock


@pytest.fixture
def sample_candidates_df():
    """Generates synthetic candidate data with 2 queries."""
    rows = []
    # Query 1: High margin (margin >= 0.07)
    rows.append({
        "query_id": "Q1",
        "candidate_id": "C1_1",
        "label": 1,
        "tfidf_score": 0.9,
        "inlegalbert_sim": 0.8,
        "composite_legal_score": 0.7,
        "bm25_score": 0.5,
        "norm_statute_jaccard": 0.1,
        "norm_tfidf_score": 0.9,
        "rhetorical_cosine": 0.8,
    })
    rows.append({
        "query_id": "Q1",
        "candidate_id": "C1_2",
        "label": 0,
        "tfidf_score": 0.2,
        "inlegalbert_sim": 0.2,
        "composite_legal_score": 0.2,
        "bm25_score": 0.2,
        "norm_statute_jaccard": 0.0,
        "norm_tfidf_score": 0.2,
        "rhetorical_cosine": 0.2,
    })

    # Query 2: Low margin with ultra-close scores where C2_2 has statutory superiority
    rows.append({
        "query_id": "Q2",
        "candidate_id": "C2_1",
        "label": 0,
        "stage1_score": 0.800,
        "norm_statute_jaccard": 0.10,
        "norm_tfidf_score": 0.80,
        "rhetorical_cosine": 0.50,
    })
    rows.append({
        "query_id": "Q2",
        "candidate_id": "C2_2",
        "label": 1,
        "stage1_score": 0.795,  # Margin = 0.005 < 0.07!
        "norm_statute_jaccard": 0.90,  # +0.80 superiority
        "norm_tfidf_score": 0.79,
        "rhetorical_cosine": 0.50,
    })

    return pd.DataFrame(rows)


def test_high_margin_preservation(sample_candidates_df, mock_pairwise_reranker):
    reranker = Phase14SelectiveReRanker(pairwise_reranker=mock_pairwise_reranker)
    out = reranker.rerank(sample_candidates_df)

    q1_sub = out[out["query_id"] == "Q1"].sort_values("p14_score", ascending=False)
    assert q1_sub.iloc[0]["candidate_id"] == "C1_1"
    assert q1_sub.iloc[0]["override_applied"] == False


def test_confidence_override_execution(sample_candidates_df, mock_pairwise_reranker):
    reranker = Phase14SelectiveReRanker(
        override_delta=0.05,
        statute_threshold=0.35,
        pairwise_reranker=mock_pairwise_reranker,
    )
    out = reranker.rerank(sample_candidates_df)

    q2_sub = out[out["query_id"] == "Q2"].sort_values("p14_score", ascending=False)
    # C2_2 should be promoted ahead of C2_1
    assert q2_sub.iloc[0]["candidate_id"] == "C2_2"
    assert q2_sub.iloc[0]["override_applied"] == True
    assert "statutory_superiority_boost" in q2_sub.iloc[0]["override_reason"]


def test_lexical_guardrail_prevents_bad_override(sample_candidates_df, mock_pairwise_reranker):
    # If lexical guardrail is very tight, override is prevented
    reranker = Phase14SelectiveReRanker(
        override_delta=0.05,
        statute_threshold=0.35,
        lexical_guardrail=0.05,  # Requires C2_2 to have strictly higher lexical score
        pairwise_reranker=mock_pairwise_reranker,
    )
    out = reranker.rerank(sample_candidates_df)

    q2_sub = out[out["query_id"] == "Q2"].sort_values("p14_score", ascending=False)
    # C2_1 should remain top because C2_2's tfidf_diff < 0.05
    assert q2_sub.iloc[0]["candidate_id"] == "C2_1"
    assert q2_sub.iloc[0]["override_applied"] == False

