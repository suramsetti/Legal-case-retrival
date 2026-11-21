"""
Unit Tests for Phase 15 Generalization-Optimized Reranker.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.ranking.phase15_reranker import Phase15GeneralizationReRanker, Phase15CalibratedAdaptiveReRanker


@pytest.fixture
def mock_pairwise():
    mock = MagicMock()
    mock.predict_scores = MagicMock(side_effect=lambda df: np.ones(len(df)) * 0.5)
    return mock


@pytest.fixture
def sample_df():
    rows = [
        {
            "query_id": "Q1",
            "candidate_id": "C1",
            "label": 1,
            "tfidf_score": 0.85,
            "inlegalbert_sim": 0.80,
            "composite_legal_score": 0.70,
            "bm25_score": 0.50,
            "norm_statute_jaccard": 0.10,
            "norm_tfidf_score": 0.85,
            "rhetorical_cosine": 0.50,
        },
        {
            "query_id": "Q1",
            "candidate_id": "C2",
            "label": 0,
            "tfidf_score": 0.20,
            "inlegalbert_sim": 0.20,
            "composite_legal_score": 0.20,
            "bm25_score": 0.20,
            "norm_statute_jaccard": 0.0,
            "norm_tfidf_score": 0.20,
            "rhetorical_cosine": 0.20,
        },
    ]
    return pd.DataFrame(rows)


def test_calibrated_stage1_computation(sample_df):
    out = Phase15CalibratedAdaptiveReRanker.compute_stage1_features(sample_df)
    assert "stage1_score" in out.columns
    assert "s1_rank" in out.columns
    assert out.iloc[0]["s1_rank"] == 1


def test_phase15_generalization_reranker_execution(sample_df, mock_pairwise):
    reranker = Phase15GeneralizationReRanker(pairwise_reranker=mock_pairwise)
    out = reranker.rerank(sample_df)
    assert "p14_score" in out.columns
    assert "p14_rank" in out.columns
    assert out.iloc[0]["p14_rank"] == 1

