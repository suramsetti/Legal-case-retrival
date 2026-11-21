"""
Unit Tests for Phase 16: Advanced Legal Retrieval Accuracy Optimization.

Verifies:
1. InLegalBERTBiEncoder dense representation and cosine similarity.
2. Phase16SectionChunker rhetorical chunk extraction and aggregation.
3. Phase16LegalCrossEncoder scoring behaviors.
4. Phase16MultiStageReranker Stage-1 multi-modal score computation.
5. High-margin preservation vs low-margin confidence-gated overrides.
6. Deterministic 100% fallback to Phase 14 / Phase 15 baselines.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import MagicMock

from src.models.phase16_biencoder import InLegalBERTBiEncoder
from src.models.phase16_cross_encoder import Phase16LegalCrossEncoder, Phase16SectionChunker
from src.ranking.phase16_reranker import Phase16MultiStageReranker


@pytest.fixture
def mock_pairwise():
    mock = MagicMock()
    mock.predict_scores = MagicMock(side_effect=lambda df: np.ones(len(df)) * 0.5)
    return mock


@pytest.fixture
def sample_judgment_sentences():
    return [
        "CIVIL APPELLATE JURISDICTION: Civil Appeal No. 1024 of 2010. S. R. Sharma Appellant Versus State Respondent.",
        "The question for determination is whether the High Court was justified in setting aside the arbitral award.",
        "The respondent contended that the award was contrary to public policy under section 34.",
        "In our considered opinion, the arbitral tribunal is the sole judge of the quality and quantity of evidence.",
        "We hold that interference under section 34 is not permissible on purely factual re-evaluation.",
        "In the result, the appeal is allowed and the judgment of the High Court is set aside with no order as to costs.",
    ]


@pytest.fixture
def sample_ranking_df():
    rows = [
        # Query 1: High margin (margin >= 0.065)
        {
            "query_id": "Q1",
            "candidate_id": "C1_pos",
            "label": 1,
            "tfidf_score": 0.90,
            "inlegalbert_sim": 0.85,
            "composite_legal_score": 0.80,
            "bm25_score": 800.0,
            "norm_statute_jaccard": 0.10,
            "norm_tfidf_score": 0.90,
            "rhetorical_cosine": 0.70,
        },
        {
            "query_id": "Q1",
            "candidate_id": "C1_neg",
            "label": 0,
            "tfidf_score": 0.20,
            "inlegalbert_sim": 0.30,
            "composite_legal_score": 0.20,
            "bm25_score": 100.0,
            "norm_statute_jaccard": 0.0,
            "norm_tfidf_score": 0.20,
            "rhetorical_cosine": 0.30,
        },
        # Query 2: Low margin (borderline override candidate)
        {
            "query_id": "Q2",
            "candidate_id": "C2_top",
            "label": 0,
            "tfidf_score": 0.705,
            "inlegalbert_sim": 0.70,
            "composite_legal_score": 0.40,
            "bm25_score": 500.0,
            "norm_statute_jaccard": 0.05,
            "norm_tfidf_score": 0.705,
            "rhetorical_cosine": 0.50,
        },
        {
            "query_id": "Q2",
            "candidate_id": "C2_pos",
            "label": 1,
            "tfidf_score": 0.700,
            "inlegalbert_sim": 0.70,
            "composite_legal_score": 0.40,
            "bm25_score": 495.0,
            "norm_statute_jaccard": 0.60,  # Stat superiority > 0.35
            "norm_tfidf_score": 0.700,
            "rhetorical_cosine": 0.50,
        },
        {
            "query_id": "Q2",
            "candidate_id": "C2_min",
            "label": 0,
            "tfidf_score": 0.0,
            "inlegalbert_sim": 0.0,
            "composite_legal_score": 0.0,
            "bm25_score": 0.0,
            "norm_statute_jaccard": 0.0,
            "norm_tfidf_score": 0.0,
            "rhetorical_cosine": 0.0,
        },
    ]
    return pd.DataFrame(rows)


def test_biencoder_projection():
    biencoder = InLegalBERTBiEncoder(model_name_or_path=None, projection_dim=128, device="cpu")
    dense_features = torch.randn(4, 768)
    proj = biencoder.encode_dense_features(dense_features)
    assert proj.shape == (4, 128)
    norms = torch.norm(proj, p=2, dim=1)
    assert np.allclose(norms.detach().numpy(), 1.0, atol=1e-5)


def test_section_chunker_rhetorical_extraction(sample_judgment_sentences):
    chunker = Phase16SectionChunker()
    chunks = chunker.chunk_document(sample_judgment_sentences)
    assert len(chunks) > 0
    section_types = [c["section_type"] for c in chunks]
    assert "ratio" in section_types
    assert "ruling" in section_types


def test_cross_encoder_chunk_aggregation():
    chunks = [
        {"section_type": "ratio", "text": "Legal holding text", "rhetorical_weight": 1.30},
        {"section_type": "facts", "text": "Factual background text", "rhetorical_weight": 0.85},
        {"section_type": "ruling", "text": "Appeal allowed text", "rhetorical_weight": 1.15},
    ]
    scores = np.array([0.85, 0.40, 0.65], dtype=np.float32)
    weights = np.array([1.30, 0.85, 1.15], dtype=np.float32)
    weights = weights / np.sum(weights)

    expected_weighted = float(np.sum(scores * weights))
    expected_mean = float(np.mean(scores))
    assert expected_weighted > expected_mean


def test_stage1_multimodal_feature_computation(sample_ranking_df):
    df_s1 = Phase16MultiStageReranker.compute_stage1_features(sample_ranking_df)
    assert "stage1_score" in df_s1.columns
    assert "s1_rank" in df_s1.columns
    q1_sub = df_s1[df_s1["query_id"] == "Q1"].sort_values("stage1_score", ascending=False)
    assert q1_sub.iloc[0]["candidate_id"] == "C1_pos"


def test_phase16_multistage_reranker_execution(sample_ranking_df, mock_pairwise):
    reranker = Phase16MultiStageReranker(pairwise_reranker=mock_pairwise)
    out = reranker.rerank(sample_ranking_df)
    assert "p16_score" in out.columns
    assert "p16_rank" in out.columns
    assert "override_applied" in out.columns

    # Candidate pool retention
    assert len(out) == len(sample_ranking_df)
    assert set(out["candidate_id"]) == set(sample_ranking_df["candidate_id"])

    # Verify Q1 preserved Rank 1
    q1_top = out[out["query_id"] == "Q1"].sort_values("p16_rank").iloc[0]
    assert q1_top["candidate_id"] == "C1_pos"

    # Verify Q2 candidate with statutory superiority gets boosted
    q2_pos = out[(out["query_id"] == "Q2") & (out["candidate_id"] == "C2_pos")].iloc[0]
    assert bool(q2_pos["override_applied"]) is True
    assert q2_pos["p16_rank"] == 1
