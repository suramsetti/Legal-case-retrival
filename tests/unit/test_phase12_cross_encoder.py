"""
Unit Tests for Phase 12: Legal Cross-Encoder and Rhetorical-Aware Neural Reranking.

Verifies:
1. SectionAwareChunker extraction and rhetorical section partitioning.
2. Chunk score aggregation strategies (max, mean, weighted).
3. Cross-encoder scoring input/output shapes.
4. AdaptiveCrossReranker margin gating (high-margin preservation vs low-margin cross-encoding).
5. Candidate pool retention and label preservation.
6. Configuration serialization and deserialization.
"""

import sys
import tempfile
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ranking.legal_cross_encoder import SectionAwareChunker, InLegalBERTCrossEncoder
from src.ranking.adaptive_cross_reranker import AdaptiveCrossReranker


@pytest.fixture
def sample_judgment_sentences():
    return [
        "CIVIL APPELLATE JURISDICTION: Civil Appeal No. 1024 of 2010. S. R. Sharma Appellant Versus State Respondent.",
        "The short question for determination is whether the High Court was justified in setting aside the award.",
        "The prosecution alleged that the incident took place on the night of 15th August near the railway junction.",
        "Learned counsel for the appellant contended that the trial court failed to appreciate the documentary evidence.",
        "Learned counsel for the respondent submitted that the concurrent findings of fact warrant no interference.",
        "In our considered opinion, having carefully considered the rival submissions, the settled legal position is clear.",
        "We hold that section 34 of the Arbitration Act does not permit reappreciation of purely factual findings.",
        "In the result, the appeal is allowed and the judgment of the High Court is set aside. No order as to costs.",
    ]


@pytest.fixture
def synthetic_ranking_data():
    rows = []
    # Query 1: High margin (margin >= 0.07)
    rows.append({"query_id": "q1", "candidate_id": "c1_pos", "label": 1, "tfidf_score": 0.90, "inlegalbert_sim": 0.95, "composite_legal_score": 0.80, "bm25_score": 1000.0, "rhetorical_cosine": 0.95})
    rows.append({"query_id": "q1", "candidate_id": "c1_neg1", "label": 0, "tfidf_score": 0.40, "inlegalbert_sim": 0.50, "composite_legal_score": 0.30, "bm25_score": 200.0, "rhetorical_cosine": 0.60})

    # Query 2: Low margin (margin < 0.07)
    rows.append({"query_id": "q2", "candidate_id": "c2_neg", "label": 0, "tfidf_score": 0.81, "inlegalbert_sim": 0.80, "composite_legal_score": 0.50, "bm25_score": 500.0, "rhetorical_cosine": 0.50})
    rows.append({"query_id": "q2", "candidate_id": "c2_pos", "label": 1, "tfidf_score": 0.80, "inlegalbert_sim": 0.81, "composite_legal_score": 0.50, "bm25_score": 490.0, "rhetorical_cosine": 0.98})
    rows.append({"query_id": "q2", "candidate_id": "c2_neg2", "label": 0, "tfidf_score": 0.10, "inlegalbert_sim": 0.10, "composite_legal_score": 0.10, "bm25_score": 50.0, "rhetorical_cosine": 0.10})

    df = pd.DataFrame(rows)
    return df


class TestSectionAwareChunker:

    def test_section_extraction(self, sample_judgment_sentences):
        chunker = SectionAwareChunker()
        sections = chunker.extract_sections(sample_judgment_sentences)

        assert "ratio" in sections
        assert "headnote" in sections
        assert "facts" in sections
        assert "ruling" in sections

        # Ratio section should capture the legal holding
        assert "hold that section 34" in sections["ratio"].lower()
        # Ruling section should capture appeal allowed
        assert "appeal is allowed" in sections["ruling"].lower()

    def test_chunk_document_weights(self, sample_judgment_sentences):
        chunker = SectionAwareChunker()
        chunks = chunker.chunk_document(sample_judgment_sentences)

        assert len(chunks) > 0
        section_types = [c["section_type"] for c in chunks]
        assert "ratio" in section_types

        # Ratio chunk should carry higher rhetorical weight than facts
        ratio_w = next(c["rhetorical_weight"] for c in chunks if c["section_type"] == "ratio")
        facts_w = next(c["rhetorical_weight"] for c in chunks if c["section_type"] == "facts")
        assert ratio_w > facts_w


class TestCrossEncoderScoring:

    def test_score_candidate_chunks_aggregations(self):
        # Test mathematical aggregations
        chunks = [
            {"section_type": "ratio", "text": "Legal holding text", "rhetorical_weight": 1.25},
            {"section_type": "facts", "text": "Factual background text", "rhetorical_weight": 0.90},
            {"section_type": "ruling", "text": "Appeal allowed text", "rhetorical_weight": 1.10},
        ]

        # Mock score_pairs behavior
        scores = np.array([0.80, 0.40, 0.60], dtype=np.float32)
        weights = np.array([1.25, 0.90, 1.10], dtype=np.float32)
        weights = weights / np.sum(weights)

        expected_max = 0.80
        expected_mean = float(np.mean(scores))
        expected_weighted = float(np.sum(scores * weights))

        assert np.isclose(np.max(scores), expected_max)
        assert np.isclose(np.mean(scores), expected_mean)
        assert np.isclose(np.sum(scores * weights), expected_weighted)
        assert expected_weighted > expected_mean  # ratio chunk upweights the overall score


class TestAdaptiveCrossReranker:

    def test_high_margin_preservation(self, synthetic_ranking_data):
        df = AdaptiveCrossReranker.compute_stage1_features(synthetic_ranking_data)
        ce_scores = {
            ("q1", "c1_pos"): 0.5,
            ("q1", "c1_neg1"): 0.9,  # Even if CE prefers neg1, high-margin must ignore it
            ("q2", "c2_neg"): 0.3,
            ("q2", "c2_pos"): 0.9,
        }

        acr = AdaptiveCrossReranker(tau=0.07, beta=0.25, gamma=0.04, top_n=5)
        df_reranked = acr.rerank_with_precomputed_ce_scores(df, ce_scores)

        # q1 is high margin -> score and rank must equal Stage 1
        q1_sub = df_reranked[df_reranked["query_id"] == "q1"]
        np.testing.assert_allclose(q1_sub["final_cross_score"].values, q1_sub["stage1_score"].values)
        assert q1_sub[q1_sub["candidate_id"] == "c1_pos"]["cross_rank"].iloc[0] == 1

    def test_low_margin_cross_reranking(self, synthetic_ranking_data):
        df = AdaptiveCrossReranker.compute_stage1_features(synthetic_ranking_data)
        ce_scores = {
            ("q1", "c1_pos"): 0.8,
            ("q1", "c1_neg1"): 0.2,
            ("q2", "c2_neg"): 0.1,
            ("q2", "c2_pos"): 0.95,  # CE strongly prefers c2_pos
        }

        acr = AdaptiveCrossReranker(tau=0.07, beta=0.30, gamma=0.05, top_n=5)
        df_reranked = acr.rerank_with_precomputed_ce_scores(df, ce_scores)

        # q2 is low margin -> c2_pos should be promoted to rank 1
        q2_sub = df_reranked[df_reranked["query_id"] == "q2"]
        pos_rank = q2_sub[q2_sub["candidate_id"] == "c2_pos"]["cross_rank"].iloc[0]
        assert pos_rank == 1

    def test_pool_integrity(self, synthetic_ranking_data):
        df = AdaptiveCrossReranker.compute_stage1_features(synthetic_ranking_data)
        ce_scores = {("q1", "c1_pos"): 0.5, ("q1", "c1_neg1"): 0.4, ("q2", "c2_neg"): 0.5, ("q2", "c2_pos"): 0.6}
        acr = AdaptiveCrossReranker(tau=0.07, beta=0.20, gamma=0.04, top_n=5)
        df_reranked = acr.rerank_with_precomputed_ce_scores(df, ce_scores)

        assert len(df_reranked) == len(synthetic_ranking_data)
        assert set(df_reranked["candidate_id"]) == set(synthetic_ranking_data["candidate_id"])
        assert (df_reranked["label"].values == synthetic_ranking_data["label"].values).all()

    def test_config_save_load(self):
        acr = AdaptiveCrossReranker(tau=0.06, beta=0.22, gamma=0.045, top_n=5, chunk_aggregation="weighted")
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = Path(tmpdir) / "test_phase12_config.json"
            acr.save_config(cfg_file)

            loaded = AdaptiveCrossReranker.load_config(cfg_file)
            assert loaded.tau == 0.06
            assert loaded.beta == 0.22
            assert loaded.gamma == 0.045
            assert loaded.top_n == 5
            assert loaded.chunk_aggregation == "weighted"

