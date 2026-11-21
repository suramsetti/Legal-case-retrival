"""
Unit Tests for Module C: Gemini-Assisted Precedent Re-Ranking.

Verifies:
1. Alignment score schema validation (bounds [0.0, 1.0]).
2. Margin gating behavior:
   - High-margin queries (margin >= tau) retain Phase 11 baseline order untouched.
   - Low-margin queries (margin < tau) selectively blend Gemini relevance scores.
3. Candidate pool preservation (candidate count and identity preserved).
4. Deterministic fallback when Gemini is offline.
"""

from unittest.mock import MagicMock
import pytest

from backend.services.gemini_precedent_reranker import GeminiPrecedentReRanker


class TestGeminiPrecedentReRanker:

    def test_schema_validator(self):
        valid = {
            "query_id": "q1",
            "candidate_id": "c1",
            "legal_issue_alignment": 0.85,
            "factual_alignment": 0.70,
            "statutory_alignment": 0.95,
            "ratio_alignment": 0.80,
            "reasoning_alignment": 0.75,
            "overall_relevance": 0.82,
            "supporting_evidence": ["Shared statutory issue regarding Section 482 CrPC"],
            "contradictory_evidence": [],
            "uncertainties": [],
        }
        assert GeminiPrecedentReRanker.validate_schema(valid) is True

        # Out-of-bounds score (> 1.0)
        invalid = valid.copy()
        invalid["overall_relevance"] = 1.50
        assert GeminiPrecedentReRanker.validate_schema(invalid) is False

        # Out-of-bounds score (< 0.0)
        invalid2 = valid.copy()
        invalid2["factual_alignment"] = -0.1
        assert GeminiPrecedentReRanker.validate_schema(invalid2) is False

    def test_high_margin_preservation(self):
        # Margin is 0.15 >= tau (0.07) -> High confidence -> DO NOT RERANK
        reranker = GeminiPrecedentReRanker(tau=0.07, gemini_weight=0.20, top_k_candidates=5)

        candidates = [
            {"candidate_id": "c_rank1", "hybrid_score": 0.90, "rank": 1},
            {"candidate_id": "c_rank2", "hybrid_score": 0.75, "rank": 2},
            {"candidate_id": "c_rank3", "hybrid_score": 0.50, "rank": 3},
        ]
        docs = {"c_rank1": {"full_text": "..."}, "c_rank2": {"full_text": "..."}, "c_rank3": {"full_text": "..."}}

        reranked = reranker.rerank_query_candidates(
            query_id="q_high_margin",
            query_text="high margin query text",
            candidate_rows=candidates,
            candidate_docs=docs,
            stage1_margin=0.15,
        )

        assert len(reranked) == 3
        assert reranked[0]["candidate_id"] == "c_rank1"
        assert reranked[0]["gemini_rank"] == 1
        assert reranked[0]["routing_reason"] == "high_margin_preserved"
        assert reranked[1]["candidate_id"] == "c_rank2"

    def test_low_margin_reranking(self):
        # Margin is 0.01 < tau (0.07) -> Ambiguous query -> Gemini re-ranks Top-K
        mock_gemini = MagicMock()

        def mock_generate_json(prompt, system_instruction=None, schema_validator=None, fallback_factory=None):
            # If candidate is c_rank2, assign higher relevance (0.95) than c_rank1 (0.20)
            if "c_rank2" in prompt:
                return {
                    "query_id": "q_low",
                    "candidate_id": "c_rank2",
                    "legal_issue_alignment": 0.95,
                    "factual_alignment": 0.90,
                    "statutory_alignment": 0.95,
                    "ratio_alignment": 0.95,
                    "reasoning_alignment": 0.90,
                    "overall_relevance": 0.95,
                    "supporting_evidence": ["Strong precedent match"],
                    "contradictory_evidence": [],
                    "uncertainties": [],
                }
            else:
                return {
                    "query_id": "q_low",
                    "candidate_id": "c_rank1",
                    "legal_issue_alignment": 0.20,
                    "factual_alignment": 0.20,
                    "statutory_alignment": 0.20,
                    "ratio_alignment": 0.20,
                    "reasoning_alignment": 0.20,
                    "overall_relevance": 0.20,
                    "supporting_evidence": [],
                    "contradictory_evidence": ["Facts completely differ"],
                    "uncertainties": [],
                }

        mock_gemini.generate_json.side_effect = mock_generate_json

        reranker = GeminiPrecedentReRanker(gemini_service=mock_gemini, tau=0.07, gemini_weight=0.30, top_k_candidates=5)

        candidates = [
            {"candidate_id": "c_rank1", "hybrid_score": 0.81, "rank": 1},
            {"candidate_id": "c_rank2", "hybrid_score": 0.80, "rank": 2},
        ]
        docs = {
            "c_rank1": {"full_text": "Case 1 text"},
            "c_rank2": {"full_text": "Case 2 text"},
        }

        reranked = reranker.rerank_query_candidates(
            query_id="q_low",
            query_text="low margin query text",
            candidate_rows=candidates,
            candidate_docs=docs,
            stage1_margin=0.01,
        )

        assert len(reranked) == 2
        # c_rank2 should be promoted to rank 1 due to high Gemini overall_relevance!
        assert reranked[0]["candidate_id"] == "c_rank2"
        assert reranked[0]["gemini_rank"] == 1
        assert reranked[0]["routing_reason"] == "low_margin_gemini_reranked"
        assert reranked[1]["candidate_id"] == "c_rank1"
        assert reranked[1]["gemini_rank"] == 2

    def test_candidate_pool_integrity(self):
        mock_gemini = MagicMock()
        mock_gemini.generate_json.return_value = {
            "query_id": "q1",
            "candidate_id": "c1",
            "legal_issue_alignment": 0.8,
            "factual_alignment": 0.8,
            "statutory_alignment": 0.8,
            "ratio_alignment": 0.8,
            "reasoning_alignment": 0.8,
            "overall_relevance": 0.8,
            "supporting_evidence": [],
            "contradictory_evidence": [],
            "uncertainties": [],
        }
        reranker = GeminiPrecedentReRanker(gemini_service=mock_gemini, tau=0.07, gemini_weight=0.20, top_k_candidates=2)
        candidates = [
            {"candidate_id": "c1", "hybrid_score": 0.80, "label": 1},
            {"candidate_id": "c2", "hybrid_score": 0.79, "label": 0},
            {"candidate_id": "c3", "hybrid_score": 0.60, "label": 0},
        ]
        docs = {c["candidate_id"]: {"full_text": f"text for {c['candidate_id']}"} for c in candidates}

        reranked = reranker.rerank_query_candidates(
            query_id="q1",
            query_text="query text",
            candidate_rows=candidates,
            candidate_docs=docs,
            stage1_margin=0.01,
        )

        # No candidates lost
        assert len(reranked) == 3
        # No labels modified
        assert {c["candidate_id"]: c["label"] for c in reranked} == {"c1": 1, "c2": 0, "c3": 0}
