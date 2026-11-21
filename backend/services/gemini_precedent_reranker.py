"""
Module C: Gemini-Assisted Precedent Re-Ranking.

Evaluates relevance between a query and candidate legal judgments across 7 dimensions:
1. Legal issue alignment
2. Material factual similarity
3. Statutory relevance
4. Ratio decidendi alignment
5. Judicial reasoning similarity
6. Precedent relationship
7. Final ruling relevance

Integration Principles:
- Preserves Phase 11 margin-gated baseline as the foundation.
- Candidates evaluated are STRICTLY those already retrieved by Stage 1 (Top-3 or Top-5).
- Never introduces un-retrieved candidates.
- High-margin queries (margin >= tau) retain Phase 11 rankings untouched.
- If Gemini is unavailable or fails validation, falls back 100% to Phase 11 baseline.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from backend.services.gemini_service import GeminiService

logger = logging.getLogger("GeminiPrecedentReRanker")


class GeminiPrecedentReRanker:
    """Manages Gemini scoring and integration into the Phase 11 adaptive pipeline."""

    RERANK_SYSTEM_PROMPT = """You are a senior judicial law clerk evaluating precedent relevance for the Supreme Court of India.
Compare the Query Case Description against the Candidate Judgment.

Evaluate alignment across these 5 core dimensions on a scale from 0.0 to 1.0:
- legal_issue_alignment (0.0 - 1.0): Similarity of framed questions of law.
- factual_alignment (0.0 - 1.0): Factual congruency and contextual similarity.
- statutory_alignment (0.0 - 1.0): Overlap in applicable statutory sections / acts.
- ratio_alignment (0.0 - 1.0): Whether the legal principle decided in the candidate controls or guides the query issue.
- reasoning_alignment (0.0 - 1.0): Compatibility of judicial reasoning.

CRITICAL RULES:
1. Scores must be floating point numbers strictly between 0.0 and 1.0.
2. overall_relevance must be between 0.0 and 1.0.
3. Every supporting evidence statement must quote or directly reference the candidate text provided.
4. Note any contradictory distinctions or factual divergences in contradictory_evidence.
5. Return ONLY a valid JSON object matching the schema below.

JSON Schema:
{
  "query_id": "string",
  "candidate_id": "string",
  "legal_issue_alignment": 0.85,
  "factual_alignment": 0.70,
  "statutory_alignment": 0.90,
  "ratio_alignment": 0.80,
  "reasoning_alignment": 0.75,
  "overall_relevance": 0.82,
  "supporting_evidence": ["list of strings (grounded quotes/facts from candidate)"],
  "contradictory_evidence": ["list of strings (material factual or legal distinctions)"],
  "uncertainties": ["list of strings"]
}
"""

    def __init__(
        self,
        gemini_service: Optional[GeminiService] = None,
        tau: float = 0.07,
        gemini_weight: float = 0.15,
        top_k_candidates: int = 5,
    ):
        self.gemini_service = gemini_service or GeminiService()
        self.tau = float(tau)
        self.gemini_weight = float(gemini_weight)
        self.top_k_candidates = int(top_k_candidates)

    @staticmethod
    def validate_schema(data: Dict[str, Any]) -> bool:
        """Validates that Gemini's scoring response contains required fields and valid numeric bounds."""
        if not isinstance(data, dict):
            return False
        score_keys = [
            "legal_issue_alignment",
            "factual_alignment",
            "statutory_alignment",
            "ratio_alignment",
            "reasoning_alignment",
            "overall_relevance",
        ]
        for sk in score_keys:
            if sk not in data:
                return False
            val = data[sk]
            if not isinstance(val, (int, float)):
                return False
            if val < 0.0 or val > 1.0:
                return False

        if not isinstance(data.get("supporting_evidence"), list):
            return False
        return True

    def fallback_score(self, query_id: str, candidate_id: str, baseline_score: float = 0.5) -> Dict[str, Any]:
        """Provides neutral fallback score when Gemini is unavailable."""
        return {
            "query_id": str(query_id),
            "candidate_id": str(candidate_id),
            "legal_issue_alignment": baseline_score,
            "factual_alignment": baseline_score,
            "statutory_alignment": baseline_score,
            "ratio_alignment": baseline_score,
            "reasoning_alignment": baseline_score,
            "overall_relevance": baseline_score,
            "supporting_evidence": [],
            "contradictory_evidence": [],
            "uncertainties": ["Gemini offline; default fallback score assigned."],
        }

    def score_pair(
        self,
        query_id: str,
        query_text: str,
        candidate_id: str,
        candidate_text: str,
        baseline_score: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Calculates Gemini alignment scores for a query-candidate pair.
        """
        # Truncate texts safely
        q_snippet = query_text[:2000].strip()
        c_snippet = candidate_text[:3500].strip()

        user_prompt = f"""Query Case #{query_id}:
\"\"\"
{q_snippet}
\"\"\"

Candidate Judgment #{candidate_id}:
\"\"\"
{c_snippet}
\"\"\"

Assess the precedent relevance and return the structured JSON:"""

        res = self.gemini_service.generate_json(
            prompt=user_prompt,
            system_instruction=self.RERANK_SYSTEM_PROMPT,
            schema_validator=self.validate_schema,
            fallback_factory=lambda: self.fallback_score(query_id, candidate_id, baseline_score),
        )

        if res is None:
            return self.fallback_score(query_id, candidate_id, baseline_score)

        res["query_id"] = str(query_id)
        res["candidate_id"] = str(candidate_id)
        return res

    def rerank_query_candidates(
        self,
        query_id: str,
        query_text: str,
        candidate_rows: List[Dict[str, Any]],
        candidate_docs: Dict[str, Dict[str, Any]],
        stage1_margin: float,
    ) -> List[Dict[str, Any]]:
        """
        Re-ranks top candidates for a single query using margin-gated Gemini fusion.

        If stage1_margin >= tau:
            Preserves Stage-1 ordering untouched (High Confidence).
        If stage1_margin < tau:
            Applies Gemini scoring to Top-K candidates:
            final_score = (1 - gemini_weight) * stage1_score + gemini_weight * gemini_relevance
        """
        if not candidate_rows:
            return []

        # If high-margin query, preserve original order untouched
        if stage1_margin >= self.tau:
            for idx, r in enumerate(candidate_rows, start=1):
                r["gemini_adjusted_score"] = float(r.get("stage1_score", r.get("hybrid_score", 0.0)))
                r["gemini_rank"] = idx
                r["routing_reason"] = "high_margin_preserved"
            return candidate_rows

        # Low-margin query: evaluate Top-K candidates with Gemini
        top_slice = candidate_rows[: self.top_k_candidates]
        other_slice = candidate_rows[self.top_k_candidates :]

        for cand in top_slice:
            cid = str(cand["candidate_id"])
            c_doc = candidate_docs.get(cid, {})
            c_text = c_doc.get("full_text", c_doc.get("snippet", ""))

            baseline = float(cand.get("stage1_score", cand.get("hybrid_score", 0.5)))
            gemini_eval = self.score_pair(
                query_id=query_id,
                query_text=query_text,
                candidate_id=cid,
                candidate_text=c_text,
                baseline_score=baseline,
            )

            cand["gemini_eval"] = gemini_eval
            g_rel = float(gemini_eval.get("overall_relevance", baseline))
            cand["gemini_raw_score"] = g_rel

            # Linear blend with Stage 1 baseline
            cand["gemini_adjusted_score"] = (
                (1.0 - self.gemini_weight) * baseline + self.gemini_weight * g_rel
            )
            cand["routing_reason"] = "low_margin_gemini_reranked"

        # Re-sort top slice descending by gemini_adjusted_score
        top_slice = sorted(top_slice, key=lambda x: x["gemini_adjusted_score"], reverse=True)

        # Candidates outside Top-K receive baseline scores discounted
        for cand in other_slice:
            base = float(cand.get("stage1_score", cand.get("hybrid_score", 0.0)))
            cand["gemini_adjusted_score"] = base - 100.0
            cand["routing_reason"] = "outside_top_k"

        all_candidates = top_slice + other_slice
        for idx, r in enumerate(all_candidates, start=1):
            r["gemini_rank"] = idx

        return all_candidates

