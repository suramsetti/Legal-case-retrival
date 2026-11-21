"""
Module D: Evidence-Based Legal Explanation Service.

Generates comprehensive, evidence-grounded legal explanations for retrieved precedents:
1. Retrieval Justification (Why this case was retrieved)
2. Matching Legal Issues (Framed controversies)
3. Matching Material Facts (Contextual facts)
4. Relevant Ratio Decidendi (Controlling rule of law)
5. Relevant Statutory Provisions (Sections & Acts)
6. Supporting Passages and Chunk References (Verifiable excerpts)
7. Distinctions & Divergences (Important differences between query and candidate)
8. Legal Caveats and Uncertainties (Non-binding nature, jurisdictional limitations)

Safety & Professional Integrity:
- Explicit disclaimer that retrieval similarity does not constitute legal advice.
- No fabricated citations or quotations.
- Grounding in source chunks.
"""

from typing import Dict, Any, List, Optional
import logging
from backend.services.gemini_service import GeminiService

logger = logging.getLogger("LegalExplanationService")


class GeminiExplanationService:
    """Generates structured, verifiable legal explanations for retrieved precedents."""

    EXPLANATION_SYSTEM_PROMPT = """You are a legal research assistant explaining Supreme Court of India precedent retrieval results.
Generate a structured, evidence-grounded explanation showing why the Candidate Precedent was retrieved for the Query Case.

CRITICAL ETHICAL AND ACCURACY REQUIREMENTS:
1. Grounding: All statements of facts and holdings MUST be grounded in the provided candidate text.
2. No Inventions: Never fabricate citations, names of judges, or statute numbers.
3. Distinctions: Highlight where the candidate case DIFFERS from the query case (vital for advocates distinguishing cases).
4. Disclaimer: Include an explicit limitation that this retrieval explanation is an AI assistance tool and does not constitute formal legal opinion or binding judicial authority.
5. Return ONLY a valid JSON object matching the schema below.

JSON Schema:
{
  "query_id": "string",
  "candidate_id": "string",
  "retrieval_justification": "string (succinct summary of why this precedent is legally relevant)",
  "matching_legal_issues": ["list of strings (matching legal questions)"],
  "matching_material_facts": ["list of strings (shared factual predicates)"],
  "relevant_ratio_decidendi": ["list of strings (applicable legal rule from candidate)"],
  "relevant_statutory_provisions": ["list of strings (shared statutes/sections)"],
  "supporting_passages": [
    {
      "source_reference": "string (e.g. 'Paragraph 14' or 'Holding Section')",
      "excerpt": "string (short verbatim quote from judgment)"
    }
  ],
  "material_differences": ["list of strings (how this case is distinguishable from the query)"],
  "limitations_and_uncertainty": ["list of strings (caveats, procedural posture differences)"],
  "disclaimer": "This explanation is generated for legal research and case retrieval assistance only and does not constitute legal advice or an authoritative judicial holding."
}
"""

    def __init__(self, gemini_service: Optional[GeminiService] = None):
        self.gemini_service = gemini_service or GeminiService()

    @staticmethod
    def validate_schema(data: Dict[str, Any]) -> bool:
        """Validates that explanation JSON conforms to requirements."""
        if not isinstance(data, dict):
            return False
        required_keys = [
            "query_id",
            "candidate_id",
            "retrieval_justification",
            "matching_legal_issues",
            "matching_material_facts",
            "relevant_ratio_decidendi",
            "relevant_statutory_provisions",
            "supporting_passages",
            "material_differences",
            "limitations_and_uncertainty",
            "disclaimer",
        ]
        for k in required_keys:
            if k not in data:
                return False
        if not isinstance(data["supporting_passages"], list):
            return False
        return True

    def fallback_explanation(
        self,
        query_id: str,
        candidate_id: str,
        query_text: str,
        candidate_text: str,
        hybrid_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Provides fallback explanation using local text snippets when Gemini is offline."""
        return {
            "query_id": str(query_id),
            "candidate_id": str(candidate_id),
            "retrieval_justification": (
                f"Retrieved by Hybrid BM25/TF-IDF/InLegalBERT scoring pipeline with hybrid score {hybrid_score:.4f}."
            ),
            "matching_legal_issues": ["Statistical and dense semantic textual similarity."],
            "matching_material_facts": [candidate_text[:200] + "..."] if candidate_text else [],
            "relevant_ratio_decidendi": ["Detailed neural discourse extraction unavailable in offline fallback mode."],
            "relevant_statutory_provisions": [],
            "supporting_passages": [
                {
                    "source_reference": "Document Opening",
                    "excerpt": candidate_text[:150] if candidate_text else "",
                }
            ],
            "material_differences": ["Detailed comparative divergence analysis unavailable in offline fallback mode."],
            "limitations_and_uncertainty": [
                "Generated via local retrieval heuristics while LLM explanation service is offline."
            ],
            "disclaimer": (
                "This explanation is generated for legal research and case retrieval assistance only "
                "and does not constitute legal advice or an authoritative judicial holding."
            ),
        }

    def explain_precedent(
        self,
        query_id: str,
        candidate_id: str,
        query_text: str,
        candidate_text: str,
        hybrid_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Generates 8-point evidence-grounded legal explanation for a query-precedent pair.
        """
        q_snip = query_text[:2200].strip()
        c_snip = candidate_text[:3800].strip()

        user_prompt = f"""Query Case #{query_id}:
\"\"\"
{q_snip}
\"\"\"

Retrieved Precedent Judgment #{candidate_id} (Hybrid Score: {hybrid_score:.4f}):
\"\"\"
{c_snip}
\"\"\"

Generate the structured evidence-grounded explanation JSON:"""

        res = self.gemini_service.generate_json(
            prompt=user_prompt,
            system_instruction=self.EXPLANATION_SYSTEM_PROMPT,
            schema_validator=self.validate_schema,
            fallback_factory=lambda: self.fallback_explanation(
                query_id, candidate_id, query_text, candidate_text, hybrid_score
            ),
        )

        if res is None:
            return self.fallback_explanation(
                query_id, candidate_id, query_text, candidate_text, hybrid_score
            )

        res["query_id"] = str(query_id)
        res["candidate_id"] = str(candidate_id)
        return res

