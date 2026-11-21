"""
Module A: Legal Query Understanding and Expansion.

Analyzes natural-language legal queries using Gemini to extract:
- Core legal issue
- Relevant legal concepts
- Potential statutory provisions (e.g. IPC, CrPC, Contract Act, Arbitration Act)
- Legal terminology and domain synonyms
- Factual elements
- Alternative legal interpretations
- Search expansion terms

Adheres to strict legal data integrity rules:
- Preserves the original query untouched.
- Never treats proposed statutes as binding law.
- Flags ambiguities in uncertainties list.
"""

from typing import Dict, Any, List, Optional
import logging
from backend.services.gemini_service import GeminiService

logger = logging.getLogger("LegalQueryExpansion")


class LegalQueryExpander:
    """Manages legal query understanding and structured query expansion."""

    EXPANSION_SYSTEM_PROMPT = """You are an expert Indian Legal Scholar and Judicial Clerk assisting in Prior Case Retrieval.
Analyze the user's legal case query description and extract the structured legal dimensions strictly according to Indian Law (Supreme Court / High Courts).

IMPORTANT INSTRUCTIONS:
1. Preserve the user's original legal intent. Never fabricate statutory provisions or case citations.
2. Identify potential statutory provisions (e.g., Indian Penal Code 1860, Code of Criminal Procedure 1973, Indian Evidence Act 1872, Arbitration and Conciliation Act 1996, Constitution of India). If recent penal reforms apply (BNS 2023 / BNSS 2023), note them alongside classic IPC provisions.
3. Clearly state uncertainties or ambiguities in the query.
4. Return ONLY a valid JSON object matching the exact schema below.

JSON Schema:
{
  "original_query": "string (the exact original query)",
  "legal_issue": "string (succinct statement of the primary legal controversy)",
  "legal_concepts": ["list of strings (key legal doctrines, e.g., 'mens rea', 'quashment under Section 482')"],
  "potential_statutes": ["list of strings (explicit statutes/sections, e.g., 'Section 302 IPC', 'Section 34 IPC')"],
  "legal_synonyms": ["list of strings (alternative legal phrasing, e.g., 'homicide', 'culpable homicide', 'unlawful assembly')"],
  "factual_elements": ["list of strings (key factual predicates mentioned in query)"],
  "expanded_queries": ["list of 2-3 search query strings optimized for legal case retrieval"],
  "uncertainties": ["list of strings (factual ambiguities or jurisdictional caveats)"]
}
"""

    def __init__(self, gemini_service: Optional[GeminiService] = None):
        self.gemini_service = gemini_service or GeminiService()

    @staticmethod
    def validate_schema(data: Dict[str, Any]) -> bool:
        """Validates that the returned JSON contains all required fields with proper types."""
        required_keys = [
            "original_query",
            "legal_issue",
            "legal_concepts",
            "potential_statutes",
            "legal_synonyms",
            "factual_elements",
            "expanded_queries",
            "uncertainties",
        ]
        if not isinstance(data, dict):
            return False
        for k in required_keys:
            if k not in data:
                return False
        if not isinstance(data["legal_concepts"], list):
            return False
        if not isinstance(data["expanded_queries"], list):
            return False
        return True

    def fallback_expansion(self, original_query: str) -> Dict[str, Any]:
        """Provides a safe, rule-based fallback expansion when Gemini is unavailable."""
        return {
            "original_query": original_query,
            "legal_issue": "Analysis unavailable (fallback mode)",
            "legal_concepts": [],
            "potential_statutes": [],
            "legal_synonyms": [],
            "factual_elements": [original_query[:150]],
            "expanded_queries": [original_query],
            "uncertainties": ["Gemini service unavailable; using raw query fallback."],
        }

    def expand_query(self, query_text: str) -> Dict[str, Any]:
        """
        Expands a natural-language legal query using Gemini with caching and validation.
        """
        q = query_text.strip()
        if not q:
            return self.fallback_expansion("")

        user_prompt = f"Legal Query Description:\n\"\"\"{q}\"\"\"\n\nReturn the structured legal analysis JSON:"

        result = self.gemini_service.generate_json(
            prompt=user_prompt,
            system_instruction=self.EXPANSION_SYSTEM_PROMPT,
            schema_validator=self.validate_schema,
            fallback_factory=lambda: self.fallback_expansion(q),
        )

        if result is None:
            return self.fallback_expansion(q)

        # Guarantee original query is explicitly set
        result["original_query"] = q
        return result

