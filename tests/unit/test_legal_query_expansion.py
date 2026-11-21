"""
Unit Tests for Module A: Legal Query Understanding & Expansion.

Verifies:
1. Schema validation with complete and missing keys.
2. Safe fallback when service is offline or response is invalid.
3. Preservation of original query untouched.
4. Parsing of legal concepts, statutory provisions, and expanded queries.
"""

from unittest.mock import MagicMock
import pytest

from backend.services.legal_query_expansion import LegalQueryExpander


class TestLegalQueryExpander:

    def test_schema_validator(self):
        valid = {
            "original_query": "Criminal appeal against conviction under Section 302 IPC",
            "legal_issue": "Whether circumstantial evidence establishes guilt beyond reasonable doubt",
            "legal_concepts": ["circumstantial evidence", "chain of circumstances"],
            "potential_statutes": ["Section 302 IPC", "Section 3 IPC"],
            "legal_synonyms": ["homicide", "mens rea"],
            "factual_elements": ["death caused by weapon", "last seen together"],
            "expanded_queries": [
                "circumstantial evidence murder Section 302 IPC",
                "last seen theory conviction chain of evidence",
            ],
            "uncertainties": ["Exact weapon recovery is disputed"],
        }
        assert LegalQueryExpander.validate_schema(valid) is True

        # Missing required field
        invalid = valid.copy()
        del invalid["legal_concepts"]
        assert LegalQueryExpander.validate_schema(invalid) is False

        # Wrong type
        invalid_type = valid.copy()
        invalid_type["legal_concepts"] = "not a list"
        assert LegalQueryExpander.validate_schema(invalid_type) is False

    def test_fallback_expansion(self):
        expander = LegalQueryExpander()
        query = "Arbitration petition under Section 34 challenging arbitral award"
        res = expander.fallback_expansion(query)

        assert res["original_query"] == query
        assert isinstance(res["legal_concepts"], list)
        assert isinstance(res["expanded_queries"], list)
        assert query in res["expanded_queries"]
        assert len(res["uncertainties"]) > 0

    def test_expand_query_mocked(self):
        mock_gemini = MagicMock()
        mock_gemini.generate_json.return_value = {
            "original_query": "ignored",
            "legal_issue": "Quashing of FIR under Section 482 CrPC for commercial dispute",
            "legal_concepts": ["abuse of judicial process", "civil dispute criminalized"],
            "potential_statutes": ["Section 482 CrPC", "Section 420 IPC"],
            "legal_synonyms": ["cheating", "inherent powers of High Court"],
            "factual_elements": ["breach of contract"],
            "expanded_queries": [
                "quash FIR commercial dispute Section 482 CrPC",
                "inherent jurisdiction High Court breach of contract",
            ],
            "uncertainties": [],
        }

        expander = LegalQueryExpander(gemini_service=mock_gemini)
        query = "Petition to quash criminal proceedings in commercial breach"
        result = expander.expand_query(query)

        assert result["original_query"] == query  # Original query preserved
        assert result["legal_issue"].startswith("Quashing of FIR")
        assert "Section 482 CrPC" in result["potential_statutes"]
        assert len(result["expanded_queries"]) == 2

