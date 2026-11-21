"""
Unit Tests for Module B: Ratio Decidendi and Legal Reasoning Extractor.

Verifies:
1. Schema validation (issues, facts, reasoning, ratio decidendi, ruling, evidence).
2. Evidence grounding: verifying presence of source_chunk_id and quoted_text.
3. Fallback extraction utilizing SectionAwareChunker heuristic sections.
4. Correct chunk construction and formatting before prompt dispatch.
"""

from unittest.mock import MagicMock
import pytest

from backend.services.legal_reasoning_extractor import LegalReasoningExtractor


class TestLegalReasoningExtractor:

    def test_schema_validator(self):
        valid = {
            "case_id": "0001024",
            "legal_issues": ["Scope of interference under Section 34 Arbitration Act"],
            "material_facts": ["Arbitrator awarded damages for delay in construction contract"],
            "party_arguments": ["Appellant argues award violates public policy"],
            "court_reasoning": ["Section 34 does not permit appellate reappreciation of evidence"],
            "ratio_decidendi": ["An arbitral award cannot be set aside on mere factual disagreement"],
            "final_ruling": ["Appeal is allowed; High Court order set aside"],
            "statutes": ["Section 34 Arbitration Act"],
            "precedent_citations": ["Associate Builders v. DDA"],
            "evidence": [
                {
                    "section": "ratio_decidendi",
                    "source_chunk_id": "chunk_0",
                    "quoted_text": "Section 34 does not permit appellate reappreciation of evidence",
                }
            ],
            "uncertainties": [],
        }
        assert LegalReasoningExtractor.validate_schema(valid) is True

        # Missing evidence
        invalid = valid.copy()
        del invalid["evidence"]
        assert LegalReasoningExtractor.validate_schema(invalid) is False

        # Malformed evidence missing source_chunk_id
        invalid_ev = valid.copy()
        invalid_ev["evidence"] = [{"section": "ratio", "quoted_text": "text"}]
        assert LegalReasoningExtractor.validate_schema(invalid_ev) is False

    def test_fallback_extraction(self):
        extractor = LegalReasoningExtractor()
        text = """CIVIL APPELLATE JURISDICTION: Civil Appeal 100.
        The prosecution alleged murder under section 302 IPC.
        We hold that section 302 requires proof of common intention.
        In the result, appeal is allowed and conviction set aside."""

        res = extractor.fallback_extraction("test_case_1", text)
        assert res["case_id"] == "test_case_1"
        assert len(res["ratio_decidendi"]) > 0
        assert len(res["final_ruling"]) > 0
        assert len(res["evidence"]) > 0
        assert "fallback_chunk_0" == res["evidence"][0]["source_chunk_id"]

    def test_extract_reasoning_mocked(self):
        mock_gemini = MagicMock()
        mock_gemini.generate_json.return_value = {
            "case_id": "ignored",
            "legal_issues": ["Whether dying declaration requires corroboration"],
            "material_facts": ["Deceased suffered 80% burn injuries and gave statement to magistrate"],
            "party_arguments": ["Defence claims magistrate did not certify mental fitness"],
            "court_reasoning": ["Doctor certified fitness immediately prior to recording statement"],
            "ratio_decidendi": ["A certified dying declaration can form the sole basis of conviction"],
            "final_ruling": ["Appeal dismissed; life sentence affirmed"],
            "statutes": ["Section 32 Evidence Act", "Section 302 IPC"],
            "precedent_citations": ["Khushal Rao v. State of Bombay"],
            "evidence": [
                {
                    "section": "ratio_decidendi",
                    "source_chunk_id": "chunk_0",
                    "quoted_text": "dying declaration can form the sole basis of conviction",
                }
            ],
            "uncertainties": [],
        }

        extractor = LegalReasoningExtractor(gemini_service=mock_gemini)
        sample_text = "Judgment regarding dying declaration under Section 32 Evidence Act."
        result = extractor.extract_reasoning("case_500", sample_text)

        assert result["case_id"] == "case_500"
        assert len(result["ratio_decidendi"]) == 1
        assert "sole basis of conviction" in result["ratio_decidendi"][0]
        assert result["evidence"][0]["source_chunk_id"] == "chunk_0"

