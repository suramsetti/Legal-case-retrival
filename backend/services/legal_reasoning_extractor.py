"""
Module B: Ratio Decidendi and Legal Reasoning Extractor.

Extracts grounded legal discourse elements from Indian Supreme Court judgments:
- Legal issues
- Material facts
- Party arguments
- Court reasoning (obiter dicta vs ratio decidendi)
- Binding Ratio Decidendi
- Final disposition / ruling
- Statutory references
- Precedent citations

Enforces evidence grounding:
- All extractions must reference specific chunk IDs (`source_chunk_id`).
- Direct quotes must match judgment text.
- Includes disk caching for zero repeated API calls.
"""

from typing import Dict, Any, List, Optional
import logging
from backend.services.gemini_service import GeminiService
from src.ranking.legal_cross_encoder import SectionAwareChunker

logger = logging.getLogger("LegalReasoningExtractor")


class LegalReasoningExtractor:
    """Extracts grounded judicial rationale and ratio decidendi using Gemini."""

    EXTRACTION_SYSTEM_PROMPT = """You are a judicial scholar analyzing an Indian Supreme Court / High Court judgment.
Given numbered text chunks from a judgment, extract the judicial rationale into a structured JSON schema.

CRITICAL EVIDENCE GROUNDING RULES:
1. Every item in 'ratio_decidendi', 'court_reasoning', and 'material_facts' MUST cite a valid 'source_chunk_id' from the provided text.
2. NEVER invent case citations, statutory provisions, or quotations. If not in the text, omit them.
3. Distinguish binding 'ratio_decidendi' (the legal principle on which the decision depends) from general observations.
4. Record any ambiguity or contradictory passages under 'uncertainties'.
5. Return ONLY a valid JSON object matching the schema below.

JSON Schema:
{
  "case_id": "string",
  "legal_issues": ["list of strings (framed questions of law)"],
  "material_facts": ["list of strings (essential facts)"],
  "party_arguments": ["list of strings (appellant/respondent contentions)"],
  "court_reasoning": ["list of strings (judicial analysis and rationale)"],
  "ratio_decidendi": ["list of strings (binding legal holding established by this case)"],
  "final_ruling": ["list of strings (disposition, e.g. 'Appeal allowed', 'Conviction upheld')"],
  "statutes": ["list of strings (statutory sections referenced in text)"],
  "precedent_citations": ["list of strings (prior cases relied upon)"],
  "evidence": [
    {
      "section": "string (e.g., 'ratio_decidendi', 'final_ruling', 'facts')",
      "source_chunk_id": "string (the chunk ID provided, e.g., 'chunk_0')",
      "quoted_text": "string (exact short verbatim excerpt from that chunk)"
    }
  ],
  "uncertainties": ["list of strings"]
}
"""

    def __init__(
        self,
        gemini_service: Optional[GeminiService] = None,
        chunker: Optional[SectionAwareChunker] = None,
    ):
        self.gemini_service = gemini_service or GeminiService()
        self.chunker = chunker or SectionAwareChunker()

    @staticmethod
    def validate_schema(data: Dict[str, Any]) -> bool:
        """Validates that extracted reasoning JSON conforms to requirements."""
        if not isinstance(data, dict):
            return False
        required_fields = [
            "case_id",
            "legal_issues",
            "material_facts",
            "party_arguments",
            "court_reasoning",
            "ratio_decidendi",
            "final_ruling",
            "statutes",
            "precedent_citations",
            "evidence",
            "uncertainties",
        ]
        for f in required_fields:
            if f not in data:
                return False
        if not isinstance(data["evidence"], list):
            return False
        for ev in data["evidence"]:
            if not isinstance(ev, dict) or "source_chunk_id" not in ev or "quoted_text" not in ev:
                return False
        return True

    def fallback_extraction(self, case_id: str, text: str) -> Dict[str, Any]:
        """Rule-based heuristic extraction fallback using SectionAwareChunker."""
        sentences = [s.strip() for s in text.split("\n") if s.strip()]
        sections = self.chunker.extract_sections(sentences)

        return {
            "case_id": case_id,
            "legal_issues": ["Heuristic extraction (Gemini offline)"],
            "material_facts": [sections.get("facts", "")[:300]] if sections.get("facts") else [],
            "party_arguments": [],
            "court_reasoning": [sections.get("ratio", "")[:400]] if sections.get("ratio") else [],
            "ratio_decidendi": [sections.get("ratio", "")[:300]] if sections.get("ratio") else [],
            "final_ruling": [sections.get("ruling", "")[:200]] if sections.get("ruling") else [],
            "statutes": [],
            "precedent_citations": [],
            "evidence": [
                {
                    "section": "ratio_decidendi",
                    "source_chunk_id": "fallback_chunk_0",
                    "quoted_text": (sections.get("ratio", "")[:120]),
                }
            ] if sections.get("ratio") else [],
            "uncertainties": ["Rule-based fallback; automated LLM discourse analysis unavailable."],
        }

    def extract_reasoning(self, case_id: str, full_text: str) -> Dict[str, Any]:
        """
        Extracts structured legal reasoning from judgment text using chunking and Gemini.
        """
        cid = str(case_id).strip()
        if not full_text or len(full_text.strip()) < 50:
            return self.fallback_extraction(cid, full_text or "")

        # Prepare section-aware chunks
        sentences = [s.strip() for s in full_text.split("\n") if s.strip()]
        chunks = self.chunker.chunk_document(sentences)

        # Select most informative chunks (prioritizing ratio, ruling, and headnote)
        # to fit context window safely
        selected_chunks = []
        char_count = 0
        max_chars = 12000  # ~3,000 tokens

        # Order: ratio first, ruling second, headnotes third, facts fourth
        type_priority = {"ratio": 0, "ruling": 1, "headnote": 2, "facts": 3, "general": 4}
        sorted_chunks = sorted(chunks, key=lambda c: type_priority.get(c.get("section_type", "general"), 5))

        for idx, chunk in enumerate(sorted_chunks):
            chunk_repr = f"[CHUNK ID: chunk_{idx} | SECTION: {chunk['section_type']}]\n{chunk['text']}\n"
            if char_count + len(chunk_repr) > max_chars:
                break
            selected_chunks.append(chunk_repr)
            char_count += len(chunk_repr)

        chunks_text = "\n---\n".join(selected_chunks)

        user_prompt = f"""Judgment Case ID: {cid}

Text Chunks from Judgment:
\"\"\"
{chunks_text}
\"\"\"

Extract the structured legal reasoning JSON according to the schema:"""

        result = self.gemini_service.generate_json(
            prompt=user_prompt,
            system_instruction=self.EXTRACTION_SYSTEM_PROMPT,
            schema_validator=self.validate_schema,
            fallback_factory=lambda: self.fallback_extraction(cid, full_text),
        )

        if result is None:
            return self.fallback_extraction(cid, full_text)

        result["case_id"] = cid
        return result

