"""
Phase 19B: Legal Section Chunker.

Implements section-aware chunking for long Indian Supreme Court judgments:
- Dissects documents into section-aware chunks rather than truncating at the document head.
- Ensures representation from critical judicial sections:
  1. Court Reasoning & Ratio Decidendi
  2. Legal Provisions & Statutes
  3. Counsel Arguments
  4. Core Facts
- Eliminates silent head-truncation bias where only introductory facts were captured.
- Tracks exact section source and rhetorical role for each chunk.
"""

from typing import List, Dict, Any, Optional
from src.preprocessing.phase19_section_segmenter import LegalSectionSegmenter


class LegalSectionChunker:
    """
    Generates section-aware chunks from segmented legal judgments.
    """

    DEFAULT_SECTION_BUDGETS = {
        "court_reasoning": 3,
        "facts": 2,
        "arguments": 2,
        "legal_provisions": 1,
        "issues": 1,
        "procedural_history": 1,
    }

    def __init__(
        self,
        max_chunk_words: int = 80,
        overlap_words: int = 15,
        total_chunks_per_doc: int = 8,
        section_budgets: Optional[Dict[str, int]] = None,
    ):
        self.max_chunk_words = max_chunk_words
        self.overlap_words = overlap_words
        self.total_chunks_per_doc = total_chunks_per_doc
        self.section_budgets = section_budgets or self.DEFAULT_SECTION_BUDGETS
        self.segmenter = LegalSectionSegmenter()

    def _chunk_text(self, text: str, max_words: int, overlap: int) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        step = max(1, max_words - overlap)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + max_words])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def chunk_document(
        self, doc_id: str, text: str, segmented: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates structured section-aware chunks for a single document.
        Returns a list of dicts:
        [
            {
                'doc_id': '...',
                'chunk_idx': 0,
                'section': 'court_reasoning',
                'text': '...',
                'word_count': 78
            },
            ...
        ]
        """
        if segmented is None:
            segmented = self.segmenter.segment(text)

        chunks_by_section: Dict[str, List[str]] = {}
        for sec, sec_text in segmented.items():
            if sec_text.strip():
                chunks_by_section[sec] = self._chunk_text(
                    sec_text, self.max_chunk_words, self.overlap_words
                )
            else:
                chunks_by_section[sec] = []

        # Select chunks prioritizing key sections according to budget
        selected_chunks = []
        global_idx = 0

        # Pass 1: Select up to budget for each section
        for sec, budget in self.section_budgets.items():
            available = chunks_by_section.get(sec, [])
            n_take = min(len(available), budget)
            for c in available[:n_take]:
                selected_chunks.append({
                    "doc_id": doc_id,
                    "chunk_idx": global_idx,
                    "section": sec,
                    "text": c,
                    "word_count": len(c.split()),
                })
                global_idx += 1
                if len(selected_chunks) >= self.total_chunks_per_doc:
                    break
            if len(selected_chunks) >= self.total_chunks_per_doc:
                break

        # Pass 2: If remaining capacity exists, fill from court_reasoning, then facts, then others
        if len(selected_chunks) < self.total_chunks_per_doc:
            fill_order = ["court_reasoning", "facts", "arguments", "legal_provisions"]
            for sec in fill_order:
                available = chunks_by_section.get(sec, [])
                budget = self.section_budgets.get(sec, 0)
                remaining = available[budget:]
                for c in remaining:
                    selected_chunks.append({
                        "doc_id": doc_id,
                        "chunk_idx": global_idx,
                        "section": sec,
                        "text": c,
                        "word_count": len(c.split()),
                    })
                    global_idx += 1
                    if len(selected_chunks) >= self.total_chunks_per_doc:
                        break
                if len(selected_chunks) >= self.total_chunks_per_doc:
                    break

        return selected_chunks

