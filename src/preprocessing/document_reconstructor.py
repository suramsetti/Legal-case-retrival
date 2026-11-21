"""Document reconstructor module.

Reconstructs full legal judgments from sentence arrays while safely handling
nulls, empty items, and encoding anomalies without altering sentence order or inventing text.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from src.preprocessing.text_cleaner import LegalTextCleaner


class DocumentReconstructor:
    """Reconstructs coherent documents from sentence lists."""

    def __init__(self, cleaner: Optional[LegalTextCleaner] = None, join_delimiter: str = " "):
        self.cleaner = cleaner or LegalTextCleaner()
        self.join_delimiter = join_delimiter

    def reconstruct(self, raw_sentences: Any) -> Dict[str, Any]:
        """Reconstruct full document from an iterable of sentence strings.

        Args:
            raw_sentences: List, tuple, or numpy array of sentence strings.

        Returns:
            Dict containing:
                - full_text: Reconstructed document string.
                - clean_sentences: List of cleaned, non-empty sentences.
                - sentence_count: Number of cleaned sentences.
                - original_sentence_count: Original count of elements.
                - empty_sentences_count: Number of null/whitespace-only sentences skipped.
                - character_count: Character count of reconstructed text.
                - word_count: Word count of reconstructed text.
        """
        if raw_sentences is None:
            return {
                "full_text": "",
                "clean_sentences": [],
                "sentence_count": 0,
                "original_sentence_count": 0,
                "empty_sentences_count": 0,
                "character_count": 0,
                "word_count": 0,
            }

        # Convert numpy array or other iterables safely
        if isinstance(raw_sentences, np.ndarray):
            raw_list = raw_sentences.tolist()
        elif isinstance(raw_sentences, (list, tuple)):
            raw_list = list(raw_sentences)
        else:
            try:
                raw_list = list(raw_sentences)
            except Exception:
                raw_list = [str(raw_sentences)]

        original_count = len(raw_list)
        clean_sentences: List[str] = []
        empty_count = 0

        for item in raw_list:
            if item is None:
                empty_count += 1
                continue
            cleaned = self.cleaner.clean_sentence(str(item))
            if cleaned:
                clean_sentences.append(cleaned)
            else:
                empty_count += 1

        full_text = self.join_delimiter.join(clean_sentences)

        return {
            "full_text": full_text,
            "clean_sentences": clean_sentences,
            "sentence_count": len(clean_sentences),
            "original_sentence_count": original_count,
            "empty_sentences_count": empty_count,
            "character_count": len(full_text),
            "word_count": len(full_text.split()) if full_text else 0,
        }

