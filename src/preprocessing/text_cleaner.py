"""Text cleaner module for legal document normalization.

Preserves legal terminology, citations, statutory references, case numbers,
and dates while normalizing Unicode, whitespace, and control characters.
"""

import re
import unicodedata
from typing import Optional


class LegalTextCleaner:
    """Normalizes legal text while preserving legal entities, citations, and structural numbers."""

    def __init__(self, normalize_unicode_form: str = "NFKC"):
        self.normalize_unicode_form = normalize_unicode_form
        
        # Regex to match non-standard whitespace characters (non-breaking, zero-width, etc.)
        self._special_spaces_pattern = re.compile(
            r"[\u00a0\u1680\u2000-\u200b\u202f\u205f\u3000\ufeff]"
        )
        
        # Control characters except standard newline (\n) and tab (\t)
        self._control_char_pattern = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
        
        # Multiple spaces or tabs (within a line)
        self._multi_space_pattern = re.compile(r"[^\S\n\r]+")
        
        # Multiple newlines
        self._multi_newline_pattern = re.compile(r"\n{3,}")

    def clean(self, text: Optional[str]) -> str:
        """Clean and normalize a single text string.

        Args:
            text: Raw input string.

        Returns:
            Cleaned and normalized text string.
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        if not text.strip():
            return ""

        # 1. Unicode normalization (NFKC decomposes compatibility chars & standardizes glyphs)
        cleaned = unicodedata.normalize(self.normalize_unicode_form, text)

        # 2. Replace special / zero-width / non-breaking spaces with standard space
        cleaned = self._special_spaces_pattern.sub(" ", cleaned)

        # 3. Strip unprintable control characters (retaining newlines and tabs)
        cleaned = self._control_char_pattern.sub("", cleaned)

        # 4. Standardize quotes and dashes gently without altering numbers or citations
        # Standardize smart quotes to ASCII equivalents
        cleaned = cleaned.replace("“", '"').replace("”", '"')
        cleaned = cleaned.replace("‘", "'").replace("’", "'").replace("`", "'")
        
        # Standardize long dashes (em-dash, en-dash) with spaces or hyphen
        cleaned = cleaned.replace("—", " - ").replace("–", "-")

        # 5. Normalize whitespace within lines
        cleaned = self._multi_space_pattern.sub(" ", cleaned)

        # 6. Normalize multiple newlines (max 2 consecutive newlines)
        cleaned = self._multi_newline_pattern.sub("\n\n", cleaned)

        # 7. Strip leading and trailing whitespace
        return cleaned.strip()

    def clean_sentence(self, sentence: Optional[str]) -> str:
        """Clean an individual sentence entry.

        Ensures that sentence-level elements remain intact and trailing newlines are removed.
        """
        cleaned = self.clean(sentence)
        # Collapse any internal newlines within a single sentence entry into a space
        cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
        return cleaned.strip()

