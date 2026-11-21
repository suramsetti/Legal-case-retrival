"""
Phase 19B: Legal Section Segmenter.

Deconstructs Indian Supreme Court judgments into canonical rhetorical sections:
- facts: Factual matrix, occurrence, background, pleadings
- procedural_history: Lower court judgments, High Court rulings, appeal trail
- issues: Questions framed, points for determination
- arguments: Counsel submissions, appellant/respondent contentions
- legal_provisions: Statutory references, constitutional provisions, rules
- court_reasoning: Judicial analysis, ratio decidendi, precedent application, evidence evaluation

Designed for robustness:
- Detects explicit section headings when available.
- Classifies paragraphs by discourse markers and legal phrases when headings are omitted.
- Provides structural zone fallback so every document achieves 100% section coverage.
- Strictly excludes concluding dispositive rulings and proof sentence markers.
"""

import re
from typing import Dict, List, Tuple, Any


# Regex patterns for explicit headings or strong paragraph openings
SECTION_HEADER_PATTERNS = {
    "facts": [
        r"^\s*(?:brief\s+)?facts(?:\s+of\s+the\s+case)?\s*[:\.\-]",
        r"^\s*factual\s+(?:matrix|background)\s*[:\.\-]",
        r"^\s*background(?:\s+facts)?\s*[:\.\-]",
        r"^\s*case\s+of\s+the\s+prosecution\s*[:\.\-]",
        r"^\s*case\s+of\s+the\s+plaintiff\s*[:\.\-]",
    ],
    "procedural_history": [
        r"^\s*procedural\s+history\s*[:\.\-]",
        r"^\s*(?:proceedings\s+before|decision\s+of)\s+the\s+(?:high|trial)\s+court\s*[:\.\-]",
        r"^\s*impugned\s+(?:order|judgment)\s*[:\.\-]",
    ],
    "issues": [
        r"^\s*issues?\s*(?:for\s+determination|framed|involved)?\s*[:\.\-]",
        r"^\s*points?\s+(?:for\s+consideration|determination)\s*[:\.\-]",
        r"^\s*questions?\s+(?:of\s+law|framed|arising)\s*[:\.\-]",
    ],
    "arguments": [
        r"^\s*arguments?\s*[:\.\-]",
        r"^\s*(?:submissions?|contentions?|arguments?)\s*(?:of|by)?\s*(?:the\s+)?(?:appellants?|respondents?|counsel)\s*[:\.\-]",
        r"^\s*(?:rival\s+)?contentions\s*[:\.\-]",
        r"^\s*submissions\s*[:\.\-]",
    ],
    "legal_provisions": [
        r"^\s*(?:relevant\s+)?(?:statutory\s+)?(?:provisions?|sections?|articles?)\s*[:\.\-]",
        r"^\s*scheme\s+of\s+the\s+act\s*[:\.\-]",
    ],
    "court_reasoning": [
        r"^\s*(?:court\s+)?reasoning\s*[:\.\-]",
        r"^\s*discussion\s*(?:and\s+analysis)?\s*[:\.\-]",
        r"^\s*analysis\s*[:\.\-]",
        r"^\s*consideration\s*(?:of\s+the\s+court)?\s*[:\.\-]",
        r"^\s*findings?\s*[:\.\-]",
    ],
}

# In-paragraph discourse cues
CUE_PATTERNS = {
    "facts": re.compile(
        r"\b(?:the\s+factual\s+matrix|prosecution\s+alleged|complainant\s+filed|suit\s+was\s+filed|incident\s+occurred|on\s+the\s+alleged\s+date|deceased\s+was|appellant\s+was\s+appointed)\b",
        re.IGNORECASE,
    ),
    "procedural_history": re.compile(
        r"\b(?:learned\s+sessions\s+judge|trial\s+court\s+convicted|trial\s+court\s+dismissed|high\s+court\s+(?:affirmed|reversed|held)|impugned\s+judgment\s+dated|appeal\s+arises\s+from|special\s+leave\s+petition)\b",
        re.IGNORECASE,
    ),
    "issues": re.compile(
        r"\b(?:question\s+arising\s+for\s+consideration|point\s+for\s+decision|issue\s+to\s+be\s+determined|the\s+short\s+question|formulated\s+the\s+question)\b",
        re.IGNORECASE,
    ),
    "arguments": re.compile(
        r"\b(?:learned\s+(?:senior\s+)?counsel\s+(?:appearing\s+for|submitted|contended|argued)|appellant\s+urged|respondent\s+supported|it\s+was\s+contended\s+that|learned\s+additional\s+solicitor\s+general)\b",
        re.IGNORECASE,
    ),
    "legal_provisions": re.compile(
        r"\b(?:section\s+\d+|article\s+\d+|clause\s+\d+|code\s+of\s+criminal\s+procedure|indian\s+penal\s+code|code\s+of\s+civil\s+procedure|constitution\s+of\s+india|rules\s+framed\s+under)\b",
        re.IGNORECASE,
    ),
    "court_reasoning": re.compile(
        r"\b(?:in\s+our\s+considered\s+(?:view|opinion)|we\s+have\s+given\s+our\s+anxious\s+consideration|settled\s+law\s+that|this\s+court\s+in\s+the\s+case\s+of|a\s+three-judge\s+bench\s+observed|we\s+are\s+unable\s+to\s+accept|ratio\s+decidendi|having\s+heard\s+the\s+learned\s+counsel|on\s+a\s+careful\s+perusal\s+of\s+the\s+record)\b",
        re.IGNORECASE,
    ),
}

DISPOSITIVE_MARKERS = [
    "[MASKED_CONCLUDING_DISPOSITION]",
    "[MASKED_PROOF_SENTENCE]",
    "[MASKED_DISPOSITIVE_RULING]",
]


class LegalSectionSegmenter:
    """
    Segments a masked legal judgment into canonical rhetorical sections.
    """

    CANONICAL_SECTIONS = [
        "facts",
        "procedural_history",
        "issues",
        "arguments",
        "legal_provisions",
        "court_reasoning",
    ]

    def __init__(self):
        self.compiled_headers = {}
        for sec, patterns in SECTION_HEADER_PATTERNS.items():
            self.compiled_headers[sec] = [
                re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns
            ]

    def clean_text_for_segmentation(self, text: str) -> str:
        """Strip dispositive markers to guarantee zero leakage."""
        cleaned = text
        for marker in DISPOSITIVE_MARKERS:
            cleaned = cleaned.replace(marker, " ")
        return cleaned

    def segment(self, text: str) -> Dict[str, str]:
        """
        Segments judgment text into dictionary of rhetorical sections:
        {
            'facts': '...',
            'procedural_history': '...',
            'issues': '...',
            'arguments': '...',
            'legal_provisions': '...',
            'court_reasoning': '...'
        }
        """
        cleaned = self.clean_text_for_segmentation(text)
        paragraphs = [p.strip() for p in cleaned.split("\n") if p.strip()]

        if not paragraphs:
            return {s: "" for s in self.CANONICAL_SECTIONS}

        # Check for explicit headers
        sections_found: Dict[str, List[str]] = {s: [] for s in self.CANONICAL_SECTIONS}
        current_section = None

        explicit_header_count = 0
        assigned_paragraphs = []

        for p in paragraphs:
            detected_sec = None
            for sec, regexes in self.compiled_headers.items():
                if any(r.search(p) for r in regexes):
                    detected_sec = sec
                    explicit_header_count += 1
                    break

            if detected_sec is not None:
                current_section = detected_sec

            assigned_paragraphs.append((current_section, p))

        # If explicit headers were found for at least 2 sections, use them
        if explicit_header_count >= 2:
            default_sec = "facts"
            for sec, p in assigned_paragraphs:
                target_sec = sec if sec is not None else default_sec
                sections_found[target_sec].append(p)
                default_sec = target_sec

            return {s: "\n".join(sections_found[s]) for s in self.CANONICAL_SECTIONS}

        # Otherwise, use discourse cues and paragraph-level heuristic
        for i, p in enumerate(paragraphs):
            # Calculate match scores for each rhetorical section
            scores = {}
            for sec, cue_regex in CUE_PATTERNS.items():
                matches = len(cue_regex.findall(p))
                scores[sec] = matches

            best_sec, best_score = max(scores.items(), key=lambda item: item[1])

            if best_score > 0:
                sections_found[best_sec].append(p)
            else:
                # Structural fallback based on normalized document position
                pos = i / len(paragraphs)
                if pos < 0.25:
                    sections_found["facts"].append(p)
                elif pos < 0.45:
                    sections_found["procedural_history"].append(p)
                elif pos < 0.70:
                    sections_found["arguments"].append(p)
                else:
                    sections_found["court_reasoning"].append(p)

        return {s: "\n".join(sections_found[s]) for s in self.CANONICAL_SECTIONS}

    def get_section_statistics(self, segmented: Dict[str, str]) -> Dict[str, Any]:
        """Calculates token counts, character lengths, and percentage representation."""
        stats = {}
        total_chars = sum(len(text) for text in segmented.values())
        for sec in self.CANONICAL_SECTIONS:
            sec_text = segmented[sec]
            char_len = len(sec_text)
            word_count = len(sec_text.split())
            pct = (char_len / total_chars * 100.0) if total_chars > 0 else 0.0
            stats[sec] = {
                "char_length": char_len,
                "word_count": word_count,
                "percentage_of_doc": round(pct, 2),
            }
        stats["total_chars"] = total_chars
        return stats

