"""
Legal Citation Extractor for Indian Legal Judgments.

Extracts and standardizes formal legal citations, including:
- All India Reporter (AIR): e.g. "AIR 1973 SC 1461", "1994 AIR 268"
- Supreme Court Cases (SCC): e.g. "(1973) 4 SCC 225", "2002 (3) SCC 586"
- Supreme Court Reports (SCR): e.g. "[1973] 3 SCR 330", "(1951) SCR 525"
- Case Appeal Numbers: e.g. "Criminal Appeal No. 92/2015", "Civil Appeal No. 1234 of 2010"
- Case Title References: e.g. "Kesavananda Bharati v. State of Kerala"
"""

from dataclasses import dataclass, asdict
import re
from typing import List, Dict, Set, Optional, Tuple


@dataclass(frozen=True)
class LegalCitation:
    reporter: str
    year: Optional[int]
    volume: Optional[str]
    page: Optional[str]
    raw: str
    start: int
    end: int
    normalized_key: str

    def to_dict(self) -> Dict[str, any]:
        return asdict(self)


class CitationExtractor:
    """Extracts, parses, and normalizes precedent citations from Indian judgments."""

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        # 1. SCC Citations:
        # e.g., (1973) 4 SCC 225, [2002] 3 SCC 586, 2018 (12) SCC 1, 1995 Supp (2) SCC 120
        self.pattern_scc = re.compile(
            r"[\(\[]?\s*(\d{4})\s*[\)\]]?\s*(?:\(?\s*(\d+|Supp(?:\s*\d+)?)\s*\)?)?\s*SCC\s*(?:\([A-Za-z&\s]+\))?\s*(\d+)",
            re.IGNORECASE,
        )

        # 2. AIR Citations:
        # e.g., AIR 1973 SC 1461, AIR 1980 SC 1160, 1994 AIR SC 268, AIR 2005 SC 2345
        self.pattern_air = re.compile(
            r"\b(?:AIR\s*(\d{4})\s*([A-Za-z]+)?\s*(\d+)|(\d{4})\s+AIR\s*([A-Za-z]+)?\s*(\d+))\b",
            re.IGNORECASE,
        )

        # 3. SCR Citations:
        # e.g., [1973] 3 SCR 330, (1951) SCR 525, 1962 (2) SCR 100
        self.pattern_scr = re.compile(
            r"[\(\[]?\s*(\d{4})\s*[\)\]]?\s*(?:\(?\s*(\d+|Supp(?:\s*\d+)?)\s*\)?)?\s*SCR\s*(\d+)",
            re.IGNORECASE,
        )

        # 4. Formal Appeal Numbers:
        # e.g. Criminal Appeal No. 92 of 2015, Civil Appeal No. 1234/2010, SLP (Crl.) No. 5678 of 2019
        self.pattern_appeals = re.compile(
            r"\b(Criminal\s+Appeal|Civil\s+Appeal|SLP\s*\([A-Za-z\.]+\)|Special\s+Leave\s+Petition\s*\([A-Za-z\.]+\)|Writ\s+Petition\s*\([A-Za-z\.]+\)|W\.P\.\s*\([A-Za-z\.]+\))\s*(?:Nos?\.?|Numbers?)\s*([0-9]+(?:\s*[-–]\s*[0-9]+)?)\s*(?:of|\/)\s*(\d{4})\b",
            re.IGNORECASE,
        )

        # 5. Precedent Case Title (X v. Y):
        # e.g. State of Haryana v. Bhajan Lal, Kesavananda Bharati v. State of Kerala
        self.pattern_case_title = re.compile(
            r"\b([A-Z][A-Za-z0-9\s\.\(\)\&\'\–\-]{2,50}?)\s+(?:v\.|vs\.?|versus)\s+([A-Z][A-Za-z0-9\s\.\(\)\&\'\–\-]{2,50}?)\b"
        )

    def extract(self, text: str) -> List[LegalCitation]:
        """
        Extracts all citation instances from the judgment text.
        
        Args:
            text: Raw judgment string.
            
        Returns:
            List of LegalCitation objects sorted by start offset.
        """
        if not text:
            return []

        citations: List[LegalCitation] = []

        # 1. SCC
        for match in self.pattern_scc.finditer(text):
            year = int(match.group(1))
            vol = match.group(2) if match.group(2) else "1"
            page = match.group(3)
            vol_clean = re.sub(r"\s+", "", vol).upper()
            norm_key = f"SCC:{year}:{vol_clean}:{page}"
            citations.append(
                LegalCitation(
                    reporter="SCC",
                    year=year,
                    volume=vol_clean,
                    page=page,
                    raw=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    normalized_key=norm_key,
                )
            )

        # 2. AIR
        for match in self.pattern_air.finditer(text):
            if match.group(1):
                year = int(match.group(1))
                court = match.group(2) if match.group(2) else "SC"
                page = match.group(3)
            else:
                year = int(match.group(4))
                court = match.group(5) if match.group(5) else "SC"
                page = match.group(6)
            court_clean = court.strip().upper()
            norm_key = f"AIR:{year}:{court_clean}:{page}"
            citations.append(
                LegalCitation(
                    reporter="AIR",
                    year=year,
                    volume=court_clean,
                    page=page,
                    raw=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    normalized_key=norm_key,
                )
            )

        # 3. SCR
        for match in self.pattern_scr.finditer(text):
            year = int(match.group(1))
            vol = match.group(2) if match.group(2) else "1"
            page = match.group(3)
            vol_clean = re.sub(r"\s+", "", vol).upper()
            norm_key = f"SCR:{year}:{vol_clean}:{page}"
            citations.append(
                LegalCitation(
                    reporter="SCR",
                    year=year,
                    volume=vol_clean,
                    page=page,
                    raw=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    normalized_key=norm_key,
                )
            )

        # 4. Formal Appeal Numbers
        for match in self.pattern_appeals.finditer(text):
            appeal_type = re.sub(r"[\s\.\(\)]+", "_", match.group(1).strip().upper()).strip("_")
            num = match.group(2).strip()
            year = int(match.group(3))
            norm_key = f"APPEAL:{appeal_type}:{num}:{year}"
            citations.append(
                LegalCitation(
                    reporter=appeal_type,
                    year=year,
                    volume=num,
                    page="",
                    raw=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    normalized_key=norm_key,
                )
            )

        # Sort by start offset
        citations.sort(key=lambda c: (c.start, c.end))
        return citations

    def extract_case_titles(self, text: str) -> List[Dict[str, any]]:
        """
        Extracts judicial case title mentions (e.g. 'State of Haryana v. Bhajan Lal').
        Filters common non-case false positives (e.g. 'Section 302 v. Section 304').
        """
        titles = []
        stop_names = {"SECTION", "SEC", "ORDER", "ARTICLE", "ACT", "RULE", "CLAUSE", "PARA", "NO"}

        for match in self.pattern_case_title.finditer(text):
            party1 = match.group(1).strip()
            party2 = match.group(2).strip()

            # Filter out non-names
            first_word_p1 = party1.split()[0].upper()
            first_word_p2 = party2.split()[0].upper()
            if first_word_p1 in stop_names or first_word_p2 in stop_names:
                continue

            # Require minimum length and capital letter patterns
            if len(party1) < 3 or len(party2) < 3:
                continue

            clean_p1 = re.sub(r"[^A-Za-z0-9]", "", party1).upper()
            clean_p2 = re.sub(r"[^A-Za-z0-9]", "", party2).upper()
            norm_key = f"TITLE:{clean_p1}_VS_{clean_p2}"

            titles.append({
                "party1": party1,
                "party2": party2,
                "raw": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "normalized_key": norm_key,
            })
        return titles

    @staticmethod
    def get_normalized_keys(citations: List[LegalCitation]) -> Set[str]:
        """Returns the unique set of normalized citation keys."""
        return {c.normalized_key for c in citations}

    @staticmethod
    def compute_overlap(
        cits_a: List[LegalCitation], cits_b: List[LegalCitation]
    ) -> Dict[str, float]:
        """
        Computes citation overlap metrics between two documents:
        - citation_jaccard: |A ∩ B| / |A ∪ B|
        - citation_shared_count: |A ∩ B|
        - citation_containment_a: |A ∩ B| / |A|
        - citation_containment_b: |A ∩ B| / |B|
        """
        keys_a = CitationExtractor.get_normalized_keys(cits_a)
        keys_b = CitationExtractor.get_normalized_keys(cits_b)

        if not keys_a and not keys_b:
            return {
                "citation_jaccard": 0.0,
                "citation_shared_count": 0.0,
                "citation_containment_a": 0.0,
                "citation_containment_b": 0.0,
            }

        intersection = keys_a.intersection(keys_b)
        union = keys_a.union(keys_b)

        return {
            "citation_jaccard": len(intersection) / max(len(union), 1),
            "citation_shared_count": float(len(intersection)),
            "citation_containment_a": len(intersection) / max(len(keys_a), 1),
            "citation_containment_b": len(intersection) / max(len(keys_b), 1),
        }

