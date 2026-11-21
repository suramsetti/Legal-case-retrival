"""
Statute Extractor for Indian Legal Judgments.

Extracts and normalizes statutory references from Indian legal text, including:
- Indian Penal Code (IPC)
- Code of Criminal Procedure (CrPC)
- Code of Civil Procedure (CPC)
- Constitution of India (Articles)
- Evidence Act, NI Act, Arbitration Act, NDPS Act, PC Act, IBC, etc.
"""

from dataclasses import dataclass, asdict
import re
from typing import List, Dict, Set, Optional, Tuple


@dataclass(frozen=True)
class StatuteReference:
    act: str
    section: str
    raw: str
    start: int
    end: int
    normalized_key: str

    def to_dict(self) -> Dict[str, any]:
        return asdict(self)


class StatuteExtractor:
    """Extracts and standardizes statutory provisions from Indian court decisions."""

    # Major Acts and Canonical Names
    ACT_CANONICAL_NAMES = {
        "ipc": "Indian Penal Code",
        "crpc": "Code of Criminal Procedure",
        "cpc": "Code of Civil Procedure",
        "const": "Constitution of India",
        "evidence": "Indian Evidence Act",
        "ni": "Negotiable Instruments Act",
        "arbitration": "Arbitration and Conciliation Act",
        "ndps": "Narcotic Drugs and Psychotropic Substances Act",
        "pc": "Prevention of Corruption Act",
        "ibc": "Insolvency and Bankruptcy Code",
        "companies": "Companies Act",
        "mv": "Motor Vehicles Act",
        "it": "Information Technology Act",
        "sarfaesi": "SARFAESI Act",
        "customs": "Customs Act",
        "income_tax": "Income Tax Act",
        "pocso": "POCSO Act",
        "specific_relief": "Specific Relief Act",
        "limitation": "Limitation Act",
        "general": "Act/Statute",
    }

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        # 1. Constitutional Articles: Article(s) 14, 19(1)(a), 21, 226, 32, 136, etc.
        self.pattern_constitution = re.compile(
            r"\b(?:Articles?|Arts?\.)\s*([0-9]+(?:\([0-9a-zA-Z]+\))*(?:\s*(?:to|and|-)\s*[0-9]+(?:\([0-9a-zA-Z]+\))*)*)"
            r"(?:\s+(?:of\s+the\s+Constitution(?:\s+of\s+India)?))?\b",
            re.IGNORECASE,
        )

        # 2. Specific Act Patterns with explicit mentions
        # IPC: Section(s) 302, 304B, 307 read with Section 34 of the Indian Penal Code / IPC
        self.pattern_ipc = re.compile(
            r"\b(?:Sections?|Secs?\.?|u/s|u/ss)\s*([0-9]+[A-Za-z]?(?:\s*(?:read\s+with|\/|and|,|-|\s+to\s+)\s*(?:Sections?|Secs?\.?|u/s)?\s*[0-9]+[A-Za-z]?)*)"
            r"(?:\s*(?:of\s+the\s+)?(?:Indian\s+Penal\s+Code|I\.?P\.?C\.?))\b",
            re.IGNORECASE,
        )

        # CrPC: Section(s) 482, 313, 164, 173(2) of CrPC / Code of Criminal Procedure
        self.pattern_crpc = re.compile(
            r"\b(?:Sections?|Secs?\.?|u/s|u/ss)\s*([0-9]+[A-Za-z]?(?:\([0-9a-zA-Z]+\))*(?:\s*(?:read\s+with|\/|and|,|-|\s+to\s+)\s*(?:Sections?|Secs?\.?|u/s)?\s*[0-9]+[A-Za-z]?(?:\([0-9a-zA-Z]+\))*)*)"
            r"(?:\s*(?:of\s+the\s+)?(?:Code\s+of\s+Criminal\s+Procedure(?:\s*,?\s*1973)?|Cr\.?P\.?C\.?))\b",
            re.IGNORECASE,
        )

        # CPC: Section(s) 9, 100, 115 of CPC / Code of Civil Procedure OR Order X Rule Y CPC
        self.pattern_cpc = re.compile(
            r"\b(?:Sections?|Secs?\.?|u/s|u/ss)\s*([0-9]+[A-Za-z]?(?:\s*(?:read\s+with|\/|and|,|-|\s+to\s+)\s*(?:Sections?|Secs?\.?|u/s)?\s*[0-9]+[A-Za-z]?)*)"
            r"(?:\s*(?:of\s+the\s+)?(?:Code\s+of\s+Civil\s+Procedure(?:\s*,?\s*1908)?|C\.?P\.?C\.?))\b",
            re.IGNORECASE,
        )
        self.pattern_cpc_orders = re.compile(
            r"\b(?:Order|O\.)\s*([0-9IVXLCDM]+)\s*(?:Rules?|R\.)\s*([0-9]+[A-Za-z]?(?:\s*(?:and|-|,)\s*[0-9]+[A-Za-z]?)*)"
            r"(?:\s*(?:of\s+the\s+)?(?:Code\s+of\s+Civil\s+Procedure|C\.?P\.?C\.?))?\b",
            re.IGNORECASE,
        )

        # Indian Evidence Act: Section(s) 27, 32(1), 3 of Evidence Act
        self.pattern_evidence = re.compile(
            r"\b(?:Sections?|Secs?\.?|u/s)\s*([0-9]+[A-Za-z]?(?:\([0-9a-zA-Z]+\))*)"
            r"(?:\s*(?:of\s+the\s+)?(?:Indian\s+Evidence\s+Act(?:\s*,?\s*1872)?|Evidence\s+Act))\b",
            re.IGNORECASE,
        )

        # Negotiable Instruments Act: Section 138 of NI Act
        self.pattern_ni = re.compile(
            r"\b(?:Sections?|Secs?\.?|u/s)\s*([0-9]+[A-Za-z]?)"
            r"(?:\s*(?:of\s+the\s+)?(?:Negotiable\s+Instruments\s+Act(?:\s*,?\s*1881)?|N\.?I\.?\s*Act))\b",
            re.IGNORECASE,
        )

        # Arbitration and Conciliation Act: Section 9, 11(6), 34, 37
        self.pattern_arbitration = re.compile(
            r"\b(?:Sections?|Secs?\.?|u/s)\s*([0-9]+[A-Za-z]?(?:\([0-9a-zA-Z]+\))*)"
            r"(?:\s*(?:of\s+the\s+)?(?:Arbitration\s+(?:and\s+Conciliation\s+)?Act(?:\s*,?\s*1996)?))\b",
            re.IGNORECASE,
        )

        # NDPS Act
        self.pattern_ndps = re.compile(
            r"\b(?:Sections?|Secs?\.?|u/s)\s*([0-9]+[A-Za-z]?(?:\([0-9a-zA-Z]+\))*)"
            r"(?:\s*(?:of\s+the\s+)?(?:Narcotic\s+Drugs\s+(?:and\s+Psychotropic\s+Substances\s+)?Act|N\.?D\.?P\.?S\.?\s*Act))\b",
            re.IGNORECASE,
        )

        # Prevention of Corruption Act (PC Act)
        self.pattern_pc = re.compile(
            r"\b(?:Sections?|Secs?\.?|u/s)\s*([0-9]+[A-Za-z]?(?:\([0-9a-zA-Z]+\))*)"
            r"(?:\s*(?:of\s+the\s+)?(?:Prevention\s+of\s+Corruption\s+Act(?:\s*,?\s*1988)?|P\.?C\.?\s*Act))\b",
            re.IGNORECASE,
        )

        # IBC / Insolvency and Bankruptcy Code
        self.pattern_ibc = re.compile(
            r"\b(?:Sections?|Secs?\.?|u/s)\s*([0-9]+[A-Za-z]?(?:\([0-9a-zA-Z]+\))*)"
            r"(?:\s*(?:of\s+the\s+)?(?:Insolvency\s+and\s+Bankruptcy\s+Code(?:\s*,?\s*2016)?|I\.?B\.?C\.?))\b",
            re.IGNORECASE,
        )

        # Generic Section X of Named Act: "Section 166 of the Motor Vehicles Act", etc.
        self.pattern_generic_act = re.compile(
            r"\b(?:Sections?|Secs?\.?|u/s)\s*([0-9]+[A-Za-z]?(?:\([0-9a-zA-Z]+\))*)\s+of\s+the\s+([A-Z][A-Za-z\s]{3,40}?\s+Act(?:\s*,\s*\d{4})?)\b"
        )

    def _split_sections(self, raw_sections: str) -> List[str]:
        """Splits composite section strings like '302, 307 read with 34' into individual section tokens."""
        clean = re.sub(r"\bread\s+with\b", ",", raw_sections, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:and|to|\/)\b", ",", clean, flags=re.IGNORECASE)
        tokens = [s.strip() for s in clean.split(",") if s.strip()]
        result = []
        for t in tokens:
            m = re.search(r"[0-9]+[A-Za-z]?(?:\([0-9a-zA-Z]+\))*", t)
            if m:
                result.append(m.group(0).upper())
        return result if result else [raw_sections.strip().upper()]

    def extract(self, text: str) -> List[StatuteReference]:
        """
        Extracts all statutory references from text with exact character spans.
        
        Args:
            text: Raw judgment or passage text.
            
        Returns:
            List of StatuteReference objects sorted by starting offset.
        """
        if not text:
            return []

        references: List[StatuteReference] = []

        # 1. Constitution
        for match in self.pattern_constitution.finditer(text):
            raw_sec = match.group(1)
            raw = match.group(0)
            sections = self._split_sections(raw_sec)
            for sec in sections:
                references.append(
                    StatuteReference(
                        act=self.ACT_CANONICAL_NAMES["const"],
                        section=f"Art_{sec}",
                        raw=raw,
                        start=match.start(),
                        end=match.end(),
                        normalized_key=f"CONST:ART_{sec}".upper(),
                    )
                )

        # 2. IPC
        for match in self.pattern_ipc.finditer(text):
            raw_sec = match.group(1)
            raw = match.group(0)
            sections = self._split_sections(raw_sec)
            for sec in sections:
                references.append(
                    StatuteReference(
                        act=self.ACT_CANONICAL_NAMES["ipc"],
                        section=sec,
                        raw=raw,
                        start=match.start(),
                        end=match.end(),
                        normalized_key=f"IPC:{sec}".upper(),
                    )
                )

        # 3. CrPC
        for match in self.pattern_crpc.finditer(text):
            raw_sec = match.group(1)
            raw = match.group(0)
            sections = self._split_sections(raw_sec)
            for sec in sections:
                references.append(
                    StatuteReference(
                        act=self.ACT_CANONICAL_NAMES["crpc"],
                        section=sec,
                        raw=raw,
                        start=match.start(),
                        end=match.end(),
                        normalized_key=f"CRPC:{sec}".upper(),
                    )
                )

        # 4. CPC Sections
        for match in self.pattern_cpc.finditer(text):
            raw_sec = match.group(1)
            raw = match.group(0)
            sections = self._split_sections(raw_sec)
            for sec in sections:
                references.append(
                    StatuteReference(
                        act=self.ACT_CANONICAL_NAMES["cpc"],
                        section=sec,
                        raw=raw,
                        start=match.start(),
                        end=match.end(),
                        normalized_key=f"CPC:{sec}".upper(),
                    )
                )

        # 4b. CPC Orders and Rules
        for match in self.pattern_cpc_orders.finditer(text):
            order = match.group(1).upper()
            rule = match.group(2).upper()
            raw = match.group(0)
            sec_repr = f"O{order}_R{rule}"
            references.append(
                StatuteReference(
                    act=self.ACT_CANONICAL_NAMES["cpc"],
                    section=sec_repr,
                    raw=raw,
                    start=match.start(),
                    end=match.end(),
                    normalized_key=f"CPC:{sec_repr}".upper(),
                )
            )

        # 5. Evidence Act
        for match in self.pattern_evidence.finditer(text):
            sec = match.group(1).upper()
            raw = match.group(0)
            references.append(
                StatuteReference(
                    act=self.ACT_CANONICAL_NAMES["evidence"],
                    section=sec,
                    raw=raw,
                    start=match.start(),
                    end=match.end(),
                    normalized_key=f"EVIDENCE:{sec}".upper(),
                )
            )

        # 6. NI Act
        for match in self.pattern_ni.finditer(text):
            sec = match.group(1).upper()
            raw = match.group(0)
            references.append(
                StatuteReference(
                    act=self.ACT_CANONICAL_NAMES["ni"],
                    section=sec,
                    raw=raw,
                    start=match.start(),
                    end=match.end(),
                    normalized_key=f"NIACT:{sec}".upper(),
                )
            )

        # 7. Arbitration Act
        for match in self.pattern_arbitration.finditer(text):
            sec = match.group(1).upper()
            raw = match.group(0)
            references.append(
                StatuteReference(
                    act=self.ACT_CANONICAL_NAMES["arbitration"],
                    section=sec,
                    raw=raw,
                    start=match.start(),
                    end=match.end(),
                    normalized_key=f"ARBITRATION:{sec}".upper(),
                )
            )

        # 8. NDPS Act
        for match in self.pattern_ndps.finditer(text):
            sec = match.group(1).upper()
            raw = match.group(0)
            references.append(
                StatuteReference(
                    act=self.ACT_CANONICAL_NAMES["ndps"],
                    section=sec,
                    raw=raw,
                    start=match.start(),
                    end=match.end(),
                    normalized_key=f"NDPS:{sec}".upper(),
                )
            )

        # 9. Prevention of Corruption Act
        for match in self.pattern_pc.finditer(text):
            sec = match.group(1).upper()
            raw = match.group(0)
            references.append(
                StatuteReference(
                    act=self.ACT_CANONICAL_NAMES["pc"],
                    section=sec,
                    raw=raw,
                    start=match.start(),
                    end=match.end(),
                    normalized_key=f"PCACT:{sec}".upper(),
                )
            )

        # 10. IBC
        for match in self.pattern_ibc.finditer(text):
            sec = match.group(1).upper()
            raw = match.group(0)
            references.append(
                StatuteReference(
                    act=self.ACT_CANONICAL_NAMES["ibc"],
                    section=sec,
                    raw=raw,
                    start=match.start(),
                    end=match.end(),
                    normalized_key=f"IBC:{sec}".upper(),
                )
            )

        # 11. Generic Named Acts
        for match in self.pattern_generic_act.finditer(text):
            sec = match.group(1).upper()
            act_name = match.group(2).strip()
            raw = match.group(0)
            # Avoid duplicate if matched above
            norm_act = re.sub(r"[^A-Za-z0-9]", "", act_name).upper()
            references.append(
                StatuteReference(
                    act=act_name,
                    section=sec,
                    raw=raw,
                    start=match.start(),
                    end=match.end(),
                    normalized_key=f"{norm_act}:{sec}".upper(),
                )
            )

        # Sort by start offset
        references.sort(key=lambda r: (r.start, r.end))
        return references

    @staticmethod
    def get_normalized_keys(references: List[StatuteReference]) -> Set[str]:
        """Returns the unique set of normalized statute keys from a reference list."""
        return {r.normalized_key for r in references}

    @staticmethod
    def compute_overlap(
        refs_a: List[StatuteReference], refs_b: List[StatuteReference]
    ) -> Dict[str, float]:
        """
        Computes statutory overlap metrics between two documents:
        - jaccard: |A ∩ B| / |A ∪ B|
        - containment_a: |A ∩ B| / |A|
        - containment_b: |A ∩ B| / |B|
        - shared_count: |A ∩ B|
        """
        keys_a = StatuteExtractor.get_normalized_keys(refs_a)
        keys_b = StatuteExtractor.get_normalized_keys(refs_b)

        if not keys_a and not keys_b:
            return {
                "statute_jaccard": 0.0,
                "statute_containment_a": 0.0,
                "statute_containment_b": 0.0,
                "statute_shared_count": 0.0,
            }

        intersection = keys_a.intersection(keys_b)
        union = keys_a.union(keys_b)

        return {
            "statute_jaccard": len(intersection) / max(len(union), 1),
            "statute_containment_a": len(intersection) / max(len(keys_a), 1),
            "statute_containment_b": len(intersection) / max(len(keys_b), 1),
            "statute_shared_count": float(len(intersection)),
        }

