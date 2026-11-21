"""
Legal Named Entity Extractor for Indian Legal Judgments.

Extracts, normalizes, and classifies Indian legal entities including:
- Court names (Supreme Court, High Courts, Sessions, Tribunals)
- Judge names (Justices, Chief Justices, Coram)
- Parties (Appellants, Petitioners, Respondents)
- Advocates / Counsels (Senior Counsels, Advocates General)
- Organizations (State Governments, Union of India, PSUs, Statutory Boards)
- Locations (States, High Court Seats, Indian Cities)
- Legal Authorities (Magistrates, Police Officers, Tribunals)
"""

from dataclasses import dataclass, asdict
import re
from typing import List, Dict, Set, Optional, Tuple


@dataclass(frozen=True)
class LegalEntity:
    category: str  # COURT, JUDGE, PETITIONER, RESPONDENT, COUNSEL, ORG, LOC, AUTHORITY
    name: str
    raw: str
    start: int
    end: int
    normalized_key: str

    def to_dict(self) -> Dict[str, any]:
        return asdict(self)


class LegalEntityExtractor:
    """Rule-based and pattern-driven legal named entity recognizer tailored for Indian court decisions."""

    INDIAN_STATES = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
        "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
        "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
        "Odisha", "Orissa", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
        "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Jammu and Kashmir",
        "Puducherry", "Chandigarh", "Ladakh"
    ]

    MAJOR_CITIES = [
        "New Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru", "Bangalore", "Hyderabad",
        "Ahmedabad", "Pune", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane",
        "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Ghaziabad", "Ludhiana", "Agra",
        "Nashik", "Faridabad", "Meerut", "Rajkot", "Varanasi", "Srinagar", "Aurangabad",
        "Dhanbad", "Amritsar", "Allahabad", "Prayagraj", "Ranchi", "Howrah", "Coimbatore",
        "Jabalpur", "Gwalior", "Vijayawada", "Jodhpur", "Madurai", "Raipur", "Kota",
        "Guwahati", "Chandigarh", "Shimla", "Dehradun", "Cuttack", "Kochi", "Ernakulam"
    ]

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        # 1. Courts
        self.pattern_sc = re.compile(
            r"\b(?:Supreme\s+Court\s+of\s+India|Apex\s+Court|this\s+Court)\b", re.IGNORECASE
        )
        self.pattern_hc = re.compile(
            r"\b(?:High\s+Court\s+of\s+Judicature\s+at\s+[A-Za-z]+|High\s+Court\s+of\s+[A-Za-z\s]+|High\s+Court)\b",
            re.IGNORECASE,
        )
        self.pattern_tribunals = re.compile(
            r"\b(?:National\s+Company\s+Law\s+(?:Appellate\s+)?Tribunal|NCLAT|NCLT|Central\s+Administrative\s+Tribunal|CAT|Income\s+Tax\s+Appellate\s+Tribunal|ITAT|Debt\s+Recovery\s+Tribunal|DRT|National\s+Green\s+Tribunal|NGT|Consumer\s+Disputes\s+Redressal\s+Commission)\b",
            re.IGNORECASE,
        )
        self.pattern_subordinate_courts = re.compile(
            r"\b(?:Sessions\s+Court|District\s+Court|Court\s+of\s+Session|Chief\s+Judicial\s+Magistrate|CJM|Special\s+Judge(?:\s*\([A-Za-z\s]+\))?)\b",
            re.IGNORECASE,
        )

        # 2. Judges
        self.pattern_judges = re.compile(
            r"\b(?:Hon'?ble\s+)?(?:Mr\.\s+|Mrs\.\s+|Ms\.\s+)?(?:Chief\s+Justice|Justice|J\.)\s+([A-Z][a-zA-Z\.\s\-]{2,30}?)(?:\s*,\s*(?:CJI|J\.?|JJ\.?))?\b"
        )
        self.pattern_judge_suffix = re.compile(
            r"\b([A-Z][a-zA-Z\.\s\-]{2,30}?)\s*,\s*(?:CJI|J\.?|JJ\.?)\b"
        )
        # Signature lines: ...J. (V. Gopala Gowda) or (R. Banumathi), J.
        self.pattern_judge_signature = re.compile(
            r"(?:(?:J\.|JJ\.)\s*[\(\[]\s*([A-Z][a-zA-Z\.\s\-]{2,30}?)\s*[\)\]]|[\(\[]\s*([A-Z][a-zA-Z\.\s\-]{2,30}?)\s*[\)\]]\s*,?\s*(?:CJI|J\.?|JJ\.?))"
        )

        # 3. Advocates / Counsels
        self.pattern_counsel = re.compile(
            r"\b(?:Mr\.|Ms\.|Mrs\.|Dr\.)\s+([A-Z][a-zA-Z\.\s\-]{2,30}?)\s*,?\s*(?:learned\s+)?(?:Senior\s+Advocate|Senior\s+Counsel|Advocate|Counsel|Solicitor\s+General|Attorney\s+General|A\.?S\.?G\.?|Addl\.\s+Solicitor\s+General)\b"
        )
        self.pattern_counsel_for = re.compile(
            r"\b(?:learned\s+counsel\s+(?:appearing\s+)?for\s+(?:the\s+)?(?:appellant|petitioner|respondent|State|accused))\s*,?\s*(?:Mr\.|Ms\.|Mrs\.|Dr\.)?\s*([A-Z][a-zA-Z\.\s\-]{2,30}?)\b",
            re.IGNORECASE,
        )

        # 4. Organizations (State of X, Union of India, PSUs, Regulatory Agencies)
        states_pattern = "|".join([re.escape(s) for s in self.INDIAN_STATES])
        self.pattern_state_org = re.compile(
            rf"\bState\s+of\s+({states_pattern})\b", re.IGNORECASE
        )
        self.pattern_union = re.compile(
            r"\b(?:Union\s+of\s+India|Govt\.\s+of\s+India|Government\s+of\s+India)\b", re.IGNORECASE
        )
        self.pattern_agencies = re.compile(
            r"\b(?:Reserve\s+Bank\s+of\s+India|RBI|Securities\s+and\s+Exchange\s+Board\s+of\s+India|SEBI|Central\s+Bureau\s+of\s+Investigation|CBI|Enforcement\s+Directorate|ED|Bar\s+Council\s+of\s+India|State\s+Bank\s+of\s+India|SBI|Life\s+Insurance\s+Corporation|LIC)\b"
        )

        # 5. Locations (States and Major Cities)
        cities_pattern = "|".join([re.escape(c) for c in self.MAJOR_CITIES])
        self.pattern_locations = re.compile(
            rf"\b({states_pattern}|{cities_pattern})\b"
        )

        # 6. Legal Authorities
        self.pattern_authorities = re.compile(
            r"\b(?:Station\s+House\s+Officer|S\.?H\.?O\.?|Investigating\s+Officer|I\.?O\.?|Sub[- ]Divisional\s+Magistrate|S\.?D\.?M\.?|District\s+Magistrate|D\.?M\.?|Tahsildar|Tehsildar|Disciplinary\s+Authority|Competent\s+Authority|Official\s+Liquidator|Commissioner\s+of\s+Police)\b",
            re.IGNORECASE,
        )

        # 7. Parties (Appellant / Respondent from headers or text)
        self.pattern_appellant = re.compile(
            r"\b(?:Appellant|Petitioner)(?:s)?\s*[:\-]?\s*([A-Z][a-zA-Z0-9\s\.\(\)\&\'\–\-]{2,50}?)(?:\n|\r|\s+versus|\s+v\.|\s+vs\.?)",
            re.IGNORECASE,
        )
        self.pattern_respondent = re.compile(
            r"\b(?:Respondent|Defendant)(?:s)?\s*[:\-]?\s*([A-Z][a-zA-Z0-9\s\.\(\)\&\'\–\-]{2,50}?)(?:\n|\r|\s+Dated|\s+CORAM|\s+BEFORE)",
            re.IGNORECASE,
        )

    def extract(self, text: str) -> List[LegalEntity]:
        """
        Extracts legal entities from text with exact character spans and normalized forms.
        """
        if not text:
            return []

        entities: List[LegalEntity] = []

        # 1. Courts
        for m in self.pattern_sc.finditer(text):
            entities.append(LegalEntity(
                category="COURT",
                name="Supreme Court of India",
                raw=m.group(0),
                start=m.start(),
                end=m.end(),
                normalized_key="COURT:SUPREME_COURT_OF_INDIA",
            ))

        for m in self.pattern_hc.finditer(text):
            raw = m.group(0)
            norm = re.sub(r"[^A-Za-z0-9]", "_", raw).upper()
            entities.append(LegalEntity(
                category="COURT",
                name=raw.strip(),
                raw=raw,
                start=m.start(),
                end=m.end(),
                normalized_key=f"COURT:{norm}",
            ))

        for m in self.pattern_tribunals.finditer(text):
            raw = m.group(0)
            norm = re.sub(r"[^A-Za-z0-9]", "_", raw).upper()
            entities.append(LegalEntity(
                category="COURT",
                name=raw.strip(),
                raw=raw,
                start=m.start(),
                end=m.end(),
                normalized_key=f"COURT:TRIBUNAL_{norm}",
            ))

        for m in self.pattern_subordinate_courts.finditer(text):
            raw = m.group(0)
            norm = re.sub(r"[^A-Za-z0-9]", "_", raw).upper()
            entities.append(LegalEntity(
                category="COURT",
                name=raw.strip(),
                raw=raw,
                start=m.start(),
                end=m.end(),
                normalized_key=f"COURT:SUB_{norm}",
            ))

        # 2. Judges
        for m in self.pattern_judges.finditer(text):
            jname = m.group(1).strip()
            if len(jname) > 2 and not any(w in jname.lower() for w in ["court", "order", "section"]):
                clean_name = re.sub(r"[^A-Za-z]", " ", jname).strip()
                entities.append(LegalEntity(
                    category="JUDGE",
                    name=clean_name,
                    raw=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    normalized_key=f"JUDGE:{re.sub(r'[^A-Z]', '', clean_name.upper())}",
                ))

        for m in self.pattern_judge_suffix.finditer(text):
            jname = m.group(1).strip()
            if len(jname) > 2 and not any(w in jname.lower() for w in ["court", "order", "section"]):
                clean_name = re.sub(r"[^A-Za-z]", " ", jname).strip()
                entities.append(LegalEntity(
                    category="JUDGE",
                    name=clean_name,
                    raw=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    normalized_key=f"JUDGE:{re.sub(r'[^A-Z]', '', clean_name.upper())}",
                ))

        for m in self.pattern_judge_signature.finditer(text):
            jname = (m.group(1) or m.group(2) or "").strip()
            if len(jname) > 2 and not any(w in jname.lower() for w in ["court", "order", "section"]):
                clean_name = re.sub(r"[^A-Za-z]", " ", jname).strip()
                entities.append(LegalEntity(
                    category="JUDGE",
                    name=clean_name,
                    raw=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    normalized_key=f"JUDGE:{re.sub(r'[^A-Z]', '', clean_name.upper())}",
                ))

        # 3. Counsels / Advocates
        for m in self.pattern_counsel.finditer(text):
            cname = m.group(1).strip()
            if len(cname) > 2:
                entities.append(LegalEntity(
                    category="COUNSEL",
                    name=cname,
                    raw=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    normalized_key=f"COUNSEL:{re.sub(r'[^A-Za-z0-9]', '', cname).upper()}",
                ))

        for m in self.pattern_counsel_for.finditer(text):
            cname = m.group(1).strip()
            if len(cname) > 2:
                entities.append(LegalEntity(
                    category="COUNSEL",
                    name=cname,
                    raw=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    normalized_key=f"COUNSEL:{re.sub(r'[^A-Za-z0-9]', '', cname).upper()}",
                ))

        # 4. Organizations
        for m in self.pattern_state_org.finditer(text):
            state = m.group(1).strip()
            entities.append(LegalEntity(
                category="ORG",
                name=f"State of {state}",
                raw=m.group(0),
                start=m.start(),
                end=m.end(),
                normalized_key=f"ORG:STATE_OF_{re.sub(r'[^A-Za-z]', '', state).upper()}",
            ))

        for m in self.pattern_union.finditer(text):
            entities.append(LegalEntity(
                category="ORG",
                name="Union of India",
                raw=m.group(0),
                start=m.start(),
                end=m.end(),
                normalized_key="ORG:UNION_OF_INDIA",
            ))

        for m in self.pattern_agencies.finditer(text):
            agency = m.group(0).strip()
            entities.append(LegalEntity(
                category="ORG",
                name=agency,
                raw=agency,
                start=m.start(),
                end=m.end(),
                normalized_key=f"ORG:{re.sub(r'[^A-Za-z0-9]', '_', agency).upper()}",
            ))

        # 5. Locations
        for m in self.pattern_locations.finditer(text):
            loc = m.group(1).strip()
            entities.append(LegalEntity(
                category="LOC",
                name=loc,
                raw=loc,
                start=m.start(),
                end=m.end(),
                normalized_key=f"LOC:{re.sub(r'[^A-Za-z0-9]', '_', loc).upper()}",
            ))

        # 6. Legal Authorities
        for m in self.pattern_authorities.finditer(text):
            auth = m.group(0).strip()
            entities.append(LegalEntity(
                category="AUTHORITY",
                name=auth,
                raw=auth,
                start=m.start(),
                end=m.end(),
                normalized_key=f"AUTH:{re.sub(r'[^A-Za-z0-9]', '_', auth).upper()}",
            ))

        # 7. Parties (Appellant / Respondent)
        for m in self.pattern_appellant.finditer(text[:1500]):
            pname = m.group(1).strip()
            if 3 < len(pname) < 40 and not any(w in pname.lower() for w in ["learned", "court", "order"]):
                entities.append(LegalEntity(
                    category="PETITIONER",
                    name=pname,
                    raw=pname,
                    start=m.start(),
                    end=m.end(),
                    normalized_key=f"PARTY:PET_{re.sub(r'[^A-Za-z0-9]', '', pname).upper()}",
                ))

        for m in self.pattern_respondent.finditer(text[:2000]):
            rname = m.group(1).strip()
            if 3 < len(rname) < 40 and not any(w in rname.lower() for w in ["learned", "court", "order"]):
                entities.append(LegalEntity(
                    category="RESPONDENT",
                    name=rname,
                    raw=rname,
                    start=m.start(),
                    end=m.end(),
                    normalized_key=f"PARTY:RESP_{re.sub(r'[^A-Za-z0-9]', '', rname).upper()}",
                ))

        entities.sort(key=lambda e: (e.start, e.end))
        return entities

    @staticmethod
    def get_category_breakdown(entities: List[LegalEntity]) -> Dict[str, int]:
        """Returns the entity count breakdown per category."""
        breakdown: Dict[str, int] = {}
        for e in entities:
            breakdown[e.category] = breakdown.get(e.category, 0) + 1
        return breakdown

    @staticmethod
    def compute_overlap(
        ents_a: List[LegalEntity], ents_b: List[LegalEntity]
    ) -> Dict[str, float]:
        """
        Computes entity overlap metrics between two documents:
        - entity_jaccard (all entities)
        - judge_overlap (shared judges)
        - court_overlap (shared courts)
        - org_overlap (shared organizations)
        - loc_overlap (shared geographic jurisdictions)
        """
        keys_a = {e.normalized_key for e in ents_a}
        keys_b = {e.normalized_key for e in ents_b}

        if not keys_a and not keys_b:
            jaccard = 0.0
        else:
            jaccard = len(keys_a.intersection(keys_b)) / max(len(keys_a.union(keys_b)), 1)

        def sub_overlap(cat: str) -> float:
            sub_a = {e.normalized_key for e in ents_a if e.category == cat}
            sub_b = {e.normalized_key for e in ents_b if e.category == cat}
            if not sub_a or not sub_b:
                return 0.0
            return len(sub_a.intersection(sub_b)) / max(len(sub_a.union(sub_b)), 1)

        return {
            "entity_jaccard": jaccard,
            "entity_shared_count": float(len(keys_a.intersection(keys_b))),
            "judge_overlap": sub_overlap("JUDGE"),
            "court_overlap": sub_overlap("COURT"),
            "org_overlap": sub_overlap("ORG"),
            "loc_overlap": sub_overlap("LOC"),
        }
