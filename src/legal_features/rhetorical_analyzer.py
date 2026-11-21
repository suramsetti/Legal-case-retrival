"""
Rhetorical Role Analyzer for Indian Supreme Court Judgments.

Analyzes sentence-level legal discourse and computes document-level rhetorical role distributions
aligned with the official IL-TUR RR benchmark categories:
- Fact (FAC)
- Issue (ISSUE)
- ArgumentPetitioner (ARG_PET)
- ArgumentRespondent (ARG_RESP)
- Statute (STA)
- Precedent (PREC)
- RulingByLowerCourt (RLC)
- RatioOfTheDecision (RATIO)
- RulingByPresentCourt (RPC)
- Dissent (DIS)
"""

from dataclasses import dataclass, asdict
import re
from typing import List, Dict, Set, Optional, Tuple
import numpy as np
from scipy.spatial.distance import jensenshannon


# Canonical Role definitions matching IL-TUR RR
RHETORICAL_ROLES = [
    "Fact",                  # 0
    "Issue",                 # 1
    "ArgumentPetitioner",    # 2
    "ArgumentRespondent",    # 3
    "Statute",               # 4
    "Dissent",               # 5
    "PrecedentReliedUpon",   # 6
    "RulingByLowerCourt",    # 9
    "RatioOfTheDecision",    # 10
    "RulingByPresentCourt",  # 11
    "Other",                 # 12
]

ROLE_TO_IDX = {role: idx for idx, role in enumerate(RHETORICAL_ROLES)}
NUM_ROLES = len(RHETORICAL_ROLES)


@dataclass
class SentenceRole:
    sentence_idx: int
    text: str
    predicted_role: str
    confidence: float

    def to_dict(self) -> Dict[str, any]:
        return asdict(self)


class RhetoricalAnalyzer:
    """Discourse analyzer for Indian judicial opinions classifying rhetorical stages."""

    def __init__(self):
        self._compile_lexicons()

    def _compile_lexicons(self):
        # 1. Facts
        self.p_fact = re.compile(
            r"\b(?:case\s+of\s+the\s+prosecution|briefly\s+stated|facts\s+of\s+the\s+case|prosecution\s+story|"
            r"according\s+to\s+the\s+FIR|the\s+deceased|the\s+incident\s+took\s+place|the\s+appellant\s+was\s+arrested|"
            r"on\s+\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+\d{4}|complainant\s+filed|the\s+suit\s+was\s+filed)\b",
            re.IGNORECASE,
        )

        # 2. Issue
        self.p_issue = re.compile(
            r"\b(?:the\s+short\s+question|the\s+question\s+(?:that\s+)?arises|point\s+for\s+determination|"
            r"whether\s+the\s+High\s+Court\s+was\s+justified|the\s+issue\s+before\s+us|the\s+sole\s+question|"
            r"whether\s+or\s+not|the\s+moot\s+question|the\s+core\s+issue)\b",
            re.IGNORECASE,
        )

        # 3. Argument Petitioner
        self.p_arg_pet = re.compile(
            r"\b(?:learned\s+(?:senior\s+)?counsel\s+(?:appearing\s+)?(?:for|on\s+behalf\s+of)\s+the\s+appellant|"
            r"learned\s+counsel\s+for\s+the\s+petitioner|appellant('s)?\s+contention|submitted\s+by\s+the\s+appellant|"
            r"contended\s+by\s+the\s+learned\s+counsel\s+for\s+the\s+appellant|appellant\s+argued\s+that)\b",
            re.IGNORECASE,
        )

        # 4. Argument Respondent
        self.p_arg_resp = re.compile(
            r"\b(?:learned\s+(?:senior\s+)?counsel\s+(?:appearing\s+)?(?:for|on\s+behalf\s+of)\s+the\s+respondent|"
            r"learned\s+(?:Additional\s+)?Solicitor\s+General|counsel\s+for\s+the\s+State|respondent('s)?\s+submission|"
            r"on\s+the\s+other\s+hand\s*,?\s*it\s+was\s+submitted|contended\s+on\s+behalf\s+of\s+the\s+respondent)\b",
            re.IGNORECASE,
        )

        # 5. Statute
        self.p_statute = re.compile(
            r"\b(?:Section\s+\d+|Article\s+\d+|Order\s+[0-9IVXLCDM]+\s+Rule|provisions?\s+of\s+(?:the\s+)?Act|"
            r"sub-section\s+\(\d+\)|a\s+plain\s+reading\s+of\s+Section|the\s+statutory\s+scheme|"
            r"as\s+defined\s+under\s+Section|stipulates\s+that|reads\s+as\s+under)\b",
            re.IGNORECASE,
        )

        # 6. Precedent
        self.p_precedent = re.compile(
            r"\b(?:in\s+the\s+case\s+of|reliance\s+was\s+placed\s+on|this\s+Court\s+in|Constitution\s+Bench\s+in|"
            r"SCC|AIR\s+\d{4}|SCR|as\s+held\s+in|referred\s+to\s+the\s+judgment\s+in|decision\s+of\s+this\s+Court\s+in|"
            r"v\.\s+State|versus\s+Union\s+of\s+India)\b",
            re.IGNORECASE,
        )

        # 7. Ruling by Lower Court
        self.p_rlc = re.compile(
            r"\b(?:the\s+Trial\s+Court\s+(?:convicted|acquitted|held|dismissed)|the\s+High\s+Court\s+by\s+(?:the\s+)?impugned\s+judgment|"
            r"impugned\s+order\s+dated|aggrieved\s+by\s+the\s+(?:said\s+)?judgment|Sessions\s+Judge\s+convicted|"
            r"the\s+learned\s+Single\s+Judge\s+held|the\s+Division\s+Bench\s+confirmed)\b",
            re.IGNORECASE,
        )

        # 8. Ratio of the Decision
        self.p_ratio = re.compile(
            r"\b(?:in\s+our\s+considered\s+opinion|we\s+are\s+of\s+the\s+(?:firm\s+)?view|it\s+is\s+well\s+settled\s+that|"
            r"we\s+hold\s+that|the\s+settled\s+legal\s+position|having\s+carefully\s+considered|we\s+find\s+that|"
            r"the\s+inevitable\s+conclusion|on\s+a\s+conjoint\s+reading|we\s+have\s+no\s+hesitation\s+in\s+holding|"
            r"the\s+principles?\s+that\s+emerge|the\s+law\s+on\s+this\s+issue)\b",
            re.IGNORECASE,
        )

        # 9. Ruling by Present Court
        self.p_rpc = re.compile(
            r"\b(?:in\s+the\s+result\s*,?\s*the\s+appeal|the\s+appeal\s+is\s+(?:hereby\s+)?(?:allowed|dismissed|disposed\s+of)|"
            r"the\s+impugned\s+judgment\s+(?:is\s+set\s+aside|is\s+upheld)|conviction\s+is\s+set\s+aside|"
            r"appellant\s+is\s+acquitted|bail\s+is\s+granted|no\s+order\s+as\s+to\s+costs|"
            r"appeals\s+stand\s+disposed\s+of|stands\s+quashed)\b",
            re.IGNORECASE,
        )

    def classify_sentence(self, text: str, position_ratio: float = 0.5) -> Tuple[str, float]:
        """
        Classifies an individual sentence into a rhetorical category using discourse cues and document positional priors.
        
        Args:
            text: Sentence text.
            position_ratio: Normalized position of the sentence in the document [0.0, 1.0].
            
        Returns:
            Tuple of (predicted_role, confidence_score)
        """
        scores: Dict[str, float] = {r: 0.05 for r in RHETORICAL_ROLES}

        # Positional priors based on legal discourse structure:
        # Early: Facts, Lower Court rulings
        # Middle: Arguments, Statutes, Precedents
        # Late: Ratio, Ruling by Present Court
        if position_ratio < 0.25:
            scores["Fact"] += 0.35
            scores["RulingByLowerCourt"] += 0.25
        elif position_ratio < 0.60:
            scores["ArgumentPetitioner"] += 0.20
            scores["ArgumentRespondent"] += 0.20
            scores["Statute"] += 0.15
            scores["PrecedentReliedUpon"] += 0.20
        elif position_ratio < 0.85:
            scores["RatioOfTheDecision"] += 0.35
            scores["PrecedentReliedUpon"] += 0.15
        else:
            scores["RulingByPresentCourt"] += 0.45
            scores["RatioOfTheDecision"] += 0.20

        # Pattern scoring
        if self.p_rpc.search(text):
            scores["RulingByPresentCourt"] += 0.85
        if self.p_ratio.search(text):
            scores["RatioOfTheDecision"] += 0.75
        if self.p_issue.search(text):
            scores["Issue"] += 0.80
        if self.p_arg_pet.search(text):
            scores["ArgumentPetitioner"] += 0.70
        if self.p_arg_resp.search(text):
            scores["ArgumentRespondent"] += 0.70
        if self.p_statute.search(text):
            scores["Statute"] += 0.65
        if self.p_precedent.search(text):
            scores["PrecedentReliedUpon"] += 0.65
        if self.p_rlc.search(text):
            scores["RulingByLowerCourt"] += 0.65
        if self.p_fact.search(text):
            scores["Fact"] += 0.55

        best_role = max(scores, key=scores.get)
        confidence = scores[best_role]
        return best_role, float(min(1.0, confidence))

    def analyze_document(self, sentences: List[str]) -> Tuple[List[SentenceRole], np.ndarray]:
        """
        Analyzes all sentences in a document and returns individual sentence classifications
        and the aggregate normalized document distribution vector across roles.
        
        Args:
            sentences: Ordered list of sentence strings.
            
        Returns:
            Tuple of (List[SentenceRole], distribution_vector of length NUM_ROLES).
        """
        n_sentences = len(sentences)
        if n_sentences == 0:
            return [], np.zeros(NUM_ROLES, dtype=np.float32)

        sentence_roles: List[SentenceRole] = []
        counts = np.zeros(NUM_ROLES, dtype=np.float32)

        for idx, sent in enumerate(sentences):
            pos_ratio = idx / max(1, n_sentences - 1)
            role, conf = self.classify_sentence(sent, position_ratio=pos_ratio)
            sentence_roles.append(SentenceRole(
                sentence_idx=idx,
                text=sent,
                predicted_role=role,
                confidence=conf,
            ))
            counts[ROLE_TO_IDX[role]] += 1.0

        # Normalize to probability distribution
        distribution = counts / max(counts.sum(), 1.0)
        return sentence_roles, distribution

    @staticmethod
    def compute_distribution_similarity(dist_a: np.ndarray, dist_b: np.ndarray) -> Dict[str, float]:
        """
        Computes distribution overlap metrics between two documents:
        - rhetorical_cosine: Cosine similarity of role distribution vectors
        - rhetorical_js_sim: 1 - Jensen-Shannon distance [0, 1]
        - ratio_overlap: Overlap in decision ratio proportion
        - fact_overlap: Overlap in factual background proportion
        """
        norm_a = np.linalg.norm(dist_a)
        norm_b = np.linalg.norm(dist_b)

        if norm_a == 0 or norm_b == 0:
            cosine = 0.0
            js_sim = 0.0
        else:
            cosine = float(np.dot(dist_a, dist_b) / (norm_a * norm_b))
            # JS distance is in [0, 1] for log base 2
            js_dist = jensenshannon(dist_a, dist_b, base=2)
            js_sim = float(max(0.0, 1.0 - js_dist))

        ratio_idx = ROLE_TO_IDX["RatioOfTheDecision"]
        fact_idx = ROLE_TO_IDX["Fact"]

        return {
            "rhetorical_cosine": cosine,
            "rhetorical_js_sim": js_sim,
            "rhetorical_ratio_diff": float(abs(dist_a[ratio_idx] - dist_b[ratio_idx])),
            "rhetorical_fact_diff": float(abs(dist_a[fact_idx] - dist_b[fact_idx])),
        }

