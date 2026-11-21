"""
Case-Level Explanation Generator for Legal Prior Case Retrieval.

Constructs factual, domain-specific structured explanations for retrieved candidate judgments:
- Decomposes overall hybrid score into modality contributions.
- Resolves exact shared statutory provisions (e.g. IPC, CrPC, CPC).
- Identifies shared SCC/AIR precedent citations and reported case references.
- Identifies common judicial benches, courts, and legal authorities.
- Analyzes rhetorical discourse alignment.
- Synthesizes matching vs non-matching substantive factors without hallucination.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple, Any, Union
import json
from pathlib import Path
import numpy as np
import pandas as pd


class CaseExplanationGenerator:
    """Generates structured, factual legal explanations for retrieved case judgments."""

    def __init__(
        self,
        query_legal_features_path: Union[str, Path] = "data/processed/IL-TUR-PCR/dev_queries/legal_features.json",
        cand_legal_features_path: Union[str, Path] = "data/processed/IL-TUR-PCR/dev_candidates/legal_features.json",
        weights: Sequence[float] = (0.0, 0.65, 0.25, 0.10),
    ):
        """
        Args:
            query_legal_features_path: Path to pre-extracted query legal features JSON.
            cand_legal_features_path: Path to pre-extracted candidate legal features JSON.
            weights: Fusion weights [w_bm25, w_tfidf, w_dense, w_legal].
        """
        self.weights = list(weights)
        self.query_legal = self._load_json(Path(query_legal_features_path))
        self.cand_legal = self._load_json(Path(cand_legal_features_path))

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_explanation(
        self,
        query_id: str,
        candidate_id: str,
        pair_row: Optional[Union[pd.Series, Dict[str, Any]]] = None,
        rank: Optional[int] = None,
        pair_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive factual explanation for a single query-candidate pair.

        Args:
            query_id: Identifier of the query case.
            candidate_id: Identifier of the candidate document.
            pair_row: Series or dict containing feature values for this pair.
            rank: Optional rank of this candidate within the query's candidate pool.
            pair_features: Optional alias for pair_row.

        Returns:
            Structured dictionary of factual explanations.
        """
        if pair_row is None:
            pair_row = pair_features or {}
        # 1. Score decomposition
        qnorm_bm25 = float(pair_row.get("_qnorm_bm25_score", 0.0))
        qnorm_tfidf = float(pair_row.get("_qnorm_tfidf_score", 0.0))
        qnorm_dense = float(pair_row.get("_qnorm_inlegalbert_sim", 0.0))
        qnorm_legal = float(pair_row.get("_qnorm_composite_legal_score", 0.0))

        contrib_bm25 = float(pair_row.get("contrib_bm25_score", self.weights[0] * qnorm_bm25))
        contrib_tfidf = float(pair_row.get("contrib_tfidf_score", self.weights[1] * qnorm_tfidf))
        contrib_dense = float(pair_row.get("contrib_inlegalbert_sim", self.weights[2] * qnorm_dense))
        contrib_legal = float(pair_row.get("contrib_composite_legal_score", self.weights[3] * qnorm_legal))

        hybrid_score = float(pair_row.get("hybrid_score", contrib_bm25 + contrib_tfidf + contrib_dense + contrib_legal))
        is_relevant = bool(pair_row.get("label", 0) == 1)

        # 2. Extract shared legal entities from stored feature structures
        q_feat = self.query_legal.get(query_id, {})
        c_feat = self.cand_legal.get(candidate_id, {})

        q_stats = set(q_feat.get("statute_keys", []))
        c_stats = set(c_feat.get("statute_keys", []))
        shared_statutes = sorted(list(q_stats.intersection(c_stats)))

        q_cits = set(q_feat.get("citation_keys", []))
        c_cits = set(c_feat.get("citation_keys", []))
        shared_citations = sorted(list(q_cits.intersection(c_cits)))

        q_ents = set(q_feat.get("entity_keys", []))
        c_ents = set(c_feat.get("entity_keys", []))
        shared_entities = sorted(list(q_ents.intersection(c_ents)))

        # Sub-divide entities
        shared_judges = [e.replace("JUDGE:", "") for e in shared_entities if e.startswith("JUDGE:")]
        shared_courts = [e.replace("COURT:", "") for e in shared_entities if e.startswith("COURT:")]

        # Rhetorical similarity
        rhetorical_sim = float(pair_row.get("rhetorical_cosine", pair_row.get("raw_rhetorical_cosine", 0.0)))

        # 3. Identify substantive matching and non-matching factors
        matching_factors = []
        non_matching_factors = []

        if qnorm_tfidf > 0.7:
            matching_factors.append(f"Strong lexical phrasing overlap (TF-IDF score {qnorm_tfidf:.2f}/1.00)")
        elif qnorm_tfidf < 0.3:
            non_matching_factors.append(f"Low lexical overlap (TF-IDF score {qnorm_tfidf:.2f}/1.00)")

        if qnorm_dense > 0.7:
            matching_factors.append(f"High semantic similarity via InLegalBERT ({qnorm_dense:.2f}/1.00)")
        elif qnorm_dense < 0.3:
            non_matching_factors.append(f"Low dense semantic similarity ({qnorm_dense:.2f}/1.00)")

        if shared_statutes:
            matching_factors.append(f"Shared statutory provisions: {', '.join(shared_statutes[:5])}")
        else:
            if q_stats:
                non_matching_factors.append("No statutory provisions shared with query")

        if shared_citations:
            matching_factors.append(f"Shared formal precedent citations: {', '.join(shared_citations[:5])}")
        else:
            if q_cits:
                non_matching_factors.append("No reported precedent citations shared with query")

        if shared_judges:
            matching_factors.append(f"Coram overlap (Shared Judges): {', '.join(shared_judges)}")

        if shared_courts:
            matching_factors.append(f"Jurisdictional forum: {', '.join(shared_courts)}")

        if rhetorical_sim > 0.8:
            matching_factors.append(f"Highly compatible rhetorical discourse structure (similarity {rhetorical_sim:.2f})")

        return {
            "query_id": query_id,
            "candidate_id": candidate_id,
            "rank": rank,
            "is_ground_truth_precedent": is_relevant,
            "hybrid_score": round(hybrid_score, 5),
            "score_breakdown": {
                "tfidf_contribution": round(contrib_tfidf, 5),
                "inlegalbert_contribution": round(contrib_dense, 5),
                "legal_knowledge_contribution": round(contrib_legal, 5),
                "bm25_contribution": round(contrib_bm25, 5),
                "percentage_tfidf": round((contrib_tfidf / (hybrid_score + 1e-9)) * 100, 1),
                "percentage_dense": round((contrib_dense / (hybrid_score + 1e-9)) * 100, 1),
                "percentage_legal": round((contrib_legal / (hybrid_score + 1e-9)) * 100, 1),
            },
            "legal_features": {
                "shared_statutory_provisions": shared_statutes,
                "shared_statutory_count": len(shared_statutes),
                "shared_precedent_citations": shared_citations,
                "shared_citation_count": len(shared_citations),
                "shared_judges": shared_judges,
                "shared_courts": shared_courts,
                "rhetorical_similarity": round(rhetorical_sim, 4),
            },
            "interpretation": {
                "primary_retrieval_driver": (
                    "TF-IDF Lexical Anchor" if contrib_tfidf >= contrib_dense and contrib_tfidf >= contrib_legal
                    else "InLegalBERT Dense Semantics" if contrib_dense >= contrib_legal
                    else "Explicit Legal Knowledge"
                ),
                "matching_factors": matching_factors,
                "non_matching_factors": non_matching_factors,
            },
        }

    def explain_top_k_for_query(
        self,
        query_id: str,
        df_attributed: pd.DataFrame,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generates explanations for the top-k retrieved candidates of a query."""
        sub = df_attributed[df_attributed["query_id"] == query_id].copy()
        if len(sub) == 0:
            raise ValueError(f"Query '{query_id}' not found.")

        sub_sorted = sub.sort_values(by="hybrid_score", ascending=False).head(top_k)

        explanations = []
        for rank, (_, row) in enumerate(sub_sorted.iterrows(), start=1):
            cand_id = str(row["candidate_id"])
            exp = self.generate_explanation(
                query_id=query_id,
                candidate_id=cand_id,
                pair_row=row,
                rank=rank,
            )
            explanations.append(exp)

        return explanations

