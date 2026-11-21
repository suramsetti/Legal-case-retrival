"""
Legal Feature Encoder for Indian Supreme Court Judgments.

Aggregates extracted legal entities, statutory provisions, precedent citations,
and rhetorical role distributions into unified document-level structured features
and pairwise similarity feature vectors for hybrid case retrieval.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re
from typing import List, Dict, Set, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.legal_features.statute_extractor import StatuteExtractor, StatuteReference
from src.legal_features.citation_extractor import CitationExtractor, LegalCitation
from src.legal_features.entity_extractor import LegalEntityExtractor, LegalEntity
from src.legal_features.rhetorical_analyzer import RhetoricalAnalyzer, SentenceRole, RHETORICAL_ROLES, NUM_ROLES


@dataclass
class DocumentLegalFeatures:
    doc_id: str
    statutes: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    rhetorical_distribution: List[float]
    num_sentences: int
    num_statutes: int
    num_citations: int
    num_entities: int
    statute_keys: List[str]
    citation_keys: List[str]
    entity_keys: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentLegalFeatures":
        return cls(**data)


class LegalFeatureEncoder:
    """Orchestrates multi-faceted legal knowledge extraction and pairwise similarity encoding."""

    def __init__(self):
        self.statute_extractor = StatuteExtractor()
        self.citation_extractor = CitationExtractor()
        self.entity_extractor = LegalEntityExtractor()
        self.rhetorical_analyzer = RhetoricalAnalyzer()

    def encode_document(
        self,
        doc_id: str,
        full_text: str,
        sentences: Optional[List[str]] = None,
    ) -> DocumentLegalFeatures:
        """
        Extracts all legal features from a single judicial opinion.
        
        Args:
            doc_id: Unique case identifier.
            full_text: Full reconstructed judgment text.
            sentences: Optional pre-tokenized sentence list.
            
        Returns:
            DocumentLegalFeatures structured object.
        """
        if sentences is None:
            # Fallback simple sentence splitting
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full_text) if s.strip()]

        # 1. Statutory references
        statute_refs = self.statute_extractor.extract(full_text)
        statute_dicts = [s.to_dict() for s in statute_refs]
        statute_keys = sorted(list(self.statute_extractor.get_normalized_keys(statute_refs)))

        # 2. Citations
        citation_refs = self.citation_extractor.extract(full_text)
        citation_dicts = [c.to_dict() for c in citation_refs]
        citation_keys = sorted(list(self.citation_extractor.get_normalized_keys(citation_refs)))

        # 3. Entities
        entity_refs = self.entity_extractor.extract(full_text)
        entity_dicts = [e.to_dict() for e in entity_refs]
        entity_keys = sorted(list({e.normalized_key for e in entity_refs}))

        # 4. Rhetorical roles
        _, distribution = self.rhetorical_analyzer.analyze_document(sentences)

        return DocumentLegalFeatures(
            doc_id=str(doc_id),
            statutes=statute_dicts,
            citations=citation_dicts,
            entities=entity_dicts,
            rhetorical_distribution=distribution.tolist(),
            num_sentences=len(sentences),
            num_statutes=len(statute_refs),
            num_citations=len(citation_refs),
            num_entities=len(entity_refs),
            statute_keys=statute_keys,
            citation_keys=citation_keys,
            entity_keys=entity_keys,
        )

    def compute_pairwise_features(
        self,
        query_feats: DocumentLegalFeatures,
        cand_feats: DocumentLegalFeatures,
    ) -> Dict[str, float]:
        """
        Computes pairwise legal similarity features between a query case and a candidate case.
        
        Args:
            query_feats: Legal features of the query case.
            cand_feats: Legal features of the candidate case.
            
        Returns:
            Dictionary of scalar legal similarity features.
        """
        # 1. Statute Overlap
        keys_q_stat = set(query_feats.statute_keys)
        keys_c_stat = set(cand_feats.statute_keys)
        if keys_q_stat or keys_c_stat:
            stat_inter = keys_q_stat.intersection(keys_c_stat)
            stat_union = keys_q_stat.union(keys_c_stat)
            stat_jaccard = len(stat_inter) / max(len(stat_union), 1)
            stat_contain_q = len(stat_inter) / max(len(keys_q_stat), 1)
            stat_contain_c = len(stat_inter) / max(len(keys_c_stat), 1)
            stat_shared = float(len(stat_inter))
        else:
            stat_jaccard = 0.0
            stat_contain_q = 0.0
            stat_contain_c = 0.0
            stat_shared = 0.0

        # 2. Citation Overlap
        keys_q_cit = set(query_feats.citation_keys)
        keys_c_cit = set(cand_feats.citation_keys)
        if keys_q_cit or keys_c_cit:
            cit_inter = keys_q_cit.intersection(keys_c_cit)
            cit_union = keys_q_cit.union(keys_c_cit)
            cit_jaccard = len(cit_inter) / max(len(cit_union), 1)
            cit_shared = float(len(cit_inter))
            cit_contain_q = len(cit_inter) / max(len(keys_q_cit), 1)
            cit_contain_c = len(cit_inter) / max(len(keys_c_cit), 1)
        else:
            cit_jaccard = 0.0
            cit_shared = 0.0
            cit_contain_q = 0.0
            cit_contain_c = 0.0

        # 3. Entity Overlap
        keys_q_ent = set(query_feats.entity_keys)
        keys_c_ent = set(cand_feats.entity_keys)
        if keys_q_ent or keys_c_ent:
            ent_inter = keys_q_ent.intersection(keys_c_ent)
            ent_union = keys_q_ent.union(keys_c_ent)
            ent_jaccard = len(ent_inter) / max(len(ent_union), 1)
            ent_shared = float(len(ent_inter))
        else:
            ent_jaccard = 0.0
            ent_shared = 0.0

        # Sub-entity overlaps (judges, courts, orgs, locs)
        def category_overlap(prefix: str) -> float:
            sub_q = {k for k in keys_q_ent if k.startswith(prefix)}
            sub_c = {k for k in keys_c_ent if k.startswith(prefix)}
            if not sub_q or not sub_c:
                return 0.0
            return len(sub_q.intersection(sub_c)) / max(len(sub_q.union(sub_c)), 1)

        judge_overlap = category_overlap("JUDGE:")
        court_overlap = category_overlap("COURT:")
        org_overlap = category_overlap("ORG:")
        loc_overlap = category_overlap("LOC:")

        # 4. Rhetorical Role Distribution Similarity
        dist_q = np.array(query_feats.rhetorical_distribution, dtype=np.float32)
        dist_c = np.array(cand_feats.rhetorical_distribution, dtype=np.float32)
        rhet_sim = self.rhetorical_analyzer.compute_distribution_similarity(dist_q, dist_c)

        # 5. Composite Legal Feature Score
        # Linear balance across the major dimensions
        composite_score = (
            0.40 * stat_jaccard +
            0.25 * cit_jaccard +
            0.15 * ent_jaccard +
            0.10 * judge_overlap +
            0.10 * rhet_sim["rhetorical_cosine"]
        )

        return {
            "statute_jaccard": float(stat_jaccard),
            "statute_containment_query": float(stat_contain_q),
            "statute_containment_cand": float(stat_contain_c),
            "statute_shared_count": float(stat_shared),
            "citation_jaccard": float(cit_jaccard),
            "citation_shared_count": float(cit_shared),
            "citation_containment_query": float(cit_contain_q),
            "citation_containment_cand": float(cit_contain_c),
            "entity_jaccard": float(ent_jaccard),
            "entity_shared_count": float(ent_shared),
            "judge_overlap": float(judge_overlap),
            "court_overlap": float(court_overlap),
            "org_overlap": float(org_overlap),
            "loc_overlap": float(loc_overlap),
            "rhetorical_cosine": float(rhet_sim["rhetorical_cosine"]),
            "rhetorical_js_sim": float(rhet_sim["rhetorical_js_sim"]),
            "rhetorical_ratio_diff": float(rhet_sim["rhetorical_ratio_diff"]),
            "rhetorical_fact_diff": float(rhet_sim["rhetorical_fact_diff"]),
            "composite_legal_score": float(composite_score),
        }

    def encode_split(
        self,
        split_name: str,
        documents_df: pd.DataFrame,
        save_path: Optional[Path] = None,
    ) -> Dict[str, DocumentLegalFeatures]:
        """
        Batch encodes all documents in a partition DataFrame and optionally saves to JSON.
        
        Args:
            split_name: Name of the partition (e.g. 'dev_queries').
            documents_df: DataFrame containing 'id', 'full_text', and optionally 'text' (sentence list).
            save_path: Optional destination json file.
            
        Returns:
            Dict mapping doc_id -> DocumentLegalFeatures.
        """
        encoded_dict: Dict[str, DocumentLegalFeatures] = {}
        total = len(documents_df)
        print(f"Extracting legal features for {split_name} ({total} documents)...")

        for idx, row in documents_df.iterrows():
            doc_id = str(row["id"])
            full_text = str(row.get("full_text", ""))
            sentences = None
            if "text" in row and isinstance(row["text"], (list, np.ndarray)):
                sentences = list(row["text"])

            feats = self.encode_document(doc_id=doc_id, full_text=full_text, sentences=sentences)
            encoded_dict[doc_id] = feats

            if (idx + 1) % 250 == 0 or (idx + 1) == total:
                print(f"  Processed {idx + 1}/{total} documents...")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump({k: v.to_dict() for k, v in encoded_dict.items()}, f, indent=2)
            print(f"Saved {len(encoded_dict)} legal feature records to {save_path}")

        return encoded_dict

    @staticmethod
    def compute_corpus_statistics(
        features_dict: Dict[str, DocumentLegalFeatures]
    ) -> Dict[str, Any]:
        """Computes extraction coverage and summary statistics across a set of legal features."""
        total_docs = len(features_dict)
        if total_docs == 0:
            return {}

        docs_with_statutes = sum(1 for f in features_dict.values() if f.num_statutes > 0)
        docs_with_citations = sum(1 for f in features_dict.values() if f.num_citations > 0)
        docs_with_entities = sum(1 for f in features_dict.values() if f.num_entities > 0)

        statute_counts = [f.num_statutes for f in features_dict.values()]
        citation_counts = [f.num_citations for f in features_dict.values()]
        entity_counts = [f.num_entities for f in features_dict.values()]

        unique_statute_keys = set()
        unique_citation_keys = set()
        unique_entity_keys = set()
        for f in features_dict.values():
            unique_statute_keys.update(f.statute_keys)
            unique_citation_keys.update(f.citation_keys)
            unique_entity_keys.update(f.entity_keys)

        return {
            "total_documents": total_docs,
            "statute_coverage_pct": round(100.0 * docs_with_statutes / total_docs, 2),
            "citation_coverage_pct": round(100.0 * docs_with_citations / total_docs, 2),
            "entity_coverage_pct": round(100.0 * docs_with_entities / total_docs, 2),
            "avg_statutes_per_doc": round(float(np.mean(statute_counts)), 2),
            "median_statutes_per_doc": float(np.median(statute_counts)),
            "avg_citations_per_doc": round(float(np.mean(citation_counts)), 2),
            "median_citations_per_doc": float(np.median(citation_counts)),
            "avg_entities_per_doc": round(float(np.mean(entity_counts)), 2),
            "median_entities_per_doc": float(np.median(entity_counts)),
            "unique_statute_provisions_count": len(unique_statute_keys),
            "unique_citations_count": len(unique_citation_keys),
            "unique_entities_count": len(unique_entity_keys),
        }

