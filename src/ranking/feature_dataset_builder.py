"""
Feature Dataset Builder for Hybrid Legal Case Retrieval.

Aligns and constructs unified query-candidate feature records across:
- Lexical retrieval: BM25 score, TF-IDF score, BM25 rank, TF-IDF rank
- Dense semantic retrieval: InLegalBERT cosine similarity & rank (and BERT-Base if present)
- Legal knowledge features: Statutory Jaccard/containment/count, citation overlap,
  entity & actor overlap, and discourse distribution alignment
- Official relevance labels: 1 for relevant precedent, 0 for non-relevant
- Query self-exclusion: Strictly enforces candidate_id != query_id
"""

from typing import Dict, List, Set, Sequence, Tuple, Any, Optional
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.tfidf_retriever import TFIDFRetriever
from src.legal_features.legal_feature_encoder import DocumentLegalFeatures, LegalFeatureEncoder
from src.ranking.negative_sampler import NegativeSampler


class FeatureDatasetBuilder:
    """Constructs aligned multi-modal pairwise feature datasets for ranking models."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.encoder = LegalFeatureEncoder()

    @staticmethod
    def _compute_ranks(score_matrix: np.ndarray) -> np.ndarray:
        """
        Computes 1-based ranks along candidate dimension (axis=1) for each query.
        Rank 1 corresponds to highest score.
        """
        # argsort twice gives ranks; descending order: invert scores
        order = np.argsort(-score_matrix, axis=1)
        ranks = np.empty_like(order)
        for i in range(score_matrix.shape[0]):
            ranks[i, order[i]] = np.arange(1, score_matrix.shape[1] + 1)
        return ranks

    def build_dataset_for_split(
        self,
        split_name: str,
        queries_df: pd.DataFrame,
        candidates_df: pd.DataFrame,
        query_embeddings: np.ndarray,
        candidate_embeddings: np.ndarray,
        query_legal_features: Dict[str, DocumentLegalFeatures],
        cand_legal_features: Dict[str, DocumentLegalFeatures],
        negative_sampler: Optional[NegativeSampler] = None,
        bert_base_q_embs: Optional[np.ndarray] = None,
        bert_base_c_embs: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Constructs the pairwise feature dataset for a partition.
        
        Args:
            split_name: 'train', 'dev', or 'test'.
            queries_df: DataFrame with 'id', 'full_text', 'relevant_candidates'.
            candidates_df: DataFrame with 'id', 'full_text'.
            query_embeddings: (n_queries, dim) InLegalBERT embeddings.
            candidate_embeddings: (n_candidates, dim) InLegalBERT embeddings.
            query_legal_features: Map of query_id -> DocumentLegalFeatures.
            cand_legal_features: Map of cand_id -> DocumentLegalFeatures.
            negative_sampler: Optional NegativeSampler (used for train; None for full dev/test matrix).
            bert_base_q_embs: Optional BERT Base query embeddings.
            bert_base_c_embs: Optional BERT Base candidate embeddings.
            
        Returns:
            pd.DataFrame containing all pairwise feature records.
        """
        t0 = time.time()
        n_queries = len(queries_df)
        n_candidates = len(candidates_df)
        print(f"\nBuilding pairwise hybrid dataset for [{split_name}]: {n_queries} queries x {n_candidates} candidates...")

        query_ids = list(queries_df["id"])
        cand_ids = list(candidates_df["id"])
        cand_id_to_idx = {cid: idx for idx, cid in enumerate(cand_ids)}

        # 1. Lexical Scoring (BM25)
        print("  [1/4] Computing BM25 lexical similarity matrix and ranks...")
        bm25 = BM25Retriever().fit(candidates_df["full_text"].values)
        bm25_matrix = bm25.score_queries(queries_df["full_text"].values).toarray()
        bm25_ranks = self._compute_ranks(bm25_matrix)

        # 2. Lexical Scoring (TF-IDF)
        print("  [2/4] Computing TF-IDF cosine similarity matrix and ranks...")
        tfidf = TFIDFRetriever().fit(candidates_df["full_text"].values)
        tfidf_matrix = tfidf.score_queries(queries_df["full_text"].values).toarray()
        tfidf_ranks = self._compute_ranks(tfidf_matrix)

        # 3. Dense Scoring (InLegalBERT)
        print("  [3/4] Computing InLegalBERT dense cosine similarity matrix and ranks...")
        # Since embeddings are L2-normalized, inner product equals cosine similarity
        inlegalbert_sim_matrix = np.dot(query_embeddings, candidate_embeddings.T)
        inlegalbert_ranks = self._compute_ranks(inlegalbert_sim_matrix)

        # Optional: BERT Base
        has_bert_base = (bert_base_q_embs is not None and bert_base_c_embs is not None)
        if has_bert_base:
            bert_base_sim_matrix = np.dot(bert_base_q_embs, bert_base_c_embs.T)
            bert_base_ranks = self._compute_ranks(bert_base_sim_matrix)

        # 4. Pre-index Legal Features for rapid vectorized / set-based lookup
        print("  [4/4] Pre-indexing legal knowledge entities, statutes, and discourse profiles...")
        cand_stat_sets = {cid: set(f.statute_keys) for cid, f in cand_legal_features.items()}
        cand_cit_sets = {cid: set(f.citation_keys) for cid, f in cand_legal_features.items()}
        cand_ent_sets = {cid: set(f.entity_keys) for cid, f in cand_legal_features.items()}

        # Rhetorical distributions matrix (n_candidates, 11)
        c_rhet_mat = np.zeros((n_candidates, 11), dtype=np.float32)
        for idx, cid in enumerate(cand_ids):
            if cid in cand_legal_features:
                c_rhet_mat[idx] = np.array(cand_legal_features[cid].rhetorical_distribution, dtype=np.float32)
        c_rhet_norms = np.linalg.norm(c_rhet_mat, axis=1, keepdims=True)
        c_rhet_norms[c_rhet_norms == 0] = 1e-6
        c_rhet_normalized = c_rhet_mat / c_rhet_norms

        # Build pair records
        records: List[Dict[str, Any]] = []
        positive_pairs_count = 0
        negative_pairs_count = 0

        for q_idx, q_row in queries_df.iterrows():
            qid = str(q_row["id"])
            pos_cands_raw = q_row.get("relevant_candidates", [])
            pos_cands = set(pos_cands_raw) if isinstance(pos_cands_raw, (list, np.ndarray, set)) else set()

            # Query legal features
            q_legal = query_legal_features.get(qid)
            q_stat_set = set(q_legal.statute_keys) if q_legal else set()
            q_cit_set = set(q_legal.citation_keys) if q_legal else set()
            q_ent_set = set(q_legal.entity_keys) if q_legal else set()
            q_rhet_vec = np.array(q_legal.rhetorical_distribution, dtype=np.float32) if q_legal else np.zeros(11, dtype=np.float32)
            q_rhet_norm = np.linalg.norm(q_rhet_vec)
            q_rhet_normed = q_rhet_vec / (q_rhet_norm if q_rhet_norm > 0 else 1e-6)

            # Vectorized rhetorical cosine with all candidates
            q_cand_rhet_cosines = np.dot(c_rhet_normalized, q_rhet_normed)

            # Determine which candidate IDs to pair with this query
            if negative_sampler is not None:
                # Sampled pairs for training
                sample_res = negative_sampler.sample_candidates_for_query(
                    query_id=qid,
                    positive_candidate_ids=pos_cands,
                    candidate_pool_ids=cand_ids,
                    candidate_scores=bm25_matrix[q_idx],
                )
                selected_cids = sample_res["selected_candidate_ids"]
                selected_labels = sample_res["labels"]
            else:
                # Complete official candidate pool for dev/test evaluation matrix (excluding self-match)
                selected_cids = [cid for cid in cand_ids if cid != qid]
                selected_labels = [1 if cid in pos_cands else 0 for cid in selected_cids]

            for cid, label in zip(selected_cids, selected_labels):
                c_idx = cand_id_to_idx[cid]

                if label == 1:
                    positive_pairs_count += 1
                else:
                    negative_pairs_count += 1

                # 1. Lexical features
                bm25_val = float(bm25_matrix[q_idx, c_idx])
                bm25_rk = int(bm25_ranks[q_idx, c_idx])
                tfidf_val = float(tfidf_matrix[q_idx, c_idx])
                tfidf_rk = int(tfidf_ranks[q_idx, c_idx])

                # 2. Dense semantic features
                dense_sim = float(inlegalbert_sim_matrix[q_idx, c_idx])
                dense_rk = int(inlegalbert_ranks[q_idx, c_idx])

                # 3. Legal features
                c_stat_set = cand_stat_sets.get(cid, set())
                c_cit_set = cand_cit_sets.get(cid, set())
                c_ent_set = cand_ent_sets.get(cid, set())

                # Statute overlap
                stat_inter = q_stat_set.intersection(c_stat_set)
                stat_union = q_stat_set.union(c_stat_set)
                stat_jaccard = len(stat_inter) / max(len(stat_union), 1)
                stat_contain = len(stat_inter) / max(len(q_stat_set), 1)
                stat_count = float(len(stat_inter))

                # Citation overlap
                cit_inter = q_cit_set.intersection(c_cit_set)
                cit_union = q_cit_set.union(c_cit_set)
                cit_jaccard = len(cit_inter) / max(len(cit_union), 1)
                cit_count = float(len(cit_inter))

                # Entity overlap
                ent_inter = q_ent_set.intersection(c_ent_set)
                ent_union = q_ent_set.union(c_ent_set)
                ent_jaccard = len(ent_inter) / max(len(ent_union), 1)
                ent_count = float(len(ent_inter))

                # Sub-entity overlap
                def cat_overlap(prefix: str) -> float:
                    sub_q = {k for k in q_ent_set if k.startswith(prefix)}
                    sub_c = {k for k in c_ent_set if k.startswith(prefix)}
                    if not sub_q or not sub_c:
                        return 0.0
                    return len(sub_q.intersection(sub_c)) / max(len(sub_q.union(sub_c)), 1)

                judge_ol = cat_overlap("JUDGE:")
                court_ol = cat_overlap("COURT:")
                org_ol = cat_overlap("ORG:")

                # Rhetorical overlap
                rhet_cos = float(q_cand_rhet_cosines[c_idx])
                c_rhet_vec = c_rhet_mat[c_idx]
                if q_rhet_norm > 0 and np.linalg.norm(c_rhet_vec) > 0:
                    js_dist = jensenshannon(q_rhet_vec, c_rhet_vec, base=2)
                    rhet_js = float(max(0.0, 1.0 - js_dist))
                else:
                    rhet_js = 0.0

                ratio_diff = float(abs(q_rhet_vec[8] - c_rhet_vec[8]))  # Ratio index is 8 (RatioOfTheDecision)

                # Composite legal score
                composite_legal = float(
                    0.40 * stat_jaccard +
                    0.25 * cit_jaccard +
                    0.15 * ent_jaccard +
                    0.10 * judge_ol +
                    0.10 * rhet_cos
                )

                row_dict: Dict[str, Any] = {
                    "query_id": qid,
                    "candidate_id": cid,
                    "split": split_name,
                    "label": int(label),
                    # Lexical
                    "bm25_score": bm25_val,
                    "bm25_rank": bm25_rk,
                    "tfidf_score": tfidf_val,
                    "tfidf_rank": tfidf_rk,
                    # Dense
                    "inlegalbert_sim": dense_sim,
                    "inlegalbert_rank": dense_rk,
                    # Legal Knowledge
                    "statute_jaccard": float(stat_jaccard),
                    "statute_containment_query": float(stat_contain),
                    "statute_shared_count": stat_count,
                    "citation_jaccard": float(cit_jaccard),
                    "citation_shared_count": cit_count,
                    "entity_jaccard": float(ent_jaccard),
                    "entity_shared_count": ent_count,
                    "judge_overlap": float(judge_ol),
                    "court_overlap": float(court_ol),
                    "org_overlap": float(org_ol),
                    "rhetorical_cosine": rhet_cos,
                    "rhetorical_js_sim": rhet_js,
                    "rhetorical_ratio_diff": ratio_diff,
                    "composite_legal_score": composite_legal,
                }

                if has_bert_base:
                    row_dict["bert_base_sim"] = float(bert_base_sim_matrix[q_idx, c_idx])
                    row_dict["bert_base_rank"] = int(bert_base_ranks[q_idx, c_idx])

                records.append(row_dict)

        df_out = pd.DataFrame(records)
        elapsed = time.time() - t0
        print(f"  Generated {len(df_out):,} pairs in {elapsed:.2f}s (Positives: {positive_pairs_count:,} | Negatives: {negative_pairs_count:,})")
        return df_out

