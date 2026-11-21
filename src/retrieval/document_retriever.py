"""
Document-Level Candidate Aggregator and Retriever.

Aggregates chunk-level similarity scores into document-level scores and
handles candidate ranking with query self-exclusion.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class DocumentRetriever:
    """
    Manages candidate document indexing, chunk-to-document score aggregation,
    and query self-exclusion ranking.
    """

    def __init__(
        self,
        candidate_doc_ids: Sequence[str],
        chunk_doc_ids: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            candidate_doc_ids: Ordered sequence of candidate document IDs.
            chunk_doc_ids: Ordered sequence of document IDs corresponding to each chunk.
                           Required if aggregating chunk scores.
        """
        # Ensure candidate doc IDs are unique and ordered
        self.candidate_doc_ids = np.array(list(dict.fromkeys(candidate_doc_ids)), dtype=object)
        self.n_docs = len(self.candidate_doc_ids)
        self.doc_id_to_idx: Dict[str, int] = {
            doc_id: idx for idx, doc_id in enumerate(self.candidate_doc_ids)
        }

        self.chunk_doc_ids = chunk_doc_ids
        self.doc_chunk_indices: Optional[List[np.ndarray]] = None
        self.chunk_doc_indices: Optional[np.ndarray] = None
        self.is_contiguous: bool = False
        self.doc_starts: Optional[np.ndarray] = None
        self.doc_ends: Optional[np.ndarray] = None

        if chunk_doc_ids is not None:
            self._build_chunk_mapping(chunk_doc_ids)

    def _build_chunk_mapping(self, chunk_doc_ids: Sequence[str]) -> None:
        """Precomputes chunk-to-doc mappings for high-performance vectorized aggregation."""
        chunk_indices = []
        doc_chunk_dict: Dict[int, List[int]] = {idx: [] for idx in range(self.n_docs)}

        for c_idx, doc_id in enumerate(chunk_doc_ids):
            if doc_id in self.doc_id_to_idx:
                d_idx = self.doc_id_to_idx[doc_id]
                chunk_indices.append(d_idx)
                doc_chunk_dict[d_idx].append(c_idx)
            else:
                chunk_indices.append(-1)

        self.chunk_doc_indices = np.array(chunk_indices, dtype=np.int32)
        self.doc_chunk_indices = [
            np.array(doc_chunk_dict[d_idx], dtype=np.int32)
            for d_idx in range(self.n_docs)
        ]

        # Check if chunks are sorted contiguously by candidate doc index
        diffs = np.diff(self.chunk_doc_indices)
        if np.all(diffs >= 0):
            self.is_contiguous = True
            doc_counts = np.bincount(self.chunk_doc_indices, minlength=self.n_docs)
            ends = np.cumsum(doc_counts)
            starts = np.r_[0, ends[:-1]]
            self.doc_starts = starts
            self.doc_ends = ends

    def aggregate_chunk_scores(
        self,
        chunk_scores: np.ndarray,
        strategy: str = "max",
        top_k: int = 3,
    ) -> np.ndarray:
        """
        Aggregates 1D array of chunk scores into candidate document scores.
        
        Args:
            chunk_scores: 1D array of shape (n_chunks,)
            strategy: Aggregation strategy ('max', 'top_k_mean', 'top_k_sum', 'mean')
            top_k: K value for top-K aggregation strategies
            
        Returns:
            np.ndarray: 1D array of document scores of shape (n_docs,)
        """
        if self.chunk_doc_indices is None or self.doc_chunk_indices is None:
            raise RuntimeError("Chunk mapping not initialized. Provide chunk_doc_ids during init.")

        strategy = strategy.lower().replace("-", "_")

        if strategy in ("max", "top_k_max"):
            doc_scores = np.full(self.n_docs, -np.inf, dtype=np.float32)
            # np.maximum.at is highly optimized in C
            np.maximum.at(doc_scores, self.chunk_doc_indices, chunk_scores)
            return doc_scores

        doc_scores = np.zeros(self.n_docs, dtype=np.float32)

        if self.is_contiguous and self.doc_starts is not None and self.doc_ends is not None:
            for d in range(self.n_docs):
                s = chunk_scores[self.doc_starts[d]:self.doc_ends[d]]
                if len(s) == 0:
                    doc_scores[d] = -np.inf
                elif strategy == "top_k_mean":
                    if len(s) <= top_k:
                        doc_scores[d] = s.mean()
                    else:
                        doc_scores[d] = (-np.partition(-s, top_k)[:top_k]).mean()
                elif strategy == "top_k_sum":
                    if len(s) <= top_k:
                        doc_scores[d] = s.sum()
                    else:
                        doc_scores[d] = (-np.partition(-s, top_k)[:top_k]).sum()
                elif strategy == "mean":
                    doc_scores[d] = s.mean()
                else:
                    raise ValueError(f"Unknown aggregation strategy: {strategy}")
            return doc_scores

        # General non-contiguous fallback using pre-grouped chunk indices
        for d in range(self.n_docs):
            c_indices = self.doc_chunk_indices[d]
            if len(c_indices) == 0:
                doc_scores[d] = -np.inf
                continue
            s = chunk_scores[c_indices]
            if strategy == "top_k_mean":
                if len(s) <= top_k:
                    doc_scores[d] = s.mean()
                else:
                    doc_scores[d] = (-np.partition(-s, top_k)[:top_k]).mean()
            elif strategy == "top_k_sum":
                if len(s) <= top_k:
                    doc_scores[d] = s.sum()
                else:
                    doc_scores[d] = (-np.partition(-s, top_k)[:top_k]).sum()
            elif strategy == "mean":
                doc_scores[d] = s.mean()
            else:
                raise ValueError(f"Unknown aggregation strategy: {strategy}")

        return doc_scores

    def rank_candidates(
        self,
        doc_scores: np.ndarray,
        query_id: Optional[str] = None,
        top_n: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Ranks candidate documents given their similarity scores, enforcing query self-exclusion.
        
        Args:
            doc_scores: 1D array of scores for each candidate document (length self.n_docs).
            query_id: Case ID of the query. If present in the candidate pool, it is excluded.
            top_n: Number of top candidate documents to return.
            
        Returns:
            List of (candidate_doc_id, score) tuples ordered descending by score.
        """
        scores = doc_scores.copy()

        # Query self-exclusion: Mask out query ID so it is never retrieved
        if query_id is not None and query_id in self.doc_id_to_idx:
            self_idx = self.doc_id_to_idx[query_id]
            scores[self_idx] = -np.inf

        num_valid = np.count_nonzero(np.isfinite(scores))
        k = min(top_n, num_valid, self.n_docs)
        if k <= 0:
            return []

        # Fast argpartition for top-k selection followed by exact descending sort
        top_indices = np.argpartition(-scores, k - 1)[:k]
        sorted_top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = [
            (str(self.candidate_doc_ids[idx]), float(scores[idx]))
            for idx in sorted_top_indices
            if np.isfinite(scores[idx])
        ]
        return results

