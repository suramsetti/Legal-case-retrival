"""
Dense Semantic Retriever for Legal Prior Case Retrieval.

Performs cosine similarity retrieval between query embeddings and candidate chunk embeddings,
aggregates chunk scores into document scores (Max, Top-K Mean), and enforces query self-exclusion.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.retrieval.document_retriever import DocumentRetriever


class DenseRetriever:
    """
    Computes dense cosine similarity rankings between queries and candidate chunks/documents.
    """

    def __init__(
        self,
        candidate_embeddings: np.ndarray,
        candidate_chunk_doc_ids: Sequence[str],
        candidate_doc_ids: Sequence[str],
    ):
        """
        Args:
            candidate_embeddings: L2-normalized ndarray of shape (n_chunks, hidden_dim).
            candidate_chunk_doc_ids: Document ID for each chunk.
            candidate_doc_ids: Unique candidate document IDs.
        """
        self.candidate_embeddings = candidate_embeddings.astype(np.float32)
        self.n_chunks, self.hidden_dim = self.candidate_embeddings.shape

        self.candidate_chunk_doc_ids = candidate_chunk_doc_ids
        self.candidate_doc_ids = candidate_doc_ids

        # Reuse DocumentRetriever's fast aggregation mappings and self-exclusion logic
        self.doc_retriever = DocumentRetriever(
            candidate_doc_ids=self.candidate_doc_ids,
            chunk_doc_ids=self.candidate_chunk_doc_ids,
        )

    def score_query_embeddings(
        self,
        query_embeddings: np.ndarray,
        batch_size: int = 128,
    ) -> np.ndarray:
        """
        Computes cosine similarity matrix between queries and all candidate chunks.
        Because both embeddings are L2 normalized, cosine similarity equals inner product.
        
        Args:
            query_embeddings: L2-normalized array of shape (n_queries, hidden_dim).
            batch_size: Query batch size for memory-bounded matrix multiplication.
            
        Returns:
            np.ndarray: Cosine similarity matrix of shape (n_queries, n_chunks).
        """
        n_queries = query_embeddings.shape[0]
        q_emb = query_embeddings.astype(np.float32)
        batches = []

        for start_idx in range(0, n_queries, batch_size):
            end_idx = min(start_idx + batch_size, n_queries)
            q_batch = q_emb[start_idx:end_idx]
            # (batch, hidden_dim) @ (hidden_dim, n_chunks) -> (batch, n_chunks)
            sim_batch = np.dot(q_batch, self.candidate_embeddings.T)
            batches.append(sim_batch)

        if len(batches) == 1:
            return batches[0]
        return np.vstack(batches)

    def retrieve_for_query(
        self,
        query_embedding: np.ndarray,
        query_id: Optional[str] = None,
        aggregation: str = "max",
        top_k: int = 3,
        top_n: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Retrieves and ranks top candidate documents for a single query embedding.
        
        Args:
            query_embedding: 1D array of shape (hidden_dim,).
            query_id: Query case ID for self-exclusion masking.
            aggregation: Aggregation strategy ('max', 'top_k_mean', 'top_k_sum').
            top_k: K for top-K aggregation.
            top_n: Number of candidates to retrieve.
            
        Returns:
            List of (candidate_doc_id, score) ordered descending by score.
        """
        # Chunk similarity vector: (n_chunks,)
        q_vec = query_embedding.astype(np.float32).reshape(1, -1)
        chunk_sims = np.dot(q_vec, self.candidate_embeddings.T).flatten()

        # Aggregate to candidate document scores
        doc_scores = self.doc_retriever.aggregate_chunk_scores(
            chunk_sims, strategy=aggregation, top_k=top_k
        )

        # Rank candidates with query self-exclusion
        return self.doc_retriever.rank_candidates(doc_scores, query_id=query_id, top_n=top_n)

    def retrieve_batch(
        self,
        query_embeddings: np.ndarray,
        query_ids: Sequence[str],
        aggregation: str = "max",
        top_k: int = 3,
        top_n: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """
        Batched retrieval for multiple queries with document score aggregation and self-exclusion.
        """
        sim_matrix = self.score_query_embeddings(query_embeddings)
        n_queries = len(query_ids)
        all_rankings = []

        for q_idx in range(n_queries):
            qid = query_ids[q_idx]
            chunk_sims = sim_matrix[q_idx]
            doc_scores = self.doc_retriever.aggregate_chunk_scores(
                chunk_sims, strategy=aggregation, top_k=top_k
            )
            ranked = self.doc_retriever.rank_candidates(doc_scores, query_id=qid, top_n=top_n)
            all_rankings.append(ranked)

        return all_rankings

