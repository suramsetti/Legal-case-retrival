"""
BM25 Lexical Retriever for Legal Prior Case Retrieval.

Implements a vectorized, memory-efficient BM25Okapi retriever using sparse CSR
matrices for rapid scoring of long legal texts and candidate chunks.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


class BM25Retriever:
    """
    Vectorized BM25Okapi retriever using sparse CSR matrix operations.
    
    Formula:
        Score(D, Q) = sum_{w in Q intersect D} IDF(w) * [TF(w, D) * (k1 + 1)] /
                      [TF(w, D) + k1 * (1 - b + b * (|D| / avgdl))]
                      
        IDF(w) = ln((N - n(w) + 0.5) / (n(w) + 0.5) + 1.0)  (Lucene/BM25+ standard)
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        max_features: Optional[int] = 50000,
        min_df: Union[int, float] = 2,
        max_df: Union[int, float] = 0.9,
        ngram_range: tuple = (1, 1),
        lowercase: bool = True,
    ):
        self.k1 = float(k1)
        self.b = float(b)
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.lowercase = lowercase

        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            lowercase=self.lowercase,
        )
        self.bm25_matrix: Optional[sp.csr_matrix] = None
        self.idf: Optional[np.ndarray] = None
        self.doc_lengths: Optional[np.ndarray] = None
        self.avgdl: float = 0.0
        self.n_docs: int = 0
        self.vocab_size: int = 0
        self.is_fitted: bool = False

    def fit(self, corpus_texts: Sequence[str]) -> "BM25Retriever":
        """
        Builds the BM25 term frequency matrix and IDF weights for corpus texts.
        
        Args:
            corpus_texts: Sequence of candidate documents or chunks.
            
        Returns:
            self
        """
        raw_counts = self.vectorizer.fit_transform(corpus_texts)
        self.n_docs, self.vocab_size = raw_counts.shape

        # Calculate document lengths and average length
        self.doc_lengths = np.array(raw_counts.sum(axis=1)).flatten()
        self.avgdl = float(self.doc_lengths.mean()) if self.n_docs > 0 else 1.0

        # Calculate document frequency n(w) for each vocabulary term
        df = np.bincount(raw_counts.indices, minlength=self.vocab_size)

        # Standard Lucene/BM25+ IDF: ln((N - df + 0.5) / (df + 0.5) + 1.0)
        self.idf = np.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

        # Vectorized BM25 term frequency transformation on CSR data array
        len_norm = 1.0 - self.b + self.b * (self.doc_lengths / self.avgdl)
        row_indices = np.repeat(np.arange(self.n_docs), np.diff(raw_counts.indptr))
        
        denom = raw_counts.data + self.k1 * len_norm[row_indices]
        bm25_data = (raw_counts.data * (self.k1 + 1.0)) / denom

        self.bm25_matrix = sp.csr_matrix(
            (bm25_data, raw_counts.indices, raw_counts.indptr),
            shape=raw_counts.shape,
            dtype=np.float32,
        )

        self.is_fitted = True
        return self

    def score_queries(
        self,
        query_texts: Sequence[str],
        batch_size: int = 256,
    ) -> sp.csr_matrix:
        """
        Computes BM25 similarity scores for a batch of query texts against the corpus.
        
        Args:
            query_texts: Sequence of query strings.
            batch_size: Batch size for transforming and multiplying query vectors.
            
        Returns:
            sp.csr_matrix: Score matrix of shape (n_queries, n_docs).
        """
        if not self.is_fitted or self.bm25_matrix is None or self.idf is None:
            raise RuntimeError("BM25Retriever must be fitted before scoring queries.")

        n_queries = len(query_texts)
        batches = []

        for start_idx in range(0, n_queries, batch_size):
            end_idx = min(start_idx + batch_size, n_queries)
            q_batch = query_texts[start_idx:end_idx]

            # Raw query term frequencies
            q_counts = self.vectorizer.transform(q_batch)

            # Query weight: query_tf * idf
            # Multiplying CSR matrix by 1D array scales each column by corresponding idf
            q_weighted = q_counts.multiply(self.idf).tocsr()

            # Dot product with BM25 document matrix: (batch_size, vocab) x (vocab, n_docs)
            batch_scores = q_weighted.dot(self.bm25_matrix.T)
            batches.append(batch_scores)

        if len(batches) == 1:
            return batches[0]
        return sp.vstack(batches, format="csr")

    def get_index_stats(self) -> Dict[str, Any]:
        """Returns metadata and statistics for the fitted BM25 index."""
        return {
            "retriever_type": "BM25Okapi",
            "is_fitted": self.is_fitted,
            "n_docs": self.n_docs,
            "vocab_size": self.vocab_size,
            "avg_doc_length": round(self.avgdl, 2),
            "k1": self.k1,
            "b": self.b,
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "index_nonzeros": int(self.bm25_matrix.nnz) if self.bm25_matrix is not None else 0,
        }

