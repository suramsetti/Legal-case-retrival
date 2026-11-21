"""
TF-IDF Lexical Retriever for Legal Prior Case Retrieval.

Implements word-level TF-IDF representation with L2 normalization and cosine
similarity using sparse matrix operations.
"""

from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFRetriever:
    """
    TF-IDF retriever utilizing scikit-learn's TfidfVectorizer with sparse CSR
    cosine similarity calculation.
    """

    def __init__(
        self,
        ngram_range: tuple = (1, 1),
        min_df: Union[int, float] = 2,
        max_df: Union[int, float] = 0.9,
        max_features: Optional[int] = 50000,
        norm: str = "l2",
        sublinear_tf: bool = True,
        lowercase: bool = True,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.norm = norm
        self.sublinear_tf = sublinear_tf
        self.lowercase = lowercase

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            norm=self.norm,
            sublinear_tf=self.sublinear_tf,
            lowercase=self.lowercase,
        )
        self.corpus_matrix: Optional[sp.csr_matrix] = None
        self.n_docs: int = 0
        self.vocab_size: int = 0
        self.is_fitted: bool = False

    def fit(self, corpus_texts: Sequence[str]) -> "TFIDFRetriever":
        """
        Fits TF-IDF model on candidate documents or chunks and indexes sparse representations.
        
        Args:
            corpus_texts: Sequence of candidate texts.
            
        Returns:
            self
        """
        self.corpus_matrix = self.vectorizer.fit_transform(corpus_texts)
        self.n_docs, self.vocab_size = self.corpus_matrix.shape
        self.is_fitted = True
        return self

    def score_queries(
        self,
        query_texts: Sequence[str],
        batch_size: int = 256,
    ) -> sp.csr_matrix:
        """
        Computes cosine similarity between queries and indexed corpus.
        Because both corpus and queries are L2-normalized, the cosine similarity is the dot product.
        
        Args:
            query_texts: Sequence of query strings.
            batch_size: Batch size for transforming and multiplying query vectors.
            
        Returns:
            sp.csr_matrix: Cosine similarity matrix of shape (n_queries, n_docs).
        """
        if not self.is_fitted or self.corpus_matrix is None:
            raise RuntimeError("TFIDFRetriever must be fitted before scoring queries.")

        n_queries = len(query_texts)
        batches = []

        for start_idx in range(0, n_queries, batch_size):
            end_idx = min(start_idx + batch_size, n_queries)
            q_batch = query_texts[start_idx:end_idx]

            q_vectors = self.vectorizer.transform(q_batch)
            batch_scores = q_vectors.dot(self.corpus_matrix.T)
            batches.append(batch_scores)

        if len(batches) == 1:
            return batches[0]
        return sp.vstack(batches, format="csr")

    def get_index_stats(self) -> Dict[str, Any]:
        """Returns metadata and statistics for the fitted TF-IDF index."""
        return {
            "retriever_type": "TF-IDF",
            "is_fitted": self.is_fitted,
            "n_docs": self.n_docs,
            "vocab_size": self.vocab_size,
            "norm": self.norm,
            "sublinear_tf": self.sublinear_tf,
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "index_nonzeros": int(self.corpus_matrix.nnz) if self.corpus_matrix is not None else 0,
        }

