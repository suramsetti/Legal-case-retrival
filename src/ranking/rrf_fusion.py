"""
Reciprocal Rank Fusion (RRF) for Hybrid Case Retrieval.

Combines multiple retrieval rankings using rank reciprocals:
    RRF(d) = \\sum_{m \\in M} \\frac{1}{k + rank_m(d)}

Where:
- M is the set of rankers (e.g. BM25, TF-IDF, InLegalBERT).
- rank_m(d) is the 1-indexed rank of candidate d for a given query under ranker m.
- k is a smoothing hyperparameter (standard default k=60).
"""

from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from src.evaluation.hybrid_evaluator import HybridEvaluator


class RRFFusion:
    """Implements Reciprocal Rank Fusion across multiple retrieval rankers."""

    def __init__(
        self,
        k: int = 60,
        rank_cols: Optional[List[str]] = None,
        score_cols: Optional[List[str]] = None,
    ):
        """
        Args:
            k: Smoothing constant (default 60).
            rank_cols: Existing 1-indexed rank columns in DataFrame (e.g. ['bm25_rank', ...]).
            score_cols: Score columns to dynamically rank per query (if rank_cols not provided).
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if rank_cols is None and score_cols is None:
            raise ValueError("Must provide either rank_cols or score_cols.")

        self.k = k
        self.rank_cols = rank_cols
        self.score_cols = score_cols

    def compute_scores(
        self,
        df: pd.DataFrame,
        query_col: str = "query_id",
        k: Optional[int] = None,
    ) -> np.ndarray:
        """
        Computes RRF composite scores for all rows in df.

        Args:
            df: DataFrame containing query and ranking columns.
            query_col: Name of query identifier column.
            k: Optional override for smoothing parameter k.

        Returns:
            np.ndarray of composite RRF scores (higher is more relevant).
        """
        effective_k = self.k if k is None else k
        if effective_k <= 0:
            raise ValueError(f"k must be positive, got {effective_k}")

        rrf_scores = np.zeros(len(df), dtype=np.float64)

        if self.rank_cols is not None:
            for col in self.rank_cols:
                if col not in df.columns:
                    raise KeyError(f"Rank column '{col}' not found in DataFrame.")
                ranks = np.maximum(df[col].values.astype(np.float64), 1.0)
                rrf_scores += 1.0 / (effective_k + ranks)
        else:
            assert self.score_cols is not None
            for col in self.score_cols:
                if col not in df.columns:
                    raise KeyError(f"Score column '{col}' not found in DataFrame.")
                # Compute rank per query descending (rank 1 = highest score)
                ranks = df.groupby(query_col)[col].rank(ascending=False, method="min").values
                ranks = np.maximum(ranks.astype(np.float64), 1.0)
                rrf_scores += 1.0 / (effective_k + ranks)

        return rrf_scores

    @classmethod
    def evaluate_k_grid(
        cls,
        df: pd.DataFrame,
        k_values: Sequence[int] = (20, 60, 100),
        rank_cols: Optional[List[str]] = None,
        score_cols: Optional[List[str]] = None,
        query_col: str = "query_id",
        label_col: str = "label",
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluates RRF across multiple k values on an evaluation DataFrame.

        Args:
            df: Evaluation DataFrame.
            k_values: Sequence of k values to evaluate (e.g. [20, 60, 100]).
            rank_cols: Optional rank columns to use.
            score_cols: Optional score columns to use.
            query_col: Query identifier column.
            label_col: Binary label column.

        Returns:
            Dictionary mapping each k to its evaluation metrics dict.
        """
        eval_results: Dict[int, Dict[str, float]] = {}
        df_eval = df.copy()

        for k in k_values:
            fusion = cls(k=k, rank_cols=rank_cols, score_cols=score_cols)
            score_col_name = f"_rrf_k_{k}"
            df_eval[score_col_name] = fusion.compute_scores(df_eval, query_col=query_col, k=k)
            metrics = HybridEvaluator.evaluate_predictions(
                df_eval,
                score_col=score_col_name,
                query_col=query_col,
                label_col=label_col,
            )
            eval_results[k] = metrics
            df_eval.drop(columns=[score_col_name], inplace=True)

        return eval_results

