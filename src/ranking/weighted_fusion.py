"""
Weighted Linear Score Fusion for Hybrid Case Retrieval.

Combines normalized retrieval signals:
    S_hybrid(d) = w_bm25 * S_norm(bm25) + w_tfidf * S_norm(tfidf) + w_dense * S_norm(dense) + w_legal * S_norm(legal)

Subject to:
    \\sum_i w_i = 1.0,  w_i >= 0

Features support query-level min-max normalization or pre-standardized inputs.
Weight optimization is strictly performed on the validation/development split.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation.hybrid_evaluator import HybridEvaluator


class WeightedLinearFusion:
    """Weighted linear score combiner with vectorized grid search optimization."""

    DEFAULT_FEATURE_COLS = [
        "bm25_score",
        "tfidf_score",
        "inlegalbert_sim",
        "composite_legal_score",
    ]

    def __init__(
        self,
        weights: Optional[Sequence[float]] = None,
        feature_cols: Optional[List[str]] = None,
        use_query_min_max: bool = True,
    ):
        """
        Args:
            weights: Optional pre-set weights matching feature_cols.
            feature_cols: Names of the feature columns to fuse.
            use_query_min_max: Whether to apply query-level min-max normalization.
        """
        self.feature_cols = feature_cols or list(self.DEFAULT_FEATURE_COLS)
        self.use_query_min_max = use_query_min_max
        self.weights: Optional[np.ndarray] = (
            np.asarray(weights, dtype=np.float64) if weights is not None else None
        )
        if self.weights is not None:
            self._validate_weights(self.weights)

    def _validate_weights(self, weights: np.ndarray) -> None:
        if len(weights) != len(self.feature_cols):
            raise ValueError(
                f"Weights length ({len(weights)}) must match feature_cols ({len(self.feature_cols)})."
            )
        if np.any(weights < -1e-6):
            raise ValueError(f"All weights must be non-negative, got {weights}")
        total = float(np.sum(weights))
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def normalize_features(
        self,
        df: pd.DataFrame,
        query_col: str = "query_id",
    ) -> pd.DataFrame:
        """
        Applies query-level min-max normalization to the fusion feature columns.
        """
        df_out = df.copy()
        for col in self.feature_cols:
            if col not in df_out.columns:
                raise KeyError(f"Feature column '{col}' missing from DataFrame.")

            col_vals = df_out[col].values
            mn = df_out.groupby(query_col)[col].transform("min").values
            mx = df_out.groupby(query_col)[col].transform("max").values
            diff = mx - mn
            df_out[f"_qnorm_{col}"] = np.where(diff > 1e-9, (col_vals - mn) / diff, 0.0)
        return df_out

    def compute_scores(
        self,
        df: pd.DataFrame,
        weights: Optional[Sequence[float]] = None,
        query_col: str = "query_id",
    ) -> np.ndarray:
        """
        Computes composite fusion scores for all candidate rows.
        """
        w = np.asarray(weights if weights is not None else self.weights, dtype=np.float64)
        if w is None or len(w) == 0:
            raise ValueError("No weights available. Train or specify weights.")
        self._validate_weights(w)

        if self.use_query_min_max:
            norm_df = self.normalize_features(df, query_col=query_col)
            norm_cols = [f"_qnorm_{c}" for c in self.feature_cols]
            X = norm_df[norm_cols].values
        else:
            X = df[self.feature_cols].values

        return float(1.0) * (X @ w)

    def fit_grid_search(
        self,
        dev_df: pd.DataFrame,
        step: float = 0.05,
        target_metric: str = "nDCG@10",
        query_col: str = "query_id",
        label_col: str = "label",
    ) -> Dict[str, Union[List[float], float, Dict[str, float]]]:
        """
        Fast vectorized grid search over non-negative weight combinations summing to 1.0.
        Optimized strictly on the development split.
        """
        # Pre-normalize features
        if self.use_query_min_max:
            norm_df = self.normalize_features(dev_df, query_col=query_col)
            norm_cols = [f"_qnorm_{c}" for c in self.feature_cols]
        else:
            norm_df = dev_df
            norm_cols = self.feature_cols

        # Group data into query blocks for fast vectorized score computation
        query_blocks = []
        discounts_10 = 1.0 / np.log2(np.arange(2, 12))

        for _, group in norm_df.groupby(query_col, sort=False):
            s_mat = group[norm_cols].values.astype(np.float64)
            lbls = group[label_col].values.astype(int)
            pos = int(np.sum(lbls))
            if pos == 0:
                continue
            r_ideal = [1] * min(pos, 10) + [0] * max(0, 10 - pos)
            idcg_10 = float(np.sum(r_ideal / np.log2(np.arange(2, len(r_ideal) + 2))))
            query_blocks.append((s_mat, lbls, idcg_10, pos))

        if not query_blocks:
            raise ValueError("No valid queries with positives found in dev_df.")

        # Generate weight grid
        combos = self._generate_weight_grid(n_features=len(self.feature_cols), step=step)

        best_score = -1.0
        best_w = combos[0]

        for w in combos:
            ndcg10_list = []
            for s_mat, lbls, idcg10, pos in query_blocks:
                scores = s_mat @ w
                top_k = min(10, len(scores))
                top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
                top_idx = top_idx[np.argsort(-scores[top_idx])]
                top_lbls = lbls[top_idx]
                dcg = float(np.sum(top_lbls * discounts_10[:len(top_lbls)]))
                ndcg10_list.append(dcg / idcg10 if idcg10 > 0 else 0.0)

            mean_val = float(np.mean(ndcg10_list))
            if mean_val > best_score:
                best_score = mean_val
                best_w = w

        self.weights = best_w

        # Run full evaluation with best weights
        eval_df = dev_df.copy()
        eval_df["_fusion_score"] = self.compute_scores(eval_df, query_col=query_col)
        full_metrics = HybridEvaluator.evaluate_predictions(
            eval_df,
            score_col="_fusion_score",
            query_col=query_col,
            label_col=label_col,
        )

        return {
            "best_weights": [round(float(x), 4) for x in best_w],
            "feature_names": self.feature_cols,
            "target_metric": target_metric,
            "best_score": round(best_score, 4),
            "full_metrics": full_metrics,
        }

    @staticmethod
    def _generate_weight_grid(n_features: int, step: float = 0.05) -> List[np.ndarray]:
        """Recursively generates weight combinations summing to 1.0."""
        def _recurse(remaining: float, depth: int) -> List[List[float]]:
            if depth == 1:
                return [[round(remaining, 4)]]
            results = []
            ticks = int(round(remaining / step))
            for i in range(ticks + 1):
                val = round(i * step, 4)
                sub_results = _recurse(remaining - val, depth - 1)
                for sub in sub_results:
                    results.append([val] + sub)
            return results

        combos = _recurse(1.0, n_features)
        return [np.array(c, dtype=np.float64) for c in combos]

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves learned weights and feature configuration to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_cols": self.feature_cols,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "use_query_min_max": self.use_query_min_max,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "WeightedLinearFusion":
        """Loads model from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            weights=data["weights"],
            feature_cols=data["feature_cols"],
            use_query_min_max=data["use_query_min_max"],
        )

