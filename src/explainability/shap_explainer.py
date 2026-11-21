"""
SHAP Explainability for Weighted Linear Score Fusion in Legal Retrieval.

Implements exact, closed-form Shapley value calculations for linear fusion models:
    f(x) = \\sum_{i} w_i * x_i

For a background reference distribution D, the Shapley value of feature i is:
    phi_i(x) = w_i * (x_i - E_{X ~ D}[X_i])

With the base expected score:
    E[f(X)] = \\sum_{i} w_i * E_{X ~ D}[X_i]

Satisfies the efficiency property exactly:
    \\sum_{i} phi_i(x) = f(x) - E[f(X)]

Distinguishes:
- Positive contributions (phi_i > 0): signals lifting the candidate above the average candidate score.
- Negative contributions (phi_i < 0): signals pulling the candidate below the baseline.
- Global feature importance: mean absolute SHAP value E[|phi_i|].
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
import numpy as np
import pandas as pd
import shap


class LinearFusionSHAPExplainer:
    """Exact, closed-form SHAP explainer tailored for linear score fusion architectures."""

    DEFAULT_FEATURE_COLS = [
        "bm25_score",
        "tfidf_score",
        "inlegalbert_sim",
        "composite_legal_score",
    ]

    def __init__(
        self,
        weights: Sequence[float] = (0.0, 0.65, 0.25, 0.10),
        feature_cols: Optional[List[str]] = None,
        background_df: Optional[pd.DataFrame] = None,
        query_col: str = "query_id",
        cand_col: str = "candidate_id",
        n_background_samples: int = 500,
        random_state: int = 42,
    ):
        """
        Args:
            weights: Learned linear weights matching feature_cols.
            feature_cols: Feature names used in score fusion.
            background_df: DataFrame used to estimate background expected feature values.
            query_col: Name of query identifier column.
            cand_col: Name of candidate identifier column.
            n_background_samples: Number of background pairs to sample.
            random_state: Random seed for background sampling.
        """
        self.feature_cols = feature_cols or list(self.DEFAULT_FEATURE_COLS)
        self.weights = np.asarray(weights, dtype=np.float64)
        if len(self.weights) != len(self.feature_cols):
            raise ValueError("Weights length must match feature_cols.")

        self.query_col = query_col
        self.cand_col = cand_col
        self.random_state = random_state

        self.expected_values: Optional[np.ndarray] = None
        self.base_value: float = 0.0
        self.background_sample: Optional[np.ndarray] = None

        if background_df is not None:
            self.fit_background(background_df, n_samples=n_background_samples)

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies query-level min-max scaling to fusion feature columns."""
        norm_cols = [f"_qnorm_{c}" for c in self.feature_cols]
        if all(c in df.columns for c in norm_cols):
            return df

        df_out = df.copy()
        for col in self.feature_cols:
            if col not in df_out.columns:
                raise KeyError(f"Feature '{col}' missing from DataFrame.")

            col_vals = df_out[col].values
            mn = df_out.groupby(self.query_col)[col].transform("min").values
            mx = df_out.groupby(self.query_col)[col].transform("max").values
            diff = mx - mn
            safe_diff = np.where(diff > 1e-9, diff, 1.0)
            df_out[f"_qnorm_{col}"] = np.where(diff > 1e-9, (col_vals - mn) / safe_diff, 0.0)

        return df_out

    def fit_background(
        self,
        df: pd.DataFrame,
        n_samples: int = 500,
    ) -> "LinearFusionSHAPExplainer":
        """
        Fits background reference expectations from a representative sample of candidate pairs.
        """
        norm_df = self._normalize_features(df)
        norm_cols = [f"_qnorm_{c}" for c in self.feature_cols]

        # Sample background pairs representative of development queries
        if len(norm_df) > n_samples:
            sampled_df = norm_df.sample(n=n_samples, random_state=self.random_state)
        else:
            sampled_df = norm_df

        self.background_sample = sampled_df[norm_cols].values.astype(np.float64)
        self.expected_values = np.mean(self.background_sample, axis=0)
        self.base_value = float(self.expected_values @ self.weights)
        return self

    def explain(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, float]:
        """
        Computes exact closed-form SHAP values for all rows in df.

        Returns:
            Tuple of:
            - shap_values: np.ndarray of shape (N, num_features)
            - base_value: expected prediction over the background distribution
        """
        if self.expected_values is None:
            raise RuntimeError("Background not fitted. Call fit_background() first.")

        norm_df = self._normalize_features(df)
        norm_cols = [f"_qnorm_{c}" for c in self.feature_cols]
        X = norm_df[norm_cols].values.astype(np.float64)

        # Exact linear SHAP: phi_i = w_i * (X_i - E[X_i])
        shap_values = self.weights * (X - self.expected_values)
        return shap_values, self.base_value

    def explain_instance(
        self,
        query_id: str,
        candidate_id: str,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Generates a detailed local SHAP attribution explanation for a specific pair.
        """
        norm_cols = [f"_qnorm_{c}" for c in self.feature_cols]
        if not all(c in df.columns for c in norm_cols):
            # Normalize within the query's candidate pool
            query_slice = df[df[self.query_col] == query_id]
            norm_df = self._normalize_features(query_slice)
        else:
            norm_df = df

        sub = norm_df[(norm_df[self.query_col] == query_id) & (norm_df[self.cand_col] == candidate_id)]
        if len(sub) == 0:
            raise ValueError(f"Pair ({query_id}, {candidate_id}) not found in DataFrame.")

        shap_vals, base_val = self.explain(sub)
        pair_shap = shap_vals[0]
        norm_features = sub[norm_cols].iloc[0].to_dict()

        pred_score = float(base_val + np.sum(pair_shap))

        feature_explanations = []
        for i, col in enumerate(self.feature_cols):
            phi = float(pair_shap[i])
            feature_explanations.append({
                "feature": col,
                "weight": float(self.weights[i]),
                "normalized_value": round(float(norm_features[f"_qnorm_{col}"]), 4),
                "expected_value": round(float(self.expected_values[i]), 4),
                "shap_value": round(phi, 5),
                "effect": "positive" if phi > 0 else "negative" if phi < 0 else "neutral",
            })

        # Sort by absolute SHAP value
        feature_explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "query_id": query_id,
            "candidate_id": candidate_id,
            "base_value": round(base_val, 5),
            "predicted_score": round(pred_score, 5),
            "features": feature_explanations,
        }

    def get_global_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes global mean absolute SHAP feature importance: E[|phi_i|].
        """
        shap_vals, _ = self.explain(df)
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        mean_val = np.mean(shap_vals, axis=0)

        records = []
        for i, col in enumerate(self.feature_cols):
            records.append({
                "feature": col,
                "weight": self.weights[i],
                "mean_abs_shap": round(float(mean_abs[i]), 5),
                "mean_shap": round(float(mean_val[i]), 5),
            })

        df_imp = pd.DataFrame(records).sort_values(by="mean_abs_shap", ascending=False)
        return df_imp


# Alias for convenience
LinearSHAPExplainer = LinearFusionSHAPExplainer

