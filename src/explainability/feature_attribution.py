"""
Mathematical Feature Attribution and Score Decomposition for Hybrid Legal Retrieval.

Deconstructs the exact mathematical contribution of each retrieval modality and legal feature
for the validated Weighted Linear Fusion model:
    S_hybrid(d) = w_bm25 * S_norm(bm25) + w_tfidf * S_norm(tfidf) + w_dense * S_norm(dense) + w_legal * S_norm(legal)

Guarantees the exact additive score identity:
    \\sum_{m} Contribution(m, d) = S_hybrid(d)
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
import numpy as np
import pandas as pd


class FeatureAttributor:
    """Computes exact mathematical score attributions and modality decompositions."""

    DEFAULT_FEATURE_COLS = [
        "bm25_score",
        "tfidf_score",
        "inlegalbert_sim",
        "composite_legal_score",
    ]

    LEGAL_SUB_WEIGHTS = {
        "statute_jaccard": 0.35,
        "citation_jaccard": 0.35,
        "entity_jaccard": 0.15,
        "rhetorical_cosine": 0.15,
    }

    def __init__(
        self,
        weights: Sequence[float] = (0.0, 0.65, 0.25, 0.10),
        feature_cols: Optional[List[str]] = None,
        query_col: str = "query_id",
        cand_col: str = "candidate_id",
        feature_weights: Optional[Dict[str, float]] = None,
        use_query_min_max: bool = True,
    ):
        """
        Args:
            weights: Learned fusion weights matching feature_cols.
            feature_cols: Names of features used in linear fusion.
            query_col: Column name identifying queries.
            cand_col: Column name identifying candidate documents.
            feature_weights: Optional dictionary of feature_name -> weight.
            use_query_min_max: Kept for API compatibility.
        """
        if feature_weights is not None:
            self.feature_cols = list(feature_weights.keys())
            self.weights = np.asarray(list(feature_weights.values()), dtype=np.float64)
        else:
            self.feature_cols = feature_cols or list(self.DEFAULT_FEATURE_COLS)
            self.weights = np.asarray(weights, dtype=np.float64)

        if len(self.weights) != len(self.feature_cols):
            raise ValueError("Weights length must match feature_cols.")
        if not np.isclose(np.sum(self.weights), 1.0, atol=1e-3):
            raise ValueError(f"Weights must sum to 1.0, got {np.sum(self.weights)}")

        self.query_col = query_col
        self.cand_col = cand_col

    def compute_query_normalized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes query-level min-max normalized features: (x - min) / (max - min)."""
        df_out = df.copy()
        for col in self.feature_cols:
            if col not in df_out.columns:
                raise KeyError(f"Feature column '{col}' missing from DataFrame.")

            col_vals = df_out[col].values
            mn = df_out.groupby(self.query_col)[col].transform("min").values
            mx = df_out.groupby(self.query_col)[col].transform("max").values
            diff = mx - mn
            safe_diff = np.where(diff > 1e-9, diff, 1.0)
            df_out[f"_qnorm_{col}"] = np.where(diff > 1e-9, (col_vals - mn) / safe_diff, 0.0)

        return df_out

    def compute_attributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates exact score contributions for each modality and legal sub-feature.

        Returns:
            pd.DataFrame with added columns:
            - '_qnorm_{col}': normalized feature score in [0, 1]
            - 'contrib_{col}': w_i * _qnorm_{col}
            - 'contrib_legal_{subfeature}': broken-down legal knowledge sub-contribution
            - 'hybrid_score': exact sum of contributions
        """
        df_norm = self.compute_query_normalized_features(df)

        # 1. Modality contributions
        hybrid_scores = np.zeros(len(df_norm), dtype=np.float64)
        for w, col in zip(self.weights, self.feature_cols):
            norm_col = f"_qnorm_{col}"
            contrib_col = f"contrib_{col}"
            df_norm[contrib_col] = w * df_norm[norm_col].values
            hybrid_scores += df_norm[contrib_col].values

        df_norm["hybrid_score"] = hybrid_scores
        df_norm["predicted_hybrid_score"] = hybrid_scores

        # 2. Deconstruct Legal Knowledge into sub-features if present
        legal_weight = 0.0
        for w, col in zip(self.weights, self.feature_cols):
            if col == "composite_legal_score":
                legal_weight = w

        for sub_col, sub_weight in self.LEGAL_SUB_WEIGHTS.items():
            if sub_col in df_norm.columns:
                # Sub-attribution: legal_weight * normalized_subfeature_score * sub_weight
                # Query-normalized subfeature
                col_vals = df_norm[sub_col].values
                mn = df_norm.groupby(self.query_col)[sub_col].transform("min").values
                mx = df_norm.groupby(self.query_col)[sub_col].transform("max").values
                diff = mx - mn
                safe_diff = np.where(diff > 1e-9, diff, 1.0)
                norm_sub = np.where(diff > 1e-9, (col_vals - mn) / safe_diff, 0.0)
                df_norm[f"contrib_legal_{sub_col}"] = legal_weight * sub_weight * norm_sub

        return df_norm

    def get_global_attribution_summary(self, df_attributed: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes corpus-wide global feature attribution statistics across all pairs and top ranks.
        """
        # Filter top-k per query for ranking-specific analysis
        df_sorted = df_attributed.sort_values(
            by=[self.query_col, "hybrid_score"], ascending=[True, False]
        )
        df_sorted["rank"] = df_sorted.groupby(self.query_col).cumcount() + 1

        top1_mask = df_sorted["rank"] == 1
        top5_mask = df_sorted["rank"] <= 5
        top10_mask = df_sorted["rank"] <= 10
        pos_mask = df_sorted["label"] == 1 if "label" in df_sorted.columns else np.ones(len(df_sorted), dtype=bool)

        summary: Dict[str, Any] = {
            "all_pairs": {},
            "top1_candidates": {},
            "top5_candidates": {},
            "top10_candidates": {},
            "true_positives": {},
        }

        slices = [
            ("all_pairs", np.ones(len(df_sorted), dtype=bool)),
            ("top1_candidates", top1_mask),
            ("top5_candidates", top5_mask),
            ("top10_candidates", top10_mask),
            ("true_positives", pos_mask),
        ]

        for slice_name, mask in slices:
            sub = df_sorted[mask]
            slice_stats = {}
            for col in self.feature_cols:
                contrib_vals = sub[f"contrib_{col}"].values
                norm_vals = sub[f"_qnorm_{col}"].values
                slice_stats[col] = {
                    "mean_contribution": round(float(np.mean(contrib_vals)), 5),
                    "std_contribution": round(float(np.std(contrib_vals)), 5),
                    "mean_normalized_score": round(float(np.mean(norm_vals)), 5),
                    "percentage_of_score": round(
                        float(np.mean(contrib_vals) / (np.mean(sub["hybrid_score"].values) + 1e-9) * 100), 2
                    ),
                }

            # Legal sub-features
            legal_sub_stats = {}
            for sub_col in self.LEGAL_SUB_WEIGHTS.keys():
                col_name = f"contrib_legal_{sub_col}"
                if col_name in sub.columns:
                    legal_sub_stats[sub_col] = {
                        "mean_sub_contribution": round(float(np.mean(sub[col_name].values)), 5),
                    }
            slice_stats["legal_sub_contributions"] = legal_sub_stats
            summary[slice_name] = slice_stats

        return summary
