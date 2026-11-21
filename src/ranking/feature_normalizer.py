"""
Feature Normalizer and Preprocessing Pipeline for Hybrid Legal Ranking.

Applies reproducible, leakage-free feature transformations:
- Strictly fits normalization parameters on the training partition only.
- Reuses fitted parameters for dev and test partitions without leakage.
- Handles skewed count features via log1p transformation.
- Converts raw rank features into reciprocal rank (1 / rank).
- Preserves raw feature values with a 'raw_' prefix alongside normalized features.
- Validates missing-value counts and records statistical parameters.
"""

from typing import Dict, List, Set, Sequence, Tuple, Any, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd


# Explicitly defined feature groups
COUNT_FEATURES = [
    "statute_shared_count",
    "citation_shared_count",
    "entity_shared_count",
]

RANK_FEATURES = [
    "bm25_rank",
    "tfidf_rank",
    "inlegalbert_rank",
]

CONTINUOUS_FEATURES = [
    "bm25_score",
    "tfidf_score",
    "inlegalbert_sim",
    "statute_jaccard",
    "statute_containment_query",
    "citation_jaccard",
    "entity_jaccard",
    "judge_overlap",
    "court_overlap",
    "org_overlap",
    "rhetorical_cosine",
    "rhetorical_js_sim",
    "rhetorical_ratio_diff",
    "composite_legal_score",
]

ALL_FEATURE_COLUMNS = COUNT_FEATURES + RANK_FEATURES + CONTINUOUS_FEATURES


class FeatureNormalizer:
    """Preprocesses and standardizes hybrid legal retrieval features with zero split leakage."""

    def __init__(self):
        self.fitted: bool = False
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self.missing_value_report: Dict[str, int] = {}

    def fit(self, train_df: pd.DataFrame) -> "FeatureNormalizer":
        """
        Fits normalization parameters (mean, std, min, max) strictly on the training partition.
        
        Args:
            train_df: DataFrame of pairwise features from the training partition.
            
        Returns:
            self
        """
        self.feature_stats = {}
        self.missing_value_report = {}

        # 1. Transform counts and ranks in training copy for statistical estimation
        work_df = train_df.copy()

        for col in ALL_FEATURE_COLUMNS:
            if col not in work_df.columns:
                # Document missing columns
                self.missing_value_report[col] = len(work_df)
                work_df[col] = 0.0
            else:
                n_miss = int(work_df[col].isna().sum())
                self.missing_value_report[col] = n_miss
                work_df[col] = work_df[col].fillna(0.0)

        # Apply log1p on counts
        for col in COUNT_FEATURES:
            work_df[col] = np.log1p(np.maximum(work_df[col].values, 0.0))

        # Apply reciprocal rank on ranks
        for col in RANK_FEATURES:
            work_df[col] = 1.0 / np.maximum(work_df[col].values, 1.0)

        # Compute mean and standard deviation on transformed values
        for col in ALL_FEATURE_COLUMNS:
            vals = work_df[col].values.astype(np.float64)
            mean_val = float(np.mean(vals))
            # Use sample standard deviation (ddof=1) if n > 1
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else float(np.std(vals))
            if std_val < 1e-7:
                std_val = 1.0

            min_val = float(np.min(vals))
            max_val = float(np.max(vals))

            self.feature_stats[col] = {
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
            }

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms and normalizes a pairwise feature DataFrame using fitted training statistics.
        Preserves raw features as 'raw_{col}' alongside transformed 'norm_{col}'.
        
        Args:
            df: DataFrame to transform (train, dev, or test).
            
        Returns:
            pd.DataFrame containing metadata, raw features, and normalized features.
        """
        if not self.fitted:
            raise RuntimeError("FeatureNormalizer must be fitted on training data before transform.")

        out_df = df.copy()

        # 1. Preserve raw feature columns
        for col in ALL_FEATURE_COLUMNS:
            if col in out_df.columns:
                out_df[f"raw_{col}"] = out_df[col].fillna(0.0)
            else:
                out_df[f"raw_{col}"] = 0.0

        # 2. Transform and standardize
        for col in ALL_FEATURE_COLUMNS:
            raw_vals = out_df[f"raw_{col}"].values.astype(np.float64)

            # Apply domain transforms
            if col in COUNT_FEATURES:
                trans_vals = np.log1p(np.maximum(raw_vals, 0.0))
            elif col in RANK_FEATURES:
                trans_vals = 1.0 / np.maximum(raw_vals, 1.0)
            else:
                trans_vals = raw_vals

            # Standardize using fitted training stats
            stats = self.feature_stats[col]
            norm_vals = (trans_vals - stats["mean"]) / stats["std"]

            out_df[f"norm_{col}"] = norm_vals.astype(np.float32)

        return out_df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fits on training data and returns transformed training DataFrame."""
        return self.fit(train_df).transform(train_df)

    def save_stats(self, filepath: Path) -> None:
        """Saves fitted statistical parameters to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "feature_stats": self.feature_stats,
            "missing_value_report": self.missing_value_report,
            "count_features": COUNT_FEATURES,
            "rank_features": RANK_FEATURES,
            "continuous_features": CONTINUOUS_FEATURES,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_stats(self, filepath: Path) -> "FeatureNormalizer":
        """Loads fitted statistical parameters from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.feature_stats = data["feature_stats"]
        self.missing_value_report = data.get("missing_value_report", {})
        self.fitted = True
        return self
