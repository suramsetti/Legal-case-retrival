"""
LightGBM LambdaMART Ranker for Hybrid Case Retrieval.

Implements listwise learning-to-rank using LightGBM's LambdaRank algorithm:
- Query-grouped training enforcing contiguous query blocks without pair shuffling.
- Evaluated on unconstrained development candidate pools.
- Feature importance analysis (split counts and information gain).
- Model persistence to text/binary checkpoints.
"""

from typing import Dict, List, Optional, Sequence, Union
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.evaluation.hybrid_evaluator import HybridEvaluator


class LightGBMRanker:
    """Wrapper for LightGBM LambdaMART ranking model."""

    DEFAULT_FEATURE_COLS = [
        "norm_statute_shared_count",
        "norm_citation_shared_count",
        "norm_entity_shared_count",
        "norm_bm25_rank",
        "norm_tfidf_rank",
        "norm_inlegalbert_rank",
        "norm_bm25_score",
        "norm_tfidf_score",
        "norm_inlegalbert_sim",
        "norm_statute_jaccard",
        "norm_statute_containment_query",
        "norm_citation_jaccard",
        "norm_entity_jaccard",
        "norm_judge_overlap",
        "norm_court_overlap",
        "norm_org_overlap",
        "norm_rhetorical_cosine",
        "norm_rhetorical_js_sim",
        "norm_rhetorical_ratio_diff",
        "norm_composite_legal_score",
    ]

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        random_state: int = 42,
        importance_type: str = "gain",
    ):
        self.feature_cols = feature_cols or list(self.DEFAULT_FEATURE_COLS)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.importance_type = importance_type

        self.model: Optional[lgb.LGBMRanker] = None
        self.booster: Optional[lgb.Booster] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
        query_col: str = "query_id",
        label_col: str = "label",
        early_stopping_rounds: Optional[int] = 20,
    ) -> "LightGBMRanker":
        """
        Trains LambdaMART ranker ensuring contiguous query blocks.
        """
        # Sort train_df by query_id to ensure strictly contiguous groups
        df_train_sorted = train_df.sort_values(query_col).copy()
        train_groups = df_train_sorted.groupby(query_col, sort=False).size().to_numpy()
        X_train = df_train_sorted[self.feature_cols].values
        y_train = df_train_sorted[label_col].values.astype(int)

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[5, 10],
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            importance_type=self.importance_type,
            verbose=-1,
        )

        callbacks = []
        if eval_df is not None and early_stopping_rounds is not None and early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
            df_eval_sorted = eval_df.sort_values(query_col).copy()
            eval_groups = df_eval_sorted.groupby(query_col, sort=False).size().to_numpy()
            X_eval = df_eval_sorted[self.feature_cols].values
            y_eval = df_eval_sorted[label_col].values.astype(int)

            self.model.fit(
                X_train,
                y_train,
                group=train_groups,
                eval_set=[(X_eval, y_eval)],
                eval_group=[eval_groups],
                callbacks=callbacks,
            )
        else:
            self.model.fit(X_train, y_train, group=train_groups)

        self.booster = self.model.booster_
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts ranking scores for candidate documents in df."""
        X = df[self.feature_cols].values
        if self.booster is not None:
            return self.booster.predict(X)
        if self.model is not None and hasattr(self.model, "_Booster") and self.model._Booster is not None:
            return self.model._Booster.predict(X)
        if self.model is not None:
            return self.model.predict(X)
        raise RuntimeError("Model is not fitted or loaded. Call fit() or load_model() first.")

    def evaluate(
        self,
        df: pd.DataFrame,
        query_col: str = "query_id",
        label_col: str = "label",
    ) -> Dict[str, float]:
        """Generates predictions on df and evaluates IR metrics."""
        scores = self.predict(df)
        df_eval = df.copy()
        df_eval["_lgbm_score"] = scores
        return HybridEvaluator.evaluate_predictions(
            df_eval,
            score_col="_lgbm_score",
            query_col=query_col,
            label_col=label_col,
        )

    def get_feature_importances(self) -> pd.Series:
        """Returns sorted feature importances."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        importances = self.model.feature_importances_
        return pd.Series(importances, index=self.feature_cols).sort_values(ascending=False)

    def save_model(self, filepath: Union[str, Path]) -> None:
        """Saves model to text checkpoint file."""
        if self.model is None or self.model.booster_ is None:
            raise RuntimeError("No booster available to save.")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(filepath))

    def load_model(self, filepath: Union[str, Path]) -> "LightGBMRanker":
        """Loads model booster from text file."""
        self.booster = lgb.Booster(model_file=str(filepath))
        return self

