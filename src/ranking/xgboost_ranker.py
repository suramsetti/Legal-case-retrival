"""
XGBoost Ranker for Hybrid Case Retrieval.

Implements pairwise and listwise gradient-boosted ranking:
- Supports 'rank:ndcg' and 'rank:pairwise' objectives.
- Enforces strict query grouping across partitions.
- Saves model checkpoints in native JSON format.
"""

from typing import Dict, List, Optional, Sequence, Union
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from src.evaluation.hybrid_evaluator import HybridEvaluator


class XGBoostRanker:
    """Wrapper for XGBoost Ranking model."""

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
        objective: str = "rank:pairwise",
        eval_metric: str = "ndcg@10",
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        random_state: int = 42,
    ):
        self.feature_cols = feature_cols or list(self.DEFAULT_FEATURE_COLS)
        self.objective = objective
        self.eval_metric = eval_metric
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        self.model: Optional[xgb.XGBRanker] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
        query_col: str = "query_id",
        label_col: str = "label",
        early_stopping_rounds: Optional[int] = 20,
    ) -> "XGBoostRanker":
        """
        Trains XGBoost ranker with query grouping.
        """
        df_train_sorted = train_df.sort_values(query_col).copy()
        train_groups = df_train_sorted.groupby(query_col, sort=False).size().to_numpy()
        X_train = df_train_sorted[self.feature_cols].values
        y_train = df_train_sorted[label_col].values.astype(int)

        self.model = xgb.XGBRanker(
            objective=self.objective,
            eval_metric=self.eval_metric,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

        if eval_df is not None and early_stopping_rounds is not None and early_stopping_rounds > 0:
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
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train, group=train_groups, verbose=False)

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts ranking scores for candidate documents."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        X = df[self.feature_cols].values
        return self.model.predict(X)

    def evaluate(
        self,
        df: pd.DataFrame,
        query_col: str = "query_id",
        label_col: str = "label",
    ) -> Dict[str, float]:
        """Evaluates ranking metrics on df."""
        scores = self.predict(df)
        df_eval = df.copy()
        df_eval["_xgb_score"] = scores
        return HybridEvaluator.evaluate_predictions(
            df_eval,
            score_col="_xgb_score",
            query_col=query_col,
            label_col=label_col,
        )

    def get_feature_importances(self) -> pd.Series:
        """Returns sorted feature importances."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        return pd.Series(
            self.model.feature_importances_, index=self.feature_cols
        ).sort_values(ascending=False)

    def save_model(self, filepath: Union[str, Path]) -> None:
        """Saves model to JSON checkpoint file."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(filepath))

    def load_model(self, filepath: Union[str, Path]) -> "XGBoostRanker":
        """Loads model from JSON file."""
        self.model = xgb.XGBRanker()
        self.model.load_model(str(filepath))
        return self

