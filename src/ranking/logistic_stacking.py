"""
Leakage-Safe Logistic Stacking Meta-Learner for Hybrid Legal Retrieval.

Implements cross-validated, group-aware stacking:
1. Partitions queries into K folds using GroupKFold on the training split.
2. For each fold, trains modality-specific base rankers:
   - Lexical Base Ranker (BM25 + TF-IDF scores and ranks)
   - Dense Base Ranker (InLegalBERT semantic similarity and rank)
   - Legal Knowledge Base Ranker (Citations, statutes, entities, rhetorical similarity)
3. Generates Out-Of-Fold (OOF) predictions to eliminate target and query leakage.
4. Fits a Logistic Regression meta-learner on the OOF prediction matrix.
5. Employs bagged fold-averaged base estimators for inference on validation/test sets.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from src.evaluation.hybrid_evaluator import HybridEvaluator


class LeakageSafeLogisticStacking:
    """Leakage-safe stacking meta-learner using GroupKFold base models."""

    DEFAULT_LEXICAL_COLS = [
        "norm_bm25_score",
        "norm_bm25_rank",
        "norm_tfidf_score",
        "norm_tfidf_rank",
    ]

    DEFAULT_DENSE_COLS = [
        "norm_inlegalbert_sim",
        "norm_inlegalbert_rank",
    ]

    DEFAULT_LEGAL_COLS = [
        "norm_statute_shared_count",
        "norm_citation_shared_count",
        "norm_entity_shared_count",
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
        n_splits: int = 5,
        lexical_cols: Optional[List[str]] = None,
        dense_cols: Optional[List[str]] = None,
        legal_cols: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.lexical_cols = lexical_cols or list(self.DEFAULT_LEXICAL_COLS)
        self.dense_cols = dense_cols or list(self.DEFAULT_DENSE_COLS)
        self.legal_cols = legal_cols or list(self.DEFAULT_LEGAL_COLS)
        self.random_state = random_state

        self.fold_models: List[Dict[str, LogisticRegression]] = []
        self.meta_model: Optional[LogisticRegression] = None
        self.oof_predictions: Optional[np.ndarray] = None
        self.meta_coefficients: Optional[Dict[str, float]] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        query_col: str = "query_id",
        label_col: str = "label",
    ) -> "LeakageSafeLogisticStacking":
        """
        Fits base rankers via GroupKFold and trains the meta-learner on OOF predictions.
        """
        df = train_df.sort_values(query_col).copy()
        y = df[label_col].values.astype(int)
        groups = df[query_col].values

        gkf = GroupKFold(n_splits=self.n_splits)

        # 3 columns: 0=lexical, 1=dense, 2=legal
        oof_preds = np.zeros((len(df), 3), dtype=np.float64)
        self.fold_models = []

        for fold, (trn_idx, val_idx) in enumerate(gkf.split(df, y, groups=groups)):
            fold_dict = {}

            # Base model 1: Lexical
            m_lex = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=self.random_state)
            m_lex.fit(df.iloc[trn_idx][self.lexical_cols], y[trn_idx])
            oof_preds[val_idx, 0] = m_lex.predict_proba(df.iloc[val_idx][self.lexical_cols])[:, 1]
            fold_dict["lexical"] = m_lex

            # Base model 2: Dense Semantic
            m_dense = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=self.random_state)
            m_dense.fit(df.iloc[trn_idx][self.dense_cols], y[trn_idx])
            oof_preds[val_idx, 1] = m_dense.predict_proba(df.iloc[val_idx][self.dense_cols])[:, 1]
            fold_dict["dense"] = m_dense

            # Base model 3: Legal Knowledge
            m_legal = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=self.random_state)
            m_legal.fit(df.iloc[trn_idx][self.legal_cols], y[trn_idx])
            oof_preds[val_idx, 2] = m_legal.predict_proba(df.iloc[val_idx][self.legal_cols])[:, 1]
            fold_dict["legal"] = m_legal

            self.fold_models.append(fold_dict)

        self.oof_predictions = oof_preds

        # Fit Meta-Learner on Out-Of-Fold predictions
        self.meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state)
        self.meta_model.fit(oof_preds, y)

        coefs = self.meta_model.coef_[0]
        self.meta_coefficients = {
            "lexical_weight": round(float(coefs[0]), 4),
            "dense_weight": round(float(coefs[1]), 4),
            "legal_weight": round(float(coefs[2]), 4),
            "intercept": round(float(self.meta_model.intercept_[0]), 4),
        }

        return self

    def predict_base_probabilities(self, df: pd.DataFrame) -> np.ndarray:
        """
        Computes bagged (averaged over K folds) base model probabilities.
        Returns shape (N, 3): [P_lexical, P_dense, P_legal].
        """
        if not self.fold_models:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        base_preds = np.zeros((len(df), 3), dtype=np.float64)

        for fold_dict in self.fold_models:
            p_lex = fold_dict["lexical"].predict_proba(df[self.lexical_cols])[:, 1]
            p_dense = fold_dict["dense"].predict_proba(df[self.dense_cols])[:, 1]
            p_legal = fold_dict["legal"].predict_proba(df[self.legal_cols])[:, 1]

            base_preds[:, 0] += p_lex / self.n_splits
            base_preds[:, 1] += p_dense / self.n_splits
            base_preds[:, 2] += p_legal / self.n_splits

        return base_preds

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts calibrated hybrid ranking probabilities using the meta-learner."""
        if self.meta_model is None:
            raise RuntimeError("Meta-model is not fitted.")
        base_preds = self.predict_base_probabilities(df)
        return self.meta_model.predict_proba(base_preds)[:, 1]

    def evaluate(
        self,
        df: pd.DataFrame,
        query_col: str = "query_id",
        label_col: str = "label",
    ) -> Dict[str, float]:
        """Evaluates hybrid ranking metrics on df."""
        scores = self.predict(df)
        df_eval = df.copy()
        df_eval["_stacking_score"] = scores
        return HybridEvaluator.evaluate_predictions(
            df_eval,
            score_col="_stacking_score",
            query_col=query_col,
            label_col=label_col,
        )

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves fitted stacking model to joblib checkpoint."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_splits": self.n_splits,
            "lexical_cols": self.lexical_cols,
            "dense_cols": self.dense_cols,
            "legal_cols": self.legal_cols,
            "fold_models": self.fold_models,
            "meta_model": self.meta_model,
            "meta_coefficients": self.meta_coefficients,
        }
        joblib.dump(payload, str(filepath))

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LeakageSafeLogisticStacking":
        """Loads fitted stacking model from joblib file."""
        payload = joblib.load(str(filepath))
        instance = cls(
            n_splits=payload["n_splits"],
            lexical_cols=payload["lexical_cols"],
            dense_cols=payload["dense_cols"],
            legal_cols=payload["legal_cols"],
        )
        instance.fold_models = payload["fold_models"]
        instance.meta_model = payload["meta_model"]
        instance.meta_coefficients = payload["meta_coefficients"]
        return instance

