"""
Phase 20: Multi-Class Legal Outcome Prediction Models.

Classes:
0: 'Accepted'
1: 'Rejected'
2: 'Other'

Implements:
1. MulticlassTfidfClassifier: Word unigram/bigram TF-IDF with multinomial Logistic Regression.
2. MulticlassLinearSVM: Calibrated One-vs-Rest Linear SVM.
3. MulticlassCharNgramClassifier: Sub-word char 3-5 n-grams with multinomial Logistic Regression.
4. MulticlassSectionAwareClassifier: Multi-view rhetorical section TF-IDF with section weighting.
5. MulticlassCitationAwareClassifier: Lexical text + 15 dense statutory/citation features.
6. MulticlassSectionTransformer: InLegalBERT section embeddings with additive attention pooling.
7. MulticlassHierarchicalModel: InLegalBERT section embeddings -> BiLSTM -> Self-Attention -> 3-class head.
8. MulticlassHybridFusion: Late probability fusion ensemble across diverse representations.
"""

import os
import sys
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from src.preprocessing.phase19_section_segmenter import LegalSectionSegmenter
from src.features.phase19_legal_features import LegalFeatureExtractor


CANONICAL_CLASSES = ["Accepted", "Other", "Rejected"]


class MulticlassTfidfClassifier(BaseEstimator, ClassifierMixin):
    """
    Multinomial Logistic Regression over word n-grams (1-2) with class weighting options.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 25000,
        C: float = 1.0,
        class_weight: Optional[str] = None,
        random_state: int = 42,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
            min_df=3,
        )
        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=1000,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.classes_ = np.array(CANONICAL_CLASSES)

    def fit(self, texts: List[str], y: np.ndarray):
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, y)
        self.classes_ = self.classifier.classes_
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def predict(self, texts: List[str]) -> np.ndarray:
        probs = self.predict_proba(texts)
        preds_idx = np.argmax(probs, axis=1)
        return self.classes_[preds_idx]


class MulticlassLinearSVM(BaseEstimator, ClassifierMixin):
    """
    Calibrated One-vs-Rest Linear SVM for 3-class legal judgment outcome prediction.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 25000,
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.C = C
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
            min_df=3,
        )
        base_svm = LinearSVC(C=self.C, random_state=self.random_state, max_iter=2000)
        self.classifier = CalibratedClassifierCV(base_svm, cv=3)
        self.classes_ = np.array(CANONICAL_CLASSES)

    def fit(self, texts: List[str], y: np.ndarray):
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, y)
        self.classes_ = self.classifier.classes_
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def predict(self, texts: List[str]) -> np.ndarray:
        probs = self.predict_proba(texts)
        preds_idx = np.argmax(probs, axis=1)
        return self.classes_[preds_idx]


class MulticlassCharNgramClassifier(BaseEstimator, ClassifierMixin):
    """
    Sub-word character n-gram classifier (char 3-5) with multinomial Logistic Regression.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 5),
        max_features: int = 25000,
        C: float = 1.0,
        class_weight: Optional[str] = None,
        random_state: int = 42,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
            min_df=5,
        )
        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=1000,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.classes_ = np.array(CANONICAL_CLASSES)

    def fit(self, texts: List[str], y: np.ndarray):
        self.vectorizer.min_df = min(5, len(texts))
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, y)
        self.classes_ = self.classifier.classes_
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def predict(self, texts: List[str]) -> np.ndarray:
        probs = self.predict_proba(texts)
        preds_idx = np.argmax(probs, axis=1)
        return self.classes_[preds_idx]


class MulticlassSectionAwareClassifier(BaseEstimator, ClassifierMixin):
    """
    Section-aware multi-view TF-IDF classifier for 3-class prediction.
    """

    DEFAULT_SECTION_WEIGHTS = {
        "court_reasoning": 1.5,
        "legal_provisions": 1.2,
        "facts": 1.0,
        "arguments": 0.8,
        "procedural_history": 0.5,
    }

    def __init__(
        self,
        sections: Optional[List[str]] = None,
        section_weights: Optional[Dict[str, float]] = None,
        max_features_per_section: int = 10000,
        C: float = 1.0,
        class_weight: Optional[str] = None,
        random_state: int = 42,
    ):
        self.sections = sections or [
            "court_reasoning",
            "legal_provisions",
            "facts",
            "arguments",
            "procedural_history",
        ]
        self.section_weights = section_weights or self.DEFAULT_SECTION_WEIGHTS
        self.max_features_per_section = max_features_per_section
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state

        self.segmenter = LegalSectionSegmenter()
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=1000,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.classes_ = np.array(CANONICAL_CLASSES)

    def _prepare_sections(self, texts: List[str]) -> Dict[str, List[str]]:
        corpus_by_sec = {s: [] for s in self.sections}
        for t in texts:
            seg = self.segmenter.segment(t)
            for s in self.sections:
                corpus_by_sec[s].append(seg.get(s, ""))
        return corpus_by_sec

    def fit(self, texts: List[str], y: np.ndarray):
        corpus_by_sec = self._prepare_sections(texts)
        feature_blocks = []
        n_samples = max(1, len(texts))
        actual_min_df = min(3, n_samples)

        for s in self.sections:
            docs = corpus_by_sec[s]
            if all(not d.strip() for d in docs):
                docs = ["empty_section"] * len(docs)
            vec = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=self.max_features_per_section,
                sublinear_tf=True,
                min_df=actual_min_df,
            )
            try:
                X_sec = vec.fit_transform(docs)
            except ValueError:
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
                X_sec = vec.fit_transform(["empty_section"] * len(docs))
            weight = self.section_weights.get(s, 1.0)
            X_sec = X_sec.multiply(weight)
            self.vectorizers[s] = vec
            feature_blocks.append(X_sec)

        X_full = sp.hstack(feature_blocks, format="csr")
        self.classifier.fit(X_full, y)
        self.classes_ = self.classifier.classes_
        return self

    def transform(self, texts: List[str]) -> sp.csr_matrix:
        corpus_by_sec = self._prepare_sections(texts)
        feature_blocks = []
        for s in self.sections:
            vec = self.vectorizers[s]
            X_sec = vec.transform(corpus_by_sec[s])
            weight = self.section_weights.get(s, 1.0)
            X_sec = X_sec.multiply(weight)
            feature_blocks.append(X_sec)
        return sp.hstack(feature_blocks, format="csr")

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X_full = self.transform(texts)
        return self.classifier.predict_proba(X_full)

    def predict(self, texts: List[str]) -> np.ndarray:
        probs = self.predict_proba(texts)
        preds_idx = np.argmax(probs, axis=1)
        return self.classes_[preds_idx]


class MulticlassCitationAwareClassifier(BaseEstimator, ClassifierMixin):
    """
    Combines word n-gram text representations with 15 standardized dense legal features.
    """

    def __init__(
        self,
        max_features: int = 25000,
        C: float = 1.0,
        class_weight: Optional[str] = None,
        random_state: int = 42,
    ):
        self.max_features = max_features
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state

        self.text_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.max_features,
            sublinear_tf=True,
            min_df=3,
        )
        self.legal_extractor = LegalFeatureExtractor()
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=1000,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.classes_ = np.array(CANONICAL_CLASSES)

    def fit(self, texts: List[str], y: np.ndarray):
        X_text = self.text_vectorizer.fit_transform(texts)
        X_dense = self.legal_extractor.extract_feature_matrix(texts)
        X_dense_scaled = self.scaler.fit_transform(X_dense)

        X_full = sp.hstack([X_text, sp.csr_matrix(X_dense_scaled)], format="csr")
        self.classifier.fit(X_full, y)
        self.classes_ = self.classifier.classes_
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X_text = self.text_vectorizer.transform(texts)
        X_dense = self.legal_extractor.extract_feature_matrix(texts)
        X_dense_scaled = self.scaler.transform(X_dense)

        X_full = sp.hstack([X_text, sp.csr_matrix(X_dense_scaled)], format="csr")
        return self.classifier.predict_proba(X_full)

    def predict(self, texts: List[str]) -> np.ndarray:
        probs = self.predict_proba(texts)
        preds_idx = np.argmax(probs, axis=1)
        return self.classes_[preds_idx]


# Deep Learning PyTorch Models for 3-Class Legal Outcome Prediction
class MulticlassSectionAttentionLayer(nn.Module):
    """Additive attention mechanism over section embeddings."""

    def __init__(self, hidden_dim: int, attn_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False),
        )

    def forward(self, section_embeds: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.projection(section_embeds).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), section_embeds).squeeze(1)
        return context, weights


class MulticlassHierarchicalModel(nn.Module):
    """
    Hierarchical Legal Document Architecture for 3-class prediction:
    Section Vectors -> BiLSTM Document Contextualizer -> Self-Attention -> 3-Class Linear Head.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        lstm_hidden: int = 128,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = MulticlassSectionAttentionLayer(hidden_dim=lstm_hidden * 2, attn_dim=64)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, section_embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.bilstm(section_embeddings)
        context, attn_weights = self.attention(lstm_out, mask)
        context = self.dropout(context)
        logits = self.classifier(context)
        return logits, attn_weights


class MulticlassSectionTransformer(nn.Module):
    """
    Direct attention pooling over InLegalBERT section embeddings with a 3-class classification head.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        attn_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.attention = MulticlassSectionAttentionLayer(hidden_dim=embed_dim, attn_dim=attn_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, section_embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        context, attn_weights = self.attention(section_embeddings, mask)
        context = self.dropout(context)
        logits = self.classifier(context)
        return logits, attn_weights

