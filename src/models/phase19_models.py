"""
Phase 19: Advanced Legal Document Representation and Classification Models.

Implements:
1. SectionAwareTfidfClassifier: Multi-view TF-IDF vectorization across canonical legal sections.
2. CharNgramClassifier: Character-level n-gram representation for legal sub-words and morphosyntax.
3. SectionAwareInLegalBERT: Attention pooling across chronological section embeddings.
4. HierarchicalLegalModel: Token -> Section -> Document BiLSTM contextualization with attention.
5. CitationAwareClassifier: Combines lexical text features with 15 dense statutory/citation features.
6. HybridFeatureFusionClassifier: Early (feature-level) and late (probability-level) fusion architectures.
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


class SectionAwareTfidfClassifier(BaseEstimator, ClassifierMixin):
    """
    Multi-view section-aware TF-IDF classifier.
    Fits independent TF-IDF representations for each canonical legal section,
    applies rhetorical weighting, and trains a linear model.
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
        ngram_range: Tuple[int, int] = (1, 2),
        max_features_per_section: int = 10000,
        clf_type: str = "lr",
        C: float = 1.0,
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
        self.ngram_range = ngram_range
        self.max_features_per_section = max_features_per_section
        self.clf_type = clf_type
        self.C = C
        self.random_state = random_state

        self.segmenter = LegalSectionSegmenter()
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.classifier = None
        self.classes_ = np.array([0, 1])

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
                ngram_range=self.ngram_range,
                max_features=self.max_features_per_section,
                sublinear_tf=True,
                min_df=actual_min_df,
            )
            try:
                X_sec = vec.fit_transform(docs)
            except ValueError:
                vec = TfidfVectorizer(ngram_range=self.ngram_range, min_df=1)
                X_sec = vec.fit_transform(["empty_section"] * len(docs))
            weight = self.section_weights.get(s, 1.0)
            X_sec = X_sec.multiply(weight)
            self.vectorizers[s] = vec
            feature_blocks.append(X_sec)

        X_full = sp.hstack(feature_blocks, format="csr")

        if self.clf_type == "svm":
            base_svm = LinearSVC(C=self.C, random_state=self.random_state, max_iter=2000)
            self.classifier = CalibratedClassifierCV(base_svm, cv=3)
        else:
            self.classifier = LogisticRegression(
                C=self.C,
                max_iter=1000,
                random_state=self.random_state,
                solver="liblinear",
            )

        self.classifier.fit(X_full, y)
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

    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(texts)[:, 1]
        return (probs >= threshold).astype(int)


class CharNgramClassifier(BaseEstimator, ClassifierMixin):
    """
    Sub-word character n-gram classifier.
    Captures legal morphosyntax, citation patterns, and legal affixation.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 5),
        max_features: int = 25000,
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.C = C
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
            random_state=self.random_state,
            solver="liblinear",
        )
        self.classes_ = np.array([0, 1])

    def fit(self, texts: List[str], y: np.ndarray):
        self.vectorizer.min_df = min(5, len(texts))
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, y)
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(texts)[:, 1]
        return (probs >= threshold).astype(int)


class CitationAwareClassifier(BaseEstimator, ClassifierMixin):
    """
    Combines text features (TF-IDF or embeddings) with 15 dense legal features.
    """

    def __init__(
        self,
        text_vectorizer=None,
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.text_vectorizer = text_vectorizer or TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=25000,
            sublinear_tf=True,
            min_df=3,
        )
        self.legal_extractor = LegalFeatureExtractor()
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=random_state,
            solver="liblinear",
        )
        self.classes_ = np.array([0, 1])

    def fit(self, texts: List[str], y: np.ndarray):
        X_text = self.text_vectorizer.fit_transform(texts)
        X_dense = self.legal_extractor.extract_feature_matrix(texts)
        X_dense_scaled = self.scaler.fit_transform(X_dense)

        X_full = sp.hstack([X_text, sp.csr_matrix(X_dense_scaled)], format="csr")
        self.classifier.fit(X_full, y)
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X_text = self.text_vectorizer.transform(texts)
        X_dense = self.legal_extractor.extract_feature_matrix(texts)
        X_dense_scaled = self.scaler.transform(X_dense)

        X_full = sp.hstack([X_text, sp.csr_matrix(X_dense_scaled)], format="csr")
        return self.classifier.predict_proba(X_full)

    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(texts)[:, 1]
        return (probs >= threshold).astype(int)


# PyTorch Deep Learning Modules for Section-Aware & Hierarchical Legal Modeling
class SectionAttentionLayer(nn.Module):
    """Additive attention mechanism over section embeddings."""

    def __init__(self, hidden_dim: int, attn_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False),
        )

    def forward(self, section_embeds: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # section_embeds: (batch_size, num_sections, hidden_dim)
        scores = self.projection(section_embeds).squeeze(-1)  # (batch_size, num_sections)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)  # (batch_size, num_sections)
        context = torch.bmm(weights.unsqueeze(1), section_embeds).squeeze(1)  # (batch_size, hidden_dim)
        return context, weights


class HierarchicalLegalModel(nn.Module):
    """
    Hierarchical Legal Document Architecture:
    Section Vectors -> BiLSTM Document Contextualizer -> Self-Attention -> Classification Head.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        lstm_hidden: int = 128,
        num_classes: int = 1,
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
        self.attention = SectionAttentionLayer(hidden_dim=lstm_hidden * 2, attn_dim=64)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, section_embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # section_embeddings: (batch_size, num_sections, 768)
        lstm_out, _ = self.bilstm(section_embeddings)  # (batch, num_sections, 256)
        context, attn_weights = self.attention(lstm_out, mask)  # (batch, 256)
        context = self.dropout(context)
        logits = self.classifier(context)  # (batch, 1)
        return logits, attn_weights


class SectionAwareTransformerClassifier(nn.Module):
    """
    Section-Aware Attention Pooling directly on InLegalBERT section embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        attn_dim: int = 128,
        num_classes: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.attention = SectionAttentionLayer(hidden_dim=embed_dim, attn_dim=attn_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, section_embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        context, attn_weights = self.attention(section_embeddings, mask)
        context = self.dropout(context)
        logits = self.classifier(context)
        return logits, attn_weights

