"""
Unit tests for Phase 19: Advanced Legal Document Representation and Models.

Verifies:
1. Legal Section Segmentation & Discourse Fallback
2. Dispositive Marker Neutralization & Zero Leakage
3. Section-Aware Rhetorical Chunking
4. Legal Feature Extraction & Bounds
5. Section-Aware TF-IDF, Char N-Gram, Citation-Aware & Deep Model Forward Passes
6. Split Isolation & Group Awareness
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.phase19_section_segmenter import LegalSectionSegmenter
from src.preprocessing.phase19_section_chunker import LegalSectionChunker
from src.features.phase19_legal_features import LegalFeatureExtractor
from src.models.phase19_models import (
    SectionAwareTfidfClassifier,
    CharNgramClassifier,
    CitationAwareClassifier,
    SectionAwareTransformerClassifier,
    HierarchicalLegalModel,
)


class TestSectionSegmentation:
    """Verifies that legal judgments are segmented accurately into canonical sections without leakage."""

    def test_explicit_headers_segmentation(self):
        sample_text = (
            "FACTS OF THE CASE:\nThe appellant was charged under section 302 of the penal code.\n"
            "PROCEDURAL HISTORY:\nThe trial court convicted the appellant on March 15.\n"
            "ARGUMENTS:\nLearned senior counsel contended that no eyewitness was present.\n"
            "COURT REASONING:\nIn our considered view, the chain of circumstantial evidence is broken.\n"
            "[MASKED_CONCLUDING_DISPOSITION]"
        )
        segmenter = LegalSectionSegmenter()
        seg = segmenter.segment(sample_text)

        assert "appellant was charged" in seg["facts"]
        assert "trial court convicted" in seg["procedural_history"]
        assert "eyewitness was present" in seg["arguments"]
        assert "circumstantial evidence is broken" in seg["court_reasoning"]
        # Guarantee no dispositive marker leaked into any section
        for s, text in seg.items():
            assert "[MASKED_CONCLUDING_DISPOSITION]" not in text

    def test_discourse_cue_fallback(self):
        sample_text = (
            "The prosecution alleged that the incident occurred on December 5.\n"
            "The learned sessions judge convicted the accused under section 304.\n"
            "Learned counsel appearing for the appellant argued that self defence applied.\n"
            "We have given our anxious consideration and find that the evidence is unreliable."
        )
        segmenter = LegalSectionSegmenter()
        seg = segmenter.segment(sample_text)
        assert len(seg["facts"]) > 0
        assert len(seg["procedural_history"]) > 0
        assert len(seg["arguments"]) > 0
        assert len(seg["court_reasoning"]) > 0

    def test_dispositive_marker_neutralization(self):
        segmenter = LegalSectionSegmenter()
        dirty = "Factual text [MASKED_PROOF_SENTENCE] followed by [MASKED_CONCLUDING_DISPOSITION]"
        cleaned = segmenter.clean_text_for_segmentation(dirty)
        assert "[MASKED_PROOF_SENTENCE]" not in cleaned
        assert "[MASKED_CONCLUDING_DISPOSITION]" not in cleaned


class TestSectionChunking:
    """Verifies section-aware rhetorical chunking behavior."""

    def test_reasoning_chunk_capture(self):
        chunker = LegalSectionChunker(max_chunk_words=20, total_chunks_per_doc=4)
        sample_text = (
            "Facts: " + "word " * 50 + "\n"
            "Reasoning: " + "analysis " * 50
        )
        chunks = chunker.chunk_document("doc_001", sample_text)
        assert len(chunks) <= 4
        # Verify court_reasoning chunk is included
        sections = [c["section"] for c in chunks]
        assert "court_reasoning" in sections


class TestLegalFeatureExtraction:
    """Verifies structured legal feature extraction."""

    def test_feature_matrix_dimensions(self):
        extractor = LegalFeatureExtractor()
        texts = [
            "Convicted under Section 302 of Indian Penal Code and Article 21 of the Constitution.",
            "Suit under Section 96 of CPC with reference to AIR 1980 SC 123.",
        ]
        matrix = extractor.extract_feature_matrix(texts)
        assert matrix.shape == (2, 15)
        assert not np.any(np.isnan(matrix))
        assert not np.any(np.isinf(matrix))

    def test_ipc_and_const_detection(self):
        extractor = LegalFeatureExtractor()
        feats = extractor.extract_features_single("Section 302 of IPC and Article 32")
        assert feats[0] >= 1  # ipc_count
        assert feats[2] >= 1  # constitution_article_count


class TestModelArchitectures:
    """Tests forward passes and predict_proba on representative models."""

    def test_section_aware_tfidf_fit_predict(self):
        texts = [
            "Facts: robbery occurred. Reasoning: conviction set aside.",
            "Facts: contract dispute. Reasoning: appeal dismissed.",
            "Facts: murder accused. Reasoning: appeal allowed with costs.",
            "Facts: land acquisition. Reasoning: high court upheld.",
        ]
        y = np.array([1, 0, 1, 0])
        clf = SectionAwareTfidfClassifier(max_features_per_section=100)
        clf.fit(texts, y)
        probs = clf.predict_proba(texts)
        assert probs.shape == (4, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_char_ngram_fit_predict(self):
        texts = ["Legal precedent upheld.", "Order of acquittal reversed."]
        y = np.array([0, 1])
        clf = CharNgramClassifier(max_features=200)
        clf.fit(texts, y)
        probs = clf.predict_proba(texts)
        assert probs.shape == (2, 2)

    def test_hierarchical_legal_model_forward(self):
        model = HierarchicalLegalModel(embed_dim=64, lstm_hidden=32)
        # batch=2, sections=5, embed_dim=64
        x = torch.randn(2, 5, 64)
        logits, attn = model(x)
        assert logits.shape == (2, 1)
        assert attn.shape == (2, 5)
        # Attention weights should sum to 1 across sections
        assert torch.allclose(attn.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_section_aware_transformer_forward(self):
        model = SectionAwareTransformerClassifier(embed_dim=64, attn_dim=32)
        x = torch.randn(3, 4, 64)
        logits, attn = model(x)
        assert logits.shape == (3, 1)
        assert attn.shape == (3, 4)
        assert torch.allclose(attn.sum(dim=-1), torch.ones(3), atol=1e-5)


class TestSplitIsolation:
    """Verifies that no IDs or duplicate groups leak across train, val, and test partitions."""

    def test_zero_leakage_across_partitions(self):
        tr = pd.read_parquet("data/processed/judipl/phase18c/binary/train_masked.parquet")
        va = pd.read_parquet("data/processed/judipl/phase18c/binary/val_masked.parquet")
        te = pd.read_parquet("data/processed/judipl/phase18c/binary/test_masked.parquet")

        s_tr = set(tr["id"])
        s_va = set(va["id"])
        s_te = set(te["id"])

        assert len(s_tr.intersection(s_va)) == 0, "Train and Val share IDs"
        assert len(s_tr.intersection(s_te)) == 0, "Train and Test share IDs"
        assert len(s_va.intersection(s_te)) == 0, "Val and Test share IDs"

        g_tr = set(tr["duplicate_group"])
        g_va = set(va["duplicate_group"])
        g_te = set(te["duplicate_group"])

        assert len(g_tr.intersection(g_va)) == 0, "Train and Val share duplicate groups"
        assert len(g_tr.intersection(g_te)) == 0, "Train and Test share duplicate groups"
        assert len(g_va.intersection(g_te)) == 0, "Val and Test share duplicate groups"

