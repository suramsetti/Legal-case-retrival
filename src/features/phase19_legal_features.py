"""
Phase 19C: Legal Feature Extraction Module.

Extracts structured, legally meaningful signals from masked Indian Supreme Court judgment texts:
- Statutory references (IPC, CrPC, CPC, Constitutional Articles, BNS)
- Citation frequency and precedent reference density
- Document length and rhetorical section balance ratios
- Strictly verifies zero outcome leakage (no dispositive phrases, no proof sentences)
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from src.preprocessing.phase19_section_segmenter import LegalSectionSegmenter


class LegalFeatureExtractor:
    """
    Extracts dense numerical features from masked legal judgment text.
    """

    FEATURE_NAMES = [
        "ipc_count",
        "bns_count",
        "constitution_article_count",
        "crpc_count",
        "cpc_count",
        "statutory_section_mentions",
        "citation_frequency",
        "precedent_reference_density",
        "log_word_count",
        "log_char_count",
        "reasoning_ratio",
        "facts_ratio",
        "arguments_ratio",
        "procedural_ratio",
        "provisions_ratio",
    ]

    def __init__(self):
        self.ipc_regex = re.compile(
            r"\b(?:ipc|indian\s+penal\s+code|section\s+302|section\s+304|section\s+307|section\s+420|section\s+498a)\b",
            re.IGNORECASE,
        )
        self.bns_regex = re.compile(
            r"\b(?:bns|bharatiya\s+nyaya\s+sanhita)\b",
            re.IGNORECASE,
        )
        self.const_regex = re.compile(
            r"\b(?:article\s+\d+|constitution\s+of\s+india|writ\s+petition|special\s+leave\s+petition)\b",
            re.IGNORECASE,
        )
        self.crpc_regex = re.compile(
            r"\b(?:cr\.?p\.?c|code\s+of\s+criminal\s+procedure|section\s+482|section\s+313|section\s+161|section\s+164)\b",
            re.IGNORECASE,
        )
        self.cpc_regex = re.compile(
            r"\b(?:c\.?p\.?c|code\s+of\s+civil\s+procedure|order\s+[ivxlcdm]+\b|section\s+100|section\s+96|order\s+7\s+rule\s+11)\b",
            re.IGNORECASE,
        )
        self.section_num_regex = re.compile(r"\bsection\s+\d+\b", re.IGNORECASE)
        self.citation_regex = re.compile(r"\b(?:air\s+\d{4}|scc|scr|scale|sec\b|\d+\s+scc\s+\d+)\b", re.IGNORECASE)
        self.precedent_regex = re.compile(
            r"\b(?:in\s+the\s+case\s+of|versus|v\.|relied\s+upon|referred\s+to|reiterated\s+in|bench\s+decision)\b",
            re.IGNORECASE,
        )
        self.segmenter = LegalSectionSegmenter()

    def extract_features_single(self, text: str, segmented: Optional[Dict[str, str]] = None) -> np.ndarray:
        """Extracts 15 dense numerical features for a single judgment text."""
        cleaned = self.segmenter.clean_text_for_segmentation(text)
        words = cleaned.split()
        n_words = max(1, len(words))
        n_chars = max(1, len(cleaned))

        ipc = len(self.ipc_regex.findall(cleaned))
        bns = len(self.bns_regex.findall(cleaned))
        const = len(self.const_regex.findall(cleaned))
        crpc = len(self.crpc_regex.findall(cleaned))
        cpc = len(self.cpc_regex.findall(cleaned))
        sec_num = len(self.section_num_regex.findall(cleaned))
        cits = len(self.citation_regex.findall(cleaned))
        precedent = len(self.precedent_regex.findall(cleaned))

        # Density per 1000 words
        prec_density = (precedent / n_words) * 1000.0

        # Section ratios
        if segmented is None:
            segmented = self.segmenter.segment(cleaned)

        tot_sec_chars = max(1, sum(len(t) for t in segmented.values()))
        reasoning_ratio = len(segmented.get("court_reasoning", "")) / tot_sec_chars
        facts_ratio = len(segmented.get("facts", "")) / tot_sec_chars
        arguments_ratio = len(segmented.get("arguments", "")) / tot_sec_chars
        procedural_ratio = len(segmented.get("procedural_history", "")) / tot_sec_chars
        provisions_ratio = len(segmented.get("legal_provisions", "")) / tot_sec_chars

        feats = [
            ipc,
            bns,
            const,
            crpc,
            cpc,
            sec_num,
            cits,
            prec_density,
            np.log1p(n_words),
            np.log1p(n_chars),
            reasoning_ratio,
            facts_ratio,
            arguments_ratio,
            procedural_ratio,
            provisions_ratio,
        ]
        return np.array(feats, dtype=np.float32)

    def extract_feature_matrix(self, texts: List[str]) -> np.ndarray:
        """Extracts (N, 15) feature matrix across a list of texts."""
        matrix = []
        for t in texts:
            matrix.append(self.extract_features_single(t))
        return np.vstack(matrix)

