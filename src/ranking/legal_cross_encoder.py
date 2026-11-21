"""
Legal Cross-Encoder Architecture with Section-Aware Chunking and Rhetorical-Aware Scoring.

Implements:
1. Section-aware chunking: partitions lengthy judgments into structurally informative segments
   (Ratio Decidendi / Legal Reasoning, Facts, Final Ruling, Headnote) instead of blind truncation.
2. InLegalBERTCrossEncoder: Joint token-level attention over [CLS] Query [SEP] Candidate Chunk [SEP]
   using law-ai/InLegalBERT with a relevance scoring head.
3. Chunk score aggregation: Max, Top-K Mean, and Rhetorical-Weighted pooling.
4. Rhetorical-aware score adjustment combining joint cross-attention with discourse similarity.
"""

from typing import Dict, List, Optional, Tuple, Any, Sequence, Union
from pathlib import Path
import re
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from src.legal_features.rhetorical_analyzer import RhetoricalAnalyzer, ROLE_TO_IDX, RHETORICAL_ROLES


class SectionAwareChunker:
    """
    Partitions long legal judgments into structurally salient rhetorical sections:
    - Ratio Decidendi / Legal Reasoning (Role: RatioOfTheDecision, Issue, Precedent)
    - Factual & Procedural Posture (Role: Fact, RulingByLowerCourt)
    - Final Decision & Ruling (Role: RulingByPresentCourt)
    - Primary Headnote / Leading Chunk
    """

    def __init__(self, max_tokens_per_chunk: int = 256):
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.rhetorical_analyzer = RhetoricalAnalyzer()

    def extract_sections(self, sentences: Sequence[str]) -> Dict[str, str]:
        """
        Groups document sentences into coherent functional sections based on discourse cues.
        """
        if not sentences:
            return {"headnote": "", "ratio": "", "facts": "", "ruling": ""}

        headnote_sents = list(sentences[:3])
        ratio_sents: List[str] = []
        fact_sents: List[str] = []
        ruling_sents: List[str] = []

        total_sents = len(sentences)
        for idx, sent in enumerate(sentences):
            pos_ratio = idx / max(1, total_sents - 1)
            role, conf = self.rhetorical_analyzer.classify_sentence(sent, position_ratio=pos_ratio)

            if role in ("RatioOfTheDecision", "Issue", "PrecedentReliedUpon"):
                ratio_sents.append(sent)
            elif role in ("Fact", "RulingByLowerCourt"):
                fact_sents.append(sent)
            elif role == "RulingByPresentCourt":
                ruling_sents.append(sent)

        # Fallbacks if a role was not detected
        if not ratio_sents:
            mid = total_sents // 2
            ratio_sents = list(sentences[max(0, mid - 3): min(total_sents, mid + 3)])
        if not fact_sents:
            fact_sents = list(sentences[:min(5, total_sents)])
        if not ruling_sents:
            ruling_sents = list(sentences[max(0, total_sents - 3):])

        return {
            "headnote": " ".join(headnote_sents[:3])[:800],
            "ratio": " ".join(ratio_sents[:4])[:1200],
            "facts": " ".join(fact_sents[:3])[:800],
            "ruling": " ".join(ruling_sents[:3])[:800],
        }

    def chunk_document(self, text: Union[str, Sequence[str]]) -> List[Dict[str, Any]]:
        """
        Returns a list of section-aware chunks with section type and text.
        """
        if isinstance(text, str):
            # Split into sentence-like strings
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        else:
            sentences = list(text)

        sections = self.extract_sections(sentences)
        chunks = [
            {"section_type": "ratio", "text": sections["ratio"], "rhetorical_weight": 1.25},
            {"section_type": "headnote", "text": sections["headnote"], "rhetorical_weight": 1.00},
            {"section_type": "ruling", "text": sections["ruling"], "rhetorical_weight": 1.10},
            {"section_type": "facts", "text": sections["facts"], "rhetorical_weight": 0.90},
        ]
        # Filter out empty texts
        return [c for c in chunks if c["text"].strip()]


class InLegalBERTCrossEncoder(nn.Module):
    """
    Joint cross-attention scoring network over Query-Candidate pairs using InLegalBERT.
    """

    def __init__(
        self,
        model_name_or_path: str = "law-ai/InLegalBERT",
        freeze_lower_layers: int = 10,
        dropout_rate: float = 0.15,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
        self.bert = AutoModel.from_pretrained(model_name_or_path, local_files_only=True)

        # Freeze lower transformer layers for efficient training on CPU
        if freeze_lower_layers > 0:
            for name, param in self.bert.named_parameters():
                layer_match = re.search(r"layer\.(\d+)\.", name)
                if layer_match:
                    layer_num = int(layer_match.group(1))
                    if layer_num < freeze_lower_layers:
                        param.requires_grad = False
                elif "embeddings" in name:
                    param.requires_grad = False

        hidden_dim = self.bert.config.hidden_size
        self.scorer = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )
        self.to(self.device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes relevance score for query-candidate token sequences.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Pooled representation via [CLS] token (index 0)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        score = self.scorer(cls_rep).squeeze(-1)
        return score

    def score_pairs(
        self,
        query_texts: Sequence[str],
        candidate_texts: Sequence[str],
        max_length: int = 256,
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Infers scalar cross-encoder scores for (query, candidate_chunk) pairs.
        """
        self.eval()
        n_pairs = len(query_texts)
        if n_pairs == 0:
            return np.array([], dtype=np.float32)

        all_scores: List[float] = []
        with torch.no_grad():
            for i in range(0, n_pairs, batch_size):
                q_batch = list(query_texts[i : i + batch_size])
                c_batch = list(candidate_texts[i : i + batch_size])

                encoded = self.tokenizer(
                    q_batch,
                    c_batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                batch_scores = self(input_ids, attention_mask)
                all_scores.extend(batch_scores.cpu().numpy().tolist())

        return np.array(all_scores, dtype=np.float32)

    def score_candidate_chunks(
        self,
        query_text: str,
        chunks: List[Dict[str, Any]],
        aggregation: str = "weighted",
    ) -> float:
        """
        Scores multiple section-aware chunks of a single candidate and aggregates them.
        Aggregation strategies:
        - 'max': highest chunk score
        - 'mean': simple average over all chunks
        - 'weighted': rhetorical-weighted average favoring Ratio Decidendi and Ruling
        """
        if not chunks:
            return 0.0

        q_list = [query_text] * len(chunks)
        c_list = [c["text"] for c in chunks]
        scores = self.score_pairs(q_list, c_list)

        if aggregation == "max":
            return float(np.max(scores))
        elif aggregation == "mean":
            return float(np.mean(scores))
        elif aggregation == "weighted":
            weights = np.array([c.get("rhetorical_weight", 1.0) for c in chunks], dtype=np.float32)
            weights = weights / np.sum(weights)
            return float(np.sum(scores * weights))
        else:
            return float(np.max(scores))

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves model weights and configuration."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(filepath))

    def load(self, filepath: Union[str, Path]) -> None:
        """Loads model weights."""
        state_dict = torch.load(str(filepath), map_location=self.device, weights_only=True)
        self.load_state_dict(state_dict)

