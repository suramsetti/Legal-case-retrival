"""
Phase 18D: Legal Transformer Architectures for Indian Judgment Outcome Prediction.

Implements 4 architectures with chunk-level document aggregation:
1. InLegalBERTSequenceClassifier: InLegalBERT chunk embeddings + Mean/Max Document Pooling.
2. InLegalBERTBiLSTM: InLegalBERT chunk embeddings + Bidirectional LSTM over chunk sequence.
3. InLegalBERTAttentionPooling: InLegalBERT chunk embeddings + Multi-Head / Self-Attention Pooling.
4. InLegalBERTBiLSTMAttention: InLegalBERT + BiLSTM + Attention Pooling over contextualized chunks.

All models handle variable-length judgments using section/chunk-aware document representations.
Includes offline fallback mode for unit testing and CPU environments.
"""

import os
import glob
import math
from typing import Dict, List, Optional, Tuple, Union

# Set Hugging Face offline environment variables before importing transformers
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL_NAME = "law-ai/InLegalBERT"
HIDDEN_DIM = 768


def find_inlegalbert_path() -> Optional[str]:
    """Find local snapshot path for InLegalBERT that contains actual model weights."""
    cached_pattern = os.path.expanduser("~/.cache/huggingface/hub/models--law-ai--InLegalBERT/snapshots/*")
    snapshots = glob.glob(cached_pattern)
    for s in snapshots:
        if os.path.isdir(s) and (os.path.isfile(os.path.join(s, "pytorch_model.bin")) or os.path.isfile(os.path.join(s, "model.safetensors"))):
            return s
    return None


class AttentionPooling(nn.Module):
    """
    Self-attention pooling over a sequence of chunk/token vectors.
    Computes alpha_i = softmax(v^T tanh(W h_i + b)) and returns weighted sum.
    """
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        att_dim = hidden_dim or input_dim // 2
        self.projection = nn.Sequential(
            nn.Linear(input_dim, att_dim),
            nn.Tanh(),
            nn.Linear(att_dim, 1, bias=False)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, input_dim]
            mask: Optional boolean tensor of shape [batch_size, seq_len] (True for valid, False for pad)
        Returns:
            pooled: Tensor of shape [batch_size, input_dim]
            weights: Attention weights of shape [batch_size, seq_len, 1]
        """
        scores = self.projection(x)  # [batch_size, seq_len, 1]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = F.softmax(scores, dim=1)  # [batch_size, seq_len, 1]
        pooled = torch.sum(x * weights, dim=1)  # [batch_size, input_dim]
        return pooled, weights


class InLegalBERTBackbone(nn.Module):
    """
    Shared backbone wrapper for law-ai/InLegalBERT with offline fallback support.
    """
    def __init__(self, model_name_or_path: str = DEFAULT_MODEL_NAME, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.hidden_dim = hidden_dim
        self.encoder = None

        self.encoder = None
        load_path = model_name_or_path
        if load_path == DEFAULT_MODEL_NAME:
            local_path = find_inlegalbert_path()
            if local_path:
                load_path = local_path

        if load_path:
            try:
                from transformers import AutoModel
                self.encoder = AutoModel.from_pretrained(load_path, local_files_only=True)
                self.hidden_dim = self.encoder.config.hidden_size
            except Exception:
                self.encoder = None

        if self.encoder is None:
            # Lightweight offline fallback for testing & CPU execution
            self.fallback_embedding = nn.Embedding(30522, hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            cls_repr: [batch_size, hidden_dim]
        """
        if self.encoder is not None:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token representation
            return outputs.last_hidden_state[:, 0, :]
        else:
            # Fallback for CPU / offline testing
            emb = self.fallback_embedding(input_ids.clamp(0, 30521))
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_emb = torch.sum(emb * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_emb / sum_mask


class InLegalBERTSequenceClassifier(nn.Module):
    """
    Model 1: InLegalBERT Sequence Classifier with Chunk Mean/Max Pooling.
    Encodes multiple document chunks and aggregates them into a document representation.
    """
    def __init__(
        self,
        num_classes: int = 2,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = 0.2,
        pooling_mode: str = "mean"  # 'mean', 'max', or 'mean_max'
    ):
        super().__init__()
        self.backbone = InLegalBERTBackbone(model_name_or_path, hidden_dim)
        self.hidden_dim = self.backbone.hidden_dim
        self.pooling_mode = pooling_mode
        self.dropout = nn.Dropout(dropout)

        classifier_in_dim = self.hidden_dim * 2 if pooling_mode == "mean_max" else self.hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, num_classes)
        )

    def forward_from_embeddings(
        self,
        chunk_reprs: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass from pre-computed chunk embeddings [batch_size, num_chunks, hidden_dim]."""
        if chunk_mask is not None:
            c_mask_float = chunk_mask.unsqueeze(-1).float()
            if self.pooling_mode == "mean":
                sum_reprs = (chunk_reprs * c_mask_float).sum(dim=1)
                doc_repr = sum_reprs / c_mask_float.sum(dim=1).clamp(min=1e-9)
            elif self.pooling_mode == "max":
                masked_reprs = chunk_reprs.masked_fill(~chunk_mask.unsqueeze(-1), -1e9)
                doc_repr = masked_reprs.max(dim=1).values
            else:  # mean_max
                mean_p = (chunk_reprs * c_mask_float).sum(dim=1) / c_mask_float.sum(dim=1).clamp(min=1e-9)
                max_p = chunk_reprs.masked_fill(~chunk_mask.unsqueeze(-1), -1e9).max(dim=1).values
                doc_repr = torch.cat([mean_p, max_p], dim=-1)
        else:
            if self.pooling_mode == "mean":
                doc_repr = chunk_reprs.mean(dim=1)
            elif self.pooling_mode == "max":
                doc_repr = chunk_reprs.max(dim=1).values
            else:
                doc_repr = torch.cat([chunk_reprs.mean(dim=1), chunk_reprs.max(dim=1).values], dim=-1)

        doc_repr = self.dropout(doc_repr)
        return self.classifier(doc_repr)

    def forward(
        self,
        chunk_input_ids: torch.Tensor,
        chunk_attention_mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            chunk_input_ids: [batch_size, num_chunks, seq_len]
            chunk_attention_mask: [batch_size, num_chunks, seq_len]
            chunk_mask: Optional [batch_size, num_chunks] boolean mask
        Returns:
            logits: [batch_size, num_classes]
        """
        b, c, s = chunk_input_ids.shape
        flat_ids = chunk_input_ids.view(b * c, s)
        flat_mask = chunk_attention_mask.view(b * c, s)

        chunk_reprs = self.backbone(flat_ids, flat_mask)  # [b * c, hidden_dim]
        chunk_reprs = chunk_reprs.view(b, c, self.hidden_dim)  # [b, c, hidden_dim]
        return self.forward_from_embeddings(chunk_reprs, chunk_mask)


class InLegalBERTBiLSTM(nn.Module):
    """
    Model 2: InLegalBERT + Bidirectional LSTM over chunk sequence.
    Captures temporal & rhetorical progression across document sections.
    """
    def __init__(
        self,
        num_classes: int = 2,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        hidden_dim: int = HIDDEN_DIM,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.backbone = InLegalBERTBackbone(model_name_or_path, hidden_dim)
        self.hidden_dim = self.backbone.hidden_dim
        self.lstm_hidden = lstm_hidden

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(lstm_hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward_from_embeddings(
        self,
        chunk_reprs: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass from pre-computed chunk embeddings [batch_size, num_chunks, hidden_dim]."""
        lstm_out, _ = self.lstm(chunk_reprs)  # [b, c, 2 * lstm_hidden]

        if chunk_mask is not None:
            m = chunk_mask.unsqueeze(-1).float()
            doc_repr = (lstm_out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        else:
            doc_repr = lstm_out.mean(dim=1)

        doc_repr = self.layer_norm(doc_repr)
        doc_repr = self.dropout(doc_repr)
        return self.classifier(doc_repr)

    def forward(
        self,
        chunk_input_ids: torch.Tensor,
        chunk_attention_mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, c, s = chunk_input_ids.shape
        flat_ids = chunk_input_ids.view(b * c, s)
        flat_mask = chunk_attention_mask.view(b * c, s)

        chunk_reprs = self.backbone(flat_ids, flat_mask).view(b, c, self.hidden_dim)
        return self.forward_from_embeddings(chunk_reprs, chunk_mask)


class InLegalBERTAttentionPooling(nn.Module):
    """
    Model 3: InLegalBERT + Self-Attention Pooling over chunk sequence.
    Learns to dynamically weight salient sections (e.g. facts vs. arguments vs. analysis).
    """
    def __init__(
        self,
        num_classes: int = 2,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = 0.2
    ):
        super().__init__()
        self.backbone = InLegalBERTBackbone(model_name_or_path, hidden_dim)
        self.hidden_dim = self.backbone.hidden_dim

        self.attention_pool = AttentionPooling(self.hidden_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, num_classes)
        )

    def forward_from_embeddings(
        self,
        chunk_reprs: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass from pre-computed chunk embeddings [batch_size, num_chunks, hidden_dim]."""
        doc_repr, _ = self.attention_pool(chunk_reprs, chunk_mask)
        doc_repr = self.layer_norm(doc_repr)
        doc_repr = self.dropout(doc_repr)
        return self.classifier(doc_repr)

    def forward(
        self,
        chunk_input_ids: torch.Tensor,
        chunk_attention_mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, c, s = chunk_input_ids.shape
        flat_ids = chunk_input_ids.view(b * c, s)
        flat_mask = chunk_attention_mask.view(b * c, s)

        chunk_reprs = self.backbone(flat_ids, flat_mask).view(b, c, self.hidden_dim)
        return self.forward_from_embeddings(chunk_reprs, chunk_mask)


class InLegalBERTBiLSTMAttention(nn.Module):
    """
    Model 4: InLegalBERT + BiLSTM + Attention Pooling.
    Combines sequential discourse progression with learned section salience.
    """
    def __init__(
        self,
        num_classes: int = 2,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        hidden_dim: int = HIDDEN_DIM,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.backbone = InLegalBERTBackbone(model_name_or_path, hidden_dim)
        self.hidden_dim = self.backbone.hidden_dim
        self.lstm_hidden = lstm_hidden

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.attention_pool = AttentionPooling(lstm_hidden * 2)
        self.layer_norm = nn.LayerNorm(lstm_hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward_from_embeddings(
        self,
        chunk_reprs: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass from pre-computed chunk embeddings [batch_size, num_chunks, hidden_dim]."""
        lstm_out, _ = self.lstm(chunk_reprs)  # [b, c, 2 * lstm_hidden]
        doc_repr, _ = self.attention_pool(lstm_out, chunk_mask)
        doc_repr = self.layer_norm(doc_repr)
        doc_repr = self.dropout(doc_repr)
        return self.classifier(doc_repr)

    def forward(
        self,
        chunk_input_ids: torch.Tensor,
        chunk_attention_mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, c, s = chunk_input_ids.shape
        flat_ids = chunk_input_ids.view(b * c, s)
        flat_mask = chunk_attention_mask.view(b * c, s)

        chunk_reprs = self.backbone(flat_ids, flat_mask).view(b, c, self.hidden_dim)
        return self.forward_from_embeddings(chunk_reprs, chunk_mask)
