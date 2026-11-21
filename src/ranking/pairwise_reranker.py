"""
Pairwise Neural Re-Ranker for Second-Stage Legal Case Retrieval.

Implements a deep interaction re-ranking network trained with pairwise margin loss
on hard negatives from the Stage-1 candidate pool.
"""

from typing import List, Optional, Tuple, Dict, Any, Union
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class PairwiseDataset(Dataset):
    """Dataset yielding (positive_features, negative_features) pairs."""

    def __init__(self, pos_features: np.ndarray, neg_features: np.ndarray):
        self.pos_features = torch.tensor(pos_features, dtype=torch.float32)
        self.neg_features = torch.tensor(neg_features, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.pos_features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pos_features[idx], self.neg_features[idx]


class PairwiseReRankingNetwork(nn.Module):
    """Deep feature interaction network for candidate re-ranking."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PairwiseNeuralReRanker:
    """Manages training, feature engineering, and inference of the Stage-2 Re-Ranker."""

    FEATURE_COLS = [
        "stage1_score",
        "qnorm_tfidf_score",
        "qnorm_inlegalbert_sim",
        "qnorm_composite_legal_score",
        "qnorm_bm25_score",
        "statute_shared_count",
        "statute_jaccard",
        "statute_containment_query",
        "citation_shared_count",
        "citation_jaccard",
        "entity_shared_count",
        "entity_jaccard",
        "judge_overlap",
        "court_overlap",
        "org_overlap",
        "rhetorical_cosine",
        "rhetorical_js_sim",
        "rhetorical_ratio_diff",
        "lexical_dense_interaction",
        "lexical_legal_interaction",
        "dense_rhetorical_interaction",
        "statutory_density",
        "authority_weight",
        "score_diff_from_top1",
    ]

    def __init__(self, hidden_dim: int = 64, lr: float = 0.001, margin: float = 1.0):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.margin = margin
        self.input_dim = len(self.FEATURE_COLS)
        self.model = PairwiseReRankingNetwork(self.input_dim, hidden_dim=hidden_dim)
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

    @classmethod
    def engineer_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Computes rich cross-modal interaction features for re-ranking."""
        df = df.copy()

        # Query-level min-max normalization
        for col in ["tfidf_score", "inlegalbert_sim", "composite_legal_score", "bm25_score"]:
            s_min = df.groupby("query_id")[col].transform("min")
            s_max = df.groupby("query_id")[col].transform("max")
            diff = s_max - s_min
            df["qnorm_" + col] = (df[col] - s_min) / np.where(diff > 1e-9, diff, 1.0)

        # Base Stage-1 hybrid score
        df["stage1_score"] = (
            0.65 * df["qnorm_tfidf_score"]
            + 0.25 * df["qnorm_inlegalbert_sim"]
            + 0.10 * df["qnorm_composite_legal_score"]
        )

        # Cross-modal interaction terms
        df["lexical_dense_interaction"] = df["qnorm_tfidf_score"] * df["qnorm_inlegalbert_sim"]
        df["lexical_legal_interaction"] = df["qnorm_tfidf_score"] * df["qnorm_composite_legal_score"]
        df["dense_rhetorical_interaction"] = df["qnorm_inlegalbert_sim"] * df["rhetorical_cosine"]
        df["statutory_density"] = df["statute_shared_count"] * (df["qnorm_tfidf_score"] + 0.1)
        df["authority_weight"] = (
            df["statute_shared_count"] * 2.0
            + df["citation_shared_count"] * 3.0
            + df["judge_overlap"] * 1.5
        )

        # Score difference relative to top candidate per query
        top1_score = df.groupby("query_id")["stage1_score"].transform("max")
        df["score_diff_from_top1"] = top1_score - df["stage1_score"]

        return df

    def prepare_pairwise_training_data(
        self,
        df: pd.DataFrame,
        top_k_candidates: int = 30,
        max_pairs_per_query: int = 40,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Constructs (positive, hard_negative) pairs from top-retrieved candidates."""
        # Rank by Stage 1 score
        df["s1_rank"] = df.groupby("query_id")["stage1_score"].rank(ascending=False, method="first")
        top_df = df[df["s1_rank"] <= top_k_candidates].copy()

        pos_list = []
        neg_list = []

        grouped = top_df.groupby("query_id")
        for qid, group in grouped:
            pos_rows = group[group["label"] == 1]
            neg_rows = group[group["label"] == 0]

            if pos_rows.empty or neg_rows.empty:
                continue

            pos_feats = pos_rows[self.FEATURE_COLS].values
            neg_feats = neg_rows[self.FEATURE_COLS].values

            # Form pairs
            count = 0
            for p in pos_feats:
                for n in neg_feats:
                    pos_list.append(p)
                    neg_list.append(n)
                    count += 1
                    if count >= max_pairs_per_query:
                        break
                if count >= max_pairs_per_query:
                    break

        pos_arr = np.array(pos_list, dtype=np.float32)
        neg_arr = np.array(neg_list, dtype=np.float32)

        return pos_arr, neg_arr

    def fit_normalizer(self, X: np.ndarray) -> None:
        """Fits mean and std for feature standardization."""
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0)
        self.feature_stds = np.where(self.feature_stds > 1e-6, self.feature_stds, 1.0)

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """Standardizes features using fitted training statistics."""
        if self.feature_means is None or self.feature_stds is None:
            return X
        return (X - self.feature_means) / self.feature_stds

    def train_epochs(
        self,
        pos_train: np.ndarray,
        neg_train: np.ndarray,
        epochs: int = 25,
        batch_size: int = 128,
        weight_decay: float = 1e-4,
    ) -> List[float]:
        """Trains the pairwise margin ranking network."""
        # Standardize features
        all_features = np.vstack([pos_train, neg_train])
        self.fit_normalizer(all_features)

        norm_pos = self.normalize(pos_train)
        norm_neg = self.normalize(neg_train)

        dataset = PairwiseDataset(norm_pos, norm_neg)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.model.train()
        loss_history = []

        for epoch in range(epochs):
            total_loss = 0.0
            for pos_b, neg_b in dataloader:
                optimizer.zero_grad()
                s_pos = self.model(pos_b)
                s_neg = self.model(neg_b)

                # Pairwise Margin + Softplus Loss:
                # Forces s_pos - s_neg > margin
                diff = s_pos - s_neg
                loss = torch.mean(torch.log1p(torch.exp(-diff)) + torch.relu(self.margin - diff))

                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())

            scheduler.step()
            avg_loss = total_loss / max(len(dataloader), 1)
            loss_history.append(avg_loss)

        return loss_history

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Infers re-ranking scores for all candidate rows."""
        X = df[self.FEATURE_COLS].values.astype(np.float32)
        X_norm = self.normalize(X)
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_norm, dtype=torch.float32)
            scores = self.model(tensor_x).cpu().numpy()
        return scores

    def rerank_query_pool(
        self,
        df: pd.DataFrame,
        top_k_candidates: int = 30,
        alpha: float = 0.5,
    ) -> pd.DataFrame:
        """
        Executes Two-Stage Re-Ranking:
        1. Filters each query pool to Top-K candidates via Stage-1 hybrid score.
        2. Re-ranks Top-K candidates using Stage-2 pairwise network predictions.
        3. Candidates outside Top-K receive lower baseline scores.
        """
        df = df.copy()
        if "s1_rank" not in df.columns:
            df["s1_rank"] = df.groupby("query_id")["stage1_score"].rank(ascending=False, method="first")

        # Top-K slice for Stage 2
        is_top = df["s1_rank"] <= top_k_candidates
        top_df = df[is_top].copy()

        stage2_raw = self.predict_scores(top_df)

        # Min-max normalize stage 2 score per query
        top_df["s2_raw"] = stage2_raw
        s2_min = top_df.groupby("query_id")["s2_raw"].transform("min")
        s2_max = top_df.groupby("query_id")["s2_raw"].transform("max")
        s2_diff = s2_max - s2_min
        top_df["s2_norm"] = (top_df["s2_raw"] - s2_min) / np.where(s2_diff > 1e-9, s2_diff, 1.0)

        # Stage 1 + Stage 2 convex ensemble
        top_df["final_reranked_score"] = (
            (1.0 - alpha) * top_df["stage1_score"] + alpha * top_df["s2_norm"]
        )

        # Assign back to df
        df["final_reranked_score"] = df["stage1_score"] - 100.0  # default heavily discounted for > top_k
        df.loc[top_df.index, "final_reranked_score"] = top_df["final_reranked_score"].values

        return df

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves model weights and normalization statistics."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "hidden_dim": self.hidden_dim,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "feature_cols": self.FEATURE_COLS,
            "margin": self.margin,
        }
        torch.save(payload, str(filepath))

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PairwiseNeuralReRanker":
        """Loads trained re-ranker from file."""
        payload = torch.load(str(filepath), map_location="cpu", weights_only=False)
        instance = cls(
            hidden_dim=payload["hidden_dim"],
            margin=payload.get("margin", 1.0),
        )
        instance.model.load_state_dict(payload["state_dict"])
        instance.feature_means = payload["feature_means"]
        instance.feature_stds = payload["feature_stds"]
        instance.model.eval()
        return instance

