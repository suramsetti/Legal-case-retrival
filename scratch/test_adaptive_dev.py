import sys
from pathlib import Path
PROJECT_ROOT = Path("c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import torch
from src.ranking.pairwise_reranker import PairwiseNeuralReRanker
from src.evaluation.hybrid_evaluator import HybridEvaluator

data_dir = PROJECT_ROOT / "data/processed/IL-TUR-PCR"
dev_df = pd.read_parquet(data_dir / "dev/hybrid_features.parquet")
test_df = pd.read_parquet(data_dir / "test/hybrid_features.parquet")

# Engineer features
dev_df = PairwiseNeuralReRanker.engineer_features(dev_df)
test_df = PairwiseNeuralReRanker.engineer_features(test_df)

# Load checkpoint
reranker = PairwiseNeuralReRanker.load(PROJECT_ROOT / "models/checkpoints/pairwise_neural_reranker.pt")

# Base Stage 1
m_s1_dev = HybridEvaluator.evaluate_predictions(dev_df, score_col="stage1_score")
print(f"Dev S1 Baseline: P@1={m_s1_dev['P@1']:.4f}, MRR={m_s1_dev['MRR']:.4f}, nDCG@10={m_s1_dev['nDCG@10']:.4f}")

# Compute query margins on dev
def get_margins(df, score_col="stage1_score"):
    margins = {}
    for qid, grp in df.groupby("query_id"):
        scores = grp[score_col].nlargest(2).values
        if len(scores) > 1:
            margins[qid] = float(scores[0] - scores[1])
        else:
            margins[qid] = 1.0
    return margins

dev_margins = get_margins(dev_df)
m_s = pd.Series(dev_margins)
print("Dev Margin distribution:")
print(m_s.describe())

# Global Stage 2 reranking on Dev (alpha=0.5 and alpha=0.2)
dev_reranked_raw = reranker.rerank_query_pool(dev_df, top_k_candidates=30, alpha=0.5)
m_s2_dev_05 = HybridEvaluator.evaluate_predictions(dev_reranked_raw, score_col="final_reranked_score")
print(f"Dev S2 (Global, alpha=0.5): P@1={m_s2_dev_05['P@1']:.4f}, MRR={m_s2_dev_05['MRR']:.4f}")

dev_reranked_raw_02 = reranker.rerank_query_pool(dev_df, top_k_candidates=30, alpha=0.2)
m_s2_dev_02 = HybridEvaluator.evaluate_predictions(dev_reranked_raw_02, score_col="final_reranked_score")
print(f"Dev S2 (Global, alpha=0.2): P@1={m_s2_dev_02['P@1']:.4f}, MRR={m_s2_dev_02['MRR']:.4f}")

