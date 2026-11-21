import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.ranking.pairwise_reranker import PairwiseNeuralReRanker
from src.evaluation.hybrid_evaluator import HybridEvaluator

dev_df = pd.read_parquet('data/processed/IL-TUR-PCR/dev/hybrid_features.parquet')
dev_df = PairwiseNeuralReRanker.engineer_features(dev_df)
m_s1 = HybridEvaluator.evaluate_predictions(dev_df, score_col='stage1_score')
print('Stage 1 Dev:', {k: round(v, 4) for k, v in m_s1.items() if isinstance(v, float)})

reranker = PairwiseNeuralReRanker.load('models/checkpoints/pairwise_neural_reranker.pt')
for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.65, 0.8, 1.0]:
    reranked = reranker.rerank_query_pool(dev_df, top_k_candidates=30, alpha=alpha)
    m = HybridEvaluator.evaluate_predictions(reranked, score_col='final_reranked_score')
    print(f"alpha={alpha:.2f}: P@1={m['P@1']:.4f}, MRR={m['MRR']:.4f}, nDCG@10={m['nDCG@10']:.4f}")
