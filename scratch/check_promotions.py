import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import torch
from src.ranking.pairwise_reranker import PairwiseNeuralReRanker

test_df = pd.read_parquet('data/processed/IL-TUR-PCR/test/hybrid_features.parquet')
test_df = PairwiseNeuralReRanker.engineer_features(test_df)
reranker = PairwiseNeuralReRanker.load('models/checkpoints/pairwise_neural_reranker.pt')

# Rerank with alpha=0.5
test_reranked = reranker.rerank_query_pool(test_df, top_k_candidates=30, alpha=0.5)

# Compare s1_rank vs reranked rank
test_df['s1_rank'] = test_df.groupby('query_id')['stage1_score'].rank(ascending=False, method='first')
test_reranked['s2_rank'] = test_reranked.groupby('query_id')['final_reranked_score'].rank(ascending=False, method='first')

s1_hits = 0
s2_hits = 0
promoted = 0
demoted = 0

for qid, grp in test_reranked.groupby('query_id'):
    pos = grp[grp['label'] == 1]
    if pos.empty:
        continue
    r1 = pos['s1_rank'].min()
    r2 = pos['s2_rank'].min()
    if r1 == 1:
        s1_hits += 1
    if r2 == 1:
        s2_hits += 1
    if r1 > 1 and r2 == 1:
        promoted += 1
    if r1 == 1 and r2 > 1:
        demoted += 1

print(f"Stage 1 Hits: {s1_hits}/234 ({s1_hits/234*100:.2f}%)")
print(f"Stage 2 Hits: {s2_hits}/234 ({s2_hits/234*100:.2f}%)")
print(f"Promoted to Rank 1: {promoted}")
print(f"Demoted from Rank 1: {demoted}")
print(f"Net change: {promoted - demoted:+d}")
