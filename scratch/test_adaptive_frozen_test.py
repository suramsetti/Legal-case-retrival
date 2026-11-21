import sys
from pathlib import Path
PROJECT_ROOT = Path("c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.ranking.pairwise_reranker import PairwiseNeuralReRanker
from src.evaluation.hybrid_evaluator import HybridEvaluator

data_dir = PROJECT_ROOT / "data/processed/IL-TUR-PCR"
test_df = pd.read_parquet(data_dir / "test/hybrid_features.parquet")
test_df = PairwiseNeuralReRanker.engineer_features(test_df)
reranker = PairwiseNeuralReRanker.load(PROJECT_ROOT / "models/checkpoints/pairwise_neural_reranker.pt")

m_s1_test = HybridEvaluator.evaluate_predictions(test_df, score_col="stage1_score")
print(f"Test S1 Baseline: P@1={m_s1_test['P@1']:.4f}, MRR={m_s1_test['MRR']:.4f}, nDCG@10={m_s1_test['nDCG@10']:.4f}")

# Compute query margins on test
top1 = test_df.groupby("query_id")["stage1_score"].transform("max")
def calc_top2(df):
    t2_dict = {}
    for qid, grp in df.groupby("query_id"):
        arr = grp["stage1_score"].nlargest(2).values
        t2_dict[qid] = arr[1] if len(arr) > 1 else arr[0]
    return df["query_id"].map(t2_dict)

test_df["top1_s1"] = top1
test_df["top2_s1"] = calc_top2(test_df)
test_df["margin"] = test_df["top1_s1"] - test_df["top2_s1"]

# Top-K slice
test_df["s1_rank"] = test_df.groupby("query_id")["stage1_score"].rank(ascending=False, method="first")
K = 30
is_top = test_df["s1_rank"] <= K
top_df = test_df[is_top].copy()
s2_raw = reranker.predict_scores(top_df)
top_df["s2_raw"] = s2_raw
s2_min = top_df.groupby("query_id")["s2_raw"].transform("min")
s2_max = top_df.groupby("query_id")["s2_raw"].transform("max")
s2_diff = s2_max - s2_min
top_df["s2_norm"] = (top_df["s2_raw"] - s2_min) / np.where(s2_diff > 1e-9, s2_diff, 1.0)

r_min = top_df.groupby("query_id")["rhetorical_cosine"].transform("min")
r_max = top_df.groupby("query_id")["rhetorical_cosine"].transform("max")
r_diff = r_max - r_min
top_df["qn_rhet"] = (top_df["rhetorical_cosine"] - r_min) / np.where(r_diff > 1e-9, r_diff, 1.0)

test_df["s2_norm"] = 0.0
test_df["qn_rhet"] = 0.0
test_df.loc[top_df.index, "s2_norm"] = top_df["s2_norm"].values
test_df.loc[top_df.index, "qn_rhet"] = top_df["qn_rhet"].values

# Test frozen dev config: tau=0.07, alpha=0.12, gamma=0.04
tau = 0.07
alpha = 0.12
gamma = 0.04

low_margin_qids = test_df[test_df["margin"] < tau]["query_id"].unique()
is_low_margin = test_df["query_id"].isin(low_margin_qids)
mask_rerank = is_low_margin & (test_df["s1_rank"] <= K)
mask_low_other = is_low_margin & (test_df["s1_rank"] > K)

score = test_df["stage1_score"].copy()
score[mask_rerank] = (
    (1.0 - alpha - gamma) * test_df.loc[mask_rerank, "stage1_score"]
    + alpha * test_df.loc[mask_rerank, "s2_norm"]
    + gamma * test_df.loc[mask_rerank, "qn_rhet"]
)
score[mask_low_other] = test_df.loc[mask_low_other, "stage1_score"] - 100.0
test_df["final_adaptive_score"] = score

m_adapt = HybridEvaluator.evaluate_predictions(test_df, score_col="final_adaptive_score")
print(f"Test Adaptive Reranking: P@1={m_adapt['P@1']:.4f}, MRR={m_adapt['MRR']:.4f}, nDCG@10={m_adapt['nDCG@10']:.4f}")
print(f"Percentage of test queries reranked: {len(low_margin_qids)} / 234 = {len(low_margin_qids)/234*100:.1f}%")

# Promotion / demotion count
test_df["s1_r"] = test_df.groupby("query_id")["stage1_score"].rank(ascending=False, method="first")
test_df["s_adapt_r"] = test_df.groupby("query_id")["final_adaptive_score"].rank(ascending=False, method="first")

promoted = 0
demoted = 0
for qid, grp in test_df.groupby("query_id"):
    pos = grp[grp["label"] == 1]
    if pos.empty:
        continue
    r1 = pos["s1_r"].min()
    r2 = pos["s_adapt_r"].min()
    if r1 > 1 and r2 == 1:
        promoted += 1
    if r1 == 1 and r2 > 1:
        demoted += 1

print(f"Promoted to Rank 1: {promoted}")
print(f"Demoted from Rank 1: {demoted}")
print(f"Net change: {promoted - demoted:+d}")

