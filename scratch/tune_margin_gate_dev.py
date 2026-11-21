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
dev_df = pd.read_parquet(data_dir / "dev/hybrid_features.parquet")
dev_df = PairwiseNeuralReRanker.engineer_features(dev_df)
reranker = PairwiseNeuralReRanker.load(PROJECT_ROOT / "models/checkpoints/pairwise_neural_reranker.pt")

m_s1_dev = HybridEvaluator.evaluate_predictions(dev_df, score_col="stage1_score")
print(f"Dev S1 Baseline: P@1={m_s1_dev['P@1']:.4f}, MRR={m_s1_dev['MRR']:.4f}, nDCG@10={m_s1_dev['nDCG@10']:.4f}")

# Compute query margins
top1 = dev_df.groupby("query_id")["stage1_score"].transform("max")
# top2 score
def calc_top2(df):
    t2_dict = {}
    for qid, grp in df.groupby("query_id"):
        arr = grp["stage1_score"].nlargest(2).values
        t2_dict[qid] = arr[1] if len(arr) > 1 else arr[0]
    return df["query_id"].map(t2_dict)

dev_df["top1_s1"] = top1
dev_df["top2_s1"] = calc_top2(dev_df)
dev_df["margin"] = dev_df["top1_s1"] - dev_df["top2_s1"]

# Compute Stage-2 scores for top-K candidates
dev_df["s1_rank"] = dev_df.groupby("query_id")["stage1_score"].rank(ascending=False, method="first")
K = 30
is_top = dev_df["s1_rank"] <= K
top_df = dev_df[is_top].copy()
s2_raw = reranker.predict_scores(top_df)
top_df["s2_raw"] = s2_raw
s2_min = top_df.groupby("query_id")["s2_raw"].transform("min")
s2_max = top_df.groupby("query_id")["s2_raw"].transform("max")
s2_diff = s2_max - s2_min
top_df["s2_norm"] = (top_df["s2_raw"] - s2_min) / np.where(s2_diff > 1e-9, s2_diff, 1.0)
dev_df["s2_norm"] = 0.0
dev_df.loc[top_df.index, "s2_norm"] = top_df["s2_norm"].values

# Test tau and alpha combinations on Dev
print("\n--- Tuning Margin-Gated Re-ranking on Dev ---")
results = []
for tau in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
    low_margin_qids = dev_df[dev_df["margin"] < tau]["query_id"].unique()
    pct_reranked = len(low_margin_qids) / 118.0 * 100.0
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.25]:
        # If margin >= tau: score = stage1_score
        # If margin < tau:
        #   if s1_rank <= K: score = (1 - alpha) * stage1_score + alpha * s2_norm
        #   else: score = stage1_score - 100.0
        is_low_margin = dev_df["query_id"].isin(low_margin_qids)
        score = dev_df["stage1_score"].copy()
        
        # apply reranker only in low margin top K
        mask_rerank = is_low_margin & (dev_df["s1_rank"] <= K)
        mask_low_other = is_low_margin & (dev_df["s1_rank"] > K)
        
        score[mask_rerank] = (1.0 - alpha) * dev_df.loc[mask_rerank, "stage1_score"] + alpha * dev_df.loc[mask_rerank, "s2_norm"]
        score[mask_low_other] = dev_df.loc[mask_low_other, "stage1_score"] - 100.0
        
        dev_df["eval_score"] = score
        m = HybridEvaluator.evaluate_predictions(dev_df, score_col="eval_score")
        results.append({
            "tau": tau,
            "alpha": alpha,
            "pct_reranked": pct_reranked,
            "P@1": m["P@1"],
            "MRR": m["MRR"],
            "nDCG@10": m["nDCG@10"]
        })
        if m["P@1"] >= m_s1_dev["P@1"]:
            print(f"tau={tau:.2f} ({pct_reranked:.1f}% q), alpha={alpha:.2f} -> P@1={m['P@1']:.4f}, MRR={m['MRR']:.4f}")

res_df = pd.DataFrame(results)
print("\nTop 5 Dev configurations by P@1, then MRR:")
print(res_df.sort_values(by=["P@1", "MRR"], ascending=[False, False]).head(5).to_string())

