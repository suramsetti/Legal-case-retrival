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

# Compute query margins
top1 = dev_df.groupby("query_id")["stage1_score"].transform("max")
def calc_top2(df):
    t2_dict = {}
    for qid, grp in df.groupby("query_id"):
        arr = grp["stage1_score"].nlargest(2).values
        t2_dict[qid] = arr[1] if len(arr) > 1 else arr[0]
    return df["query_id"].map(t2_dict)

dev_df["top1_s1"] = top1
dev_df["top2_s1"] = calc_top2(dev_df)
dev_df["margin"] = dev_df["top1_s1"] - dev_df["top2_s1"]

# Top-K slice
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

# Normalize rhetorical_cosine per query in top_df
r_min = top_df.groupby("query_id")["rhetorical_cosine"].transform("min")
r_max = top_df.groupby("query_id")["rhetorical_cosine"].transform("max")
r_diff = r_max - r_min
top_df["qn_rhet"] = (top_df["rhetorical_cosine"] - r_min) / np.where(r_diff > 1e-9, r_diff, 1.0)

# Also test Ratio Decidendi difference (rhetorical_ratio_diff is diff of RatioOfTheDecision)
# Invert ratio_diff so higher similarity = 1 - ratio_diff
top_df["ratio_sim"] = 1.0 - top_df["rhetorical_ratio_diff"]

dev_df["s2_norm"] = 0.0
dev_df["qn_rhet"] = 0.0
dev_df["ratio_sim"] = 0.0
dev_df.loc[top_df.index, "s2_norm"] = top_df["s2_norm"].values
dev_df.loc[top_df.index, "qn_rhet"] = top_df["qn_rhet"].values
dev_df.loc[top_df.index, "ratio_sim"] = top_df["ratio_sim"].values

print("\n--- Tuning Margin-Gated + Rhetorical Adjustment on Dev ---")
best_p1 = m_s1_dev["P@1"]
best_cfg = None

for tau in [0.05, 0.06, 0.07, 0.08]:
    low_margin_qids = dev_df[dev_df["margin"] < tau]["query_id"].unique()
    is_low_margin = dev_df["query_id"].isin(low_margin_qids)
    mask_rerank = is_low_margin & (dev_df["s1_rank"] <= K)
    mask_low_other = is_low_margin & (dev_df["s1_rank"] > K)
    
    for alpha in [0.08, 0.10, 0.12, 0.15]:
        for gamma in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]:
            score = dev_df["stage1_score"].copy()
            # Combine stage 1, pairwise neural, and rhetorical alignment
            s_comb = (
                (1.0 - alpha - gamma) * dev_df.loc[mask_rerank, "stage1_score"]
                + alpha * dev_df.loc[mask_rerank, "s2_norm"]
                + gamma * dev_df.loc[mask_rerank, "qn_rhet"]
            )
            score[mask_rerank] = s_comb
            score[mask_low_other] = dev_df.loc[mask_low_other, "stage1_score"] - 100.0
            
            dev_df["eval_score"] = score
            m = HybridEvaluator.evaluate_predictions(dev_df, score_col="eval_score")
            if m["P@1"] > best_p1 or (m["P@1"] == best_p1 and m["MRR"] > 0.8528):
                best_p1 = m["P@1"]
                best_cfg = (tau, alpha, gamma, m["P@1"], m["MRR"], m["nDCG@10"])
                print(f"tau={tau:.2f}, alpha={alpha:.2f}, gamma={gamma:.2f} -> P@1={m['P@1']:.4f}, MRR={m['MRR']:.4f}, nDCG@10={m['nDCG@10']:.4f}")

print("\nBest Dev Config:", best_cfg)

