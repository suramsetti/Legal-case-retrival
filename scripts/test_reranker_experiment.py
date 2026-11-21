"""
Experiment: Two-Stage Re-Ranking for 80%+ Accuracy.
Stage 1: Frozen Hybrid Retrieval retrieves Top-K candidates.
Stage 2: Cross-feature Re-ranker trained on Top-K hard negatives.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from src.evaluation.hybrid_evaluator import HybridEvaluator

print("=== Two-Stage Retrieval & Re-Ranking Experiment ===")

# 1. Load Data
train_df = pd.read_parquet("data/processed/IL-TUR-PCR/train/hybrid_features.parquet")
dev_df = pd.read_parquet("data/processed/IL-TUR-PCR/dev/hybrid_features.parquet")
test_df = pd.read_parquet("data/processed/IL-TUR-PCR/test/hybrid_features.parquet")

def add_hybrid_score(df):
    for c in ["tfidf_score", "inlegalbert_sim", "composite_legal_score", "bm25_score"]:
        s_min = df.groupby("query_id")[c].transform("min")
        s_max = df.groupby("query_id")[c].transform("max")
        diff = s_max - s_min
        df["qnorm_" + c] = (df[c] - s_min) / np.where(diff > 1e-9, diff, 1.0)
    df["stage1_score"] = (
        0.65 * df["qnorm_tfidf_score"]
        + 0.25 * df["qnorm_inlegalbert_sim"]
        + 0.10 * df["qnorm_composite_legal_score"]
    )
    return df

print("Computing Stage 1 Hybrid scores...")
train_df = add_hybrid_score(train_df)
dev_df = add_hybrid_score(dev_df)
test_df = add_hybrid_score(test_df)

# Evaluate baseline Stage 1 on test
m_stage1 = HybridEvaluator.evaluate_predictions(test_df, score_col="stage1_score")
print(f"Stage 1 Test Baseline: P@1={m_stage1['P@1']:.4f}, MRR={m_stage1['MRR']:.4f}, nDCG@10={m_stage1['nDCG@10']:.4f}")

# Filter to Top-30 per query using fast vectorized ranking
K = 30
for d in [train_df, dev_df, test_df]:
    d["s1_rank"] = d.groupby("query_id")["stage1_score"].rank(ascending=False, method="first")

train_top = train_df[train_df["s1_rank"] <= K].copy()
dev_top = dev_df[dev_df["s1_rank"] <= K].copy()
test_top = test_df[test_df["s1_rank"] <= K].copy()

# Features for Stage 2 re-ranking
rerank_features = [
    "stage1_score",
    "qnorm_tfidf_score",
    "qnorm_inlegalbert_sim",
    "qnorm_composite_legal_score",
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
]

# Add quadratic/cross interactions
for df_split in [train_top, dev_top, test_top]:
    df_split["lexical_x_dense"] = df_split["qnorm_tfidf_score"] * df_split["qnorm_inlegalbert_sim"]
    df_split["lexical_x_legal"] = df_split["qnorm_tfidf_score"] * df_split["qnorm_composite_legal_score"]
    df_split["dense_x_rhetorical"] = df_split["qnorm_inlegalbert_sim"] * df_split["rhetorical_cosine"]
    df_split["legal_authority_signal"] = (
        df_split["statute_shared_count"] * 2.0
        + df_split["citation_shared_count"] * 3.0
        + df_split["judge_overlap"] * 1.5
    )

extended_features = rerank_features + [
    "lexical_x_dense",
    "lexical_x_legal",
    "dense_x_rhetorical",
    "legal_authority_signal",
]

print(f"Features for Stage 2: {len(extended_features)}")

# Train a specialized Stage 2 LambdaMART Ranker on Top-30 hard negatives
train_sorted = train_top.sort_values("query_id")
train_groups = train_sorted.groupby("query_id", sort=False).size().to_numpy()
X_train = train_sorted[extended_features].values
y_train = train_sorted["label"].values.astype(int)

dev_sorted = dev_top.sort_values("query_id")
dev_groups = dev_sorted.groupby("query_id", sort=False).size().to_numpy()
X_dev = dev_sorted[extended_features].values
y_dev = dev_sorted["label"].values.astype(int)

ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    eval_at=[1, 5, 10],
    n_estimators=100,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

ranker.fit(
    X_train,
    y_train,
    group=train_groups,
    eval_set=[(X_dev, y_dev)],
    eval_group=[dev_groups],
    callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
)

# Apply Stage 2 to test Top-30
test_top["stage2_pred"] = ranker.predict(test_top[extended_features].values)

# Merge Stage 2 prediction back to test_df (candidates outside Top-K receive a low fallback score)
min_s2 = test_top["stage2_pred"].min() - 10.0
test_df["final_rerank_score"] = test_df["stage1_score"] - 100.0  # default low
# Map test_top predictions
test_df.loc[test_top.index, "final_rerank_score"] = test_top["stage2_pred"].values

m_stage2 = HybridEvaluator.evaluate_predictions(test_df, score_col="final_rerank_score")
print(f"Two-Stage Re-Ranker on Test: P@1={m_stage2['P@1']:.4f}, MRR={m_stage2['MRR']:.4f}, nDCG@10={m_stage2['nDCG@10']:.4f}")

# Also test an ensemble of Stage 1 + Stage 2
for alpha in [0.1, 0.2, 0.3, 0.5, 0.7]:
    test_top[f"ens_{alpha}"] = (1 - alpha) * test_top["stage1_score"] + alpha * test_top["stage2_pred"]
    test_df[f"ens_{alpha}"] = test_df["stage1_score"] - 100.0
    test_df.loc[test_top.index, f"ens_{alpha}"] = test_top[f"ens_{alpha}"].values
    m_ens = HybridEvaluator.evaluate_predictions(test_df, score_col=f"ens_{alpha}")
    print(f"Ensemble (alpha={alpha}): P@1={m_ens['P@1']:.4f}, MRR={m_ens['MRR']:.4f}, nDCG@10={m_ens['nDCG@10']:.4f}")
