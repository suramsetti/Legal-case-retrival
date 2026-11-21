"""
Phase 10 – Query-Level Error Analysis Script.

For each eligible test query (query with ≥1 positive candidate, excluding
zero-relevance queries) this script produces one row with Stage-1 vs Stage-2
rank diagnostics and feature-level differences between the true positive and
the Stage-1 false alarm.

Outputs:
  results/evaluation/phase10_query_error_analysis.csv
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.ranking.pairwise_reranker import PairwiseNeuralReRanker
from src.evaluation.hybrid_evaluator import HybridEvaluator

# ─── Constants ───────────────────────────────────────────────────
ZERO_RELEVANCE_QUERIES = {"0000431954", "0001508893", "0001864396"}
DATA_DIR = PROJECT_ROOT / "data/processed/IL-TUR-PCR"
CKPT_PATH = PROJECT_ROOT / "models/checkpoints/pairwise_neural_reranker.pt"
METRICS_JSON = PROJECT_ROOT / "results/evaluation/twostage_reranker_metrics.json"
OUT_CSV = PROJECT_ROOT / "results/evaluation/phase10_query_error_analysis.csv"

FEATURE_DIFF_COLS = [
    "tfidf_score",
    "inlegalbert_sim",
    "composite_legal_score",
    "rhetorical_cosine",
    "statute_shared_count",
    "citation_shared_count",
]


def load_and_engineer(split: str) -> pd.DataFrame:
    path = DATA_DIR / split / "hybrid_features.parquet"
    df = pd.read_parquet(path)
    df = PairwiseNeuralReRanker.engineer_features(df)
    return df


def compute_stage1_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Compute stage1_score and s1_rank (already done by engineer_features but rank col may be absent)."""
    df = df.copy()
    df["s1_rank"] = df.groupby("query_id")["stage1_score"].rank(
        ascending=False, method="first"
    )
    return df


def compute_stage2_rank(df_reranked: pd.DataFrame) -> pd.DataFrame:
    df_reranked = df_reranked.copy()
    df_reranked["s2_rank"] = df_reranked.groupby("query_id")["final_reranked_score"].rank(
        ascending=False, method="first"
    )
    return df_reranked


def main():
    print("=" * 65)
    print("Phase 10 – Query-Level Error Analysis")
    print("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────
    print("\n[1/5] Loading datasets …")
    train_df = pd.read_parquet(DATA_DIR / "train/hybrid_features.parquet")
    dev_df   = load_and_engineer("dev")
    test_df  = load_and_engineer("test")
    print(f"      Train={len(train_df):,} rows | Dev={len(dev_df):,} rows | Test={len(test_df):,} rows")

    # ── 2. Stage-1 ranks ─────────────────────────────────────────
    print("\n[2/5] Computing Stage-1 scores and ranks …")
    test_df = compute_stage1_rank(test_df)
    m_s1 = HybridEvaluator.evaluate_predictions(test_df, score_col="stage1_score")
    print(f"      Stage-1 Test: P@1={m_s1['P@1']:.4f}, MRR={m_s1['MRR']:.4f}")

    # ── 3. Stage-2 re-ranking ────────────────────────────────────
    print("\n[3/5] Loading reranker and computing Stage-2 scores …")
    reranker = PairwiseNeuralReRanker.load(CKPT_PATH)
    with open(METRICS_JSON) as f:
        saved = json.load(f)
    best_alpha = saved.get("best_alpha", 0.2)
    print(f"      Using best_alpha={best_alpha}")

    test_reranked = reranker.rerank_query_pool(test_df, top_k_candidates=30, alpha=best_alpha)
    test_reranked = compute_stage2_rank(test_reranked)
    m_s2 = HybridEvaluator.evaluate_predictions(test_reranked, score_col="final_reranked_score")
    print(f"      Stage-2 Test: P@1={m_s2['P@1']:.4f}, MRR={m_s2['MRR']:.4f}")

    # ── 4. Build query-level error rows ──────────────────────────
    print("\n[4/5] Building per-query error analysis rows …")

    rows = []
    n_promoted = 0
    n_demoted  = 0

    for qid, grp in test_reranked.groupby("query_id"):
        # Skip zero-relevance queries
        if str(qid) in ZERO_RELEVANCE_QUERIES:
            continue

        pos_rows = grp[grp["label"] == 1]
        if pos_rows.empty:
            continue  # no positives in pool → skip (evaluator also skips)

        # Best positive = the positive with the best (lowest) s1_rank
        best_pos = pos_rows.sort_values("s1_rank").iloc[0]
        s1_rank_of_best_pos = int(best_pos["s1_rank"])
        s2_rank_of_best_pos = int(best_pos["s2_rank"])

        s1_hit = s1_rank_of_best_pos == 1
        s2_hit = s2_rank_of_best_pos == 1

        promoted = (not s1_hit) and s2_hit
        demoted  = s1_hit and (not s2_hit)

        if promoted:
            n_promoted += 1
        if demoted:
            n_demoted += 1

        # Stage-1 top-1 candidate
        top1_row = grp[grp["s1_rank"] == 1].iloc[0]
        stage1_top1_score      = float(top1_row["stage1_score"])
        stage1_score_of_best_pos = float(best_pos["stage1_score"])
        margin_s1 = stage1_top1_score - stage1_score_of_best_pos

        s1_positive_candidate_id = best_pos["candidate_id"]

        # False alarm = top-1 candidate if it is NOT a positive
        if not s1_hit:
            s1_false_alarm_candidate_id = top1_row["candidate_id"]
            # Feature diffs: true_positive.feat - false_alarm.feat
            tfidf_diff         = float(best_pos["tfidf_score"])         - float(top1_row["tfidf_score"])
            inlegalbert_diff   = float(best_pos["inlegalbert_sim"])      - float(top1_row["inlegalbert_sim"])
            legal_diff         = float(best_pos["composite_legal_score"])- float(top1_row["composite_legal_score"])
            rhet_cosine_diff   = float(best_pos["rhetorical_cosine"])    - float(top1_row["rhetorical_cosine"])
            statute_diff       = float(best_pos["statute_shared_count"]) - float(top1_row["statute_shared_count"])
            citation_diff      = float(best_pos["citation_shared_count"])- float(top1_row["citation_shared_count"])
        else:
            s1_false_alarm_candidate_id = None
            tfidf_diff = inlegalbert_diff = legal_diff = None
            rhet_cosine_diff = statute_diff = citation_diff = None

        rows.append({
            "query_id":                    qid,
            "s1_rank_of_best_positive":    s1_rank_of_best_pos,
            "s2_rank_of_best_positive":    s2_rank_of_best_pos,
            "s1_hit":                      s1_hit,
            "s2_hit":                      s2_hit,
            "promoted":                    promoted,
            "demoted":                     demoted,
            "stage1_top1_score":           stage1_top1_score,
            "stage1_score_of_best_positive": stage1_score_of_best_pos,
            "margin_s1":                   margin_s1,
            "s1_positive_candidate_id":    s1_positive_candidate_id,
            "s1_false_alarm_candidate_id": s1_false_alarm_candidate_id,
            "tfidf_diff":                  tfidf_diff,
            "inlegalbert_diff":            inlegalbert_diff,
            "legal_diff":                  legal_diff,
            "rhetorical_cosine_diff":      rhet_cosine_diff,
            "statute_shared_count_diff":   statute_diff,
            "citation_shared_count_diff":  citation_diff,
        })

    error_df = pd.DataFrame(rows)

    # ── 5. Save & summarize ──────────────────────────────────────
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    error_df.to_csv(OUT_CSV, index=False)
    print(f"      Saved {len(error_df)} rows to {OUT_CSV}")

    # Summary
    net_gain = n_promoted - n_demoted
    print("\n" + "=" * 65)
    print("SUMMARY TABLE")
    print("=" * 65)
    print(f"Eligible test queries:     {len(error_df)}")
    print(f"Stage-1 P@1 hits:          {error_df['s1_hit'].sum()}  ({error_df['s1_hit'].mean()*100:.2f}%)")
    print(f"Stage-2 P@1 hits:          {error_df['s2_hit'].sum()}  ({error_df['s2_hit'].mean()*100:.2f}%)")
    print(f"Promoted to Rank-1:        {n_promoted}")
    print(f"Demoted from Rank-1:       {n_demoted}")
    print(f"Net gain:                  {net_gain:+d}")

    # Feature statistics for Stage-1 Rank-2 misses (33 queries)
    rank2_miss = error_df[
        (error_df["s1_rank_of_best_positive"] == 2) & (~error_df["s1_hit"])
    ]
    print(f"\nRank-2 miss queries (s1_rank==2 & s1_hit==False): {len(rank2_miss)}")
    if len(rank2_miss) > 0:
        feat_stats_cols = [
            "tfidf_diff", "inlegalbert_diff", "legal_diff",
            "rhetorical_cosine_diff", "statute_shared_count_diff", "citation_shared_count_diff",
            "margin_s1",
        ]
        avail_cols = [c for c in feat_stats_cols if c in rank2_miss.columns]
        print("\n  Feature statistics for Rank-2 miss queries (true_positive - false_alarm):")
        print(f"  {'Feature':<30s}  {'Mean':>8s}  {'Median':>8s}  {'Pos%':>6s}")
        print(f"  {'-'*60}")
        for col in avail_cols:
            col_data = rank2_miss[col].dropna()
            if len(col_data) == 0:
                continue
            mean_v   = col_data.mean()
            median_v = col_data.median()
            pos_pct  = (col_data > 0).mean() * 100
            print(f"  {col:<30s}  {mean_v:>+8.4f}  {median_v:>+8.4f}  {pos_pct:>5.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
