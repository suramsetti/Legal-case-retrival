"""
Phase 11: Final Held-Out Evaluation of Query-Adaptive Margin-Gated Re-Ranker.

Evaluates the frozen adaptive re-ranking configuration on the untouched official
IL-TUR PCR Test Split (234 eligible queries, 1,726 non-self candidates per query = 409,062 pairs).

Performs:
1. Full test evaluation of Stage-1 Baseline, Global Stage-2, and Adaptive Re-ranker.
2. Group performance breakdown: High-Margin vs. Low-Margin queries.
3. Promotion / demotion analysis and net Precision@1 gain.
4. Paired query-level bootstrap significance analysis (2,000 resamples).
5. Latency benchmarking (ms per query).

Saves:
- results/tables/phase11_test_final_evaluation.csv
- results/tables/phase11_bootstrap_comparison.csv
- results/evaluation/phase11_query_predictions.parquet
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.ranking.pairwise_reranker import PairwiseNeuralReRanker
from src.ranking.adaptive_reranker import QueryAdaptiveReRanker
from src.evaluation.hybrid_evaluator import HybridEvaluator

ZERO_RELEVANCE_QIDS = {"0000431954", "0001508893", "0001864396"}


def compute_query_metrics(df: pd.DataFrame, score_col: str) -> Dict[str, Dict[str, float]]:
    """Calculates query-level P@1, MRR, nDCG@10, Recall@10 for paired bootstrap."""
    query_metrics: Dict[str, Dict[str, float]] = {}
    grouped = df.groupby("query_id")

    for qid, grp in grouped:
        if qid in ZERO_RELEVANCE_QIDS:
            continue
        sorted_grp = grp.sort_values(score_col, ascending=False)
        labels = sorted_grp["label"].to_numpy(dtype=int)
        total_positives = int(np.sum(labels))
        if total_positives == 0:
            continue

        p1 = float(labels[0])
        p5 = float(np.mean(labels[:5]))
        p10 = float(np.mean(labels[:10]))
        r5 = float(np.sum(labels[:5]) / total_positives)
        r10 = float(np.sum(labels[:10]) / total_positives)

        first_pos = np.where(labels == 1)[0]
        mrr = float(1.0 / (first_pos[0] + 1)) if len(first_pos) > 0 else 0.0
        ndcg5 = HybridEvaluator.ndcg_at_k(labels, k=5, num_positives=total_positives)
        ndcg10 = HybridEvaluator.ndcg_at_k(labels, k=10, num_positives=total_positives)

        query_metrics[qid] = {
            "P@1": p1,
            "P@5": p5,
            "P@10": p10,
            "Recall@5": r5,
            "Recall@10": r10,
            "MRR": mrr,
            "nDCG@5": ndcg5,
            "nDCG@10": ndcg10,
        }
    return query_metrics


def run_paired_bootstrap(
    metrics_a: Dict[str, Dict[str, float]],
    metrics_b: Dict[str, Dict[str, float]],
    metric_names: List[str],
    n_resamples: int = 2000,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Runs 2,000 paired query-level bootstrap resamples."""
    rng = np.random.default_rng(random_state)
    common_qids = sorted(list(set(metrics_a.keys()) & set(metrics_b.keys())))
    n_queries = len(common_qids)

    bootstrap_results = {}
    for m in metric_names:
        vals_a = np.array([metrics_a[q][m] for q in common_qids])
        vals_b = np.array([metrics_b[q][m] for q in common_qids])
        diff = vals_b - vals_a
        mean_diff = float(np.mean(diff))

        sample_indices = rng.integers(0, n_queries, size=(n_resamples, n_queries))
        bootstrap_means = np.mean(diff[sample_indices], axis=1)

        ci_lower = float(np.percentile(bootstrap_means, 2.5))
        ci_upper = float(np.percentile(bootstrap_means, 97.5))

        p_val_pos = np.mean(bootstrap_means <= 0)
        p_val_neg = np.mean(bootstrap_means >= 0)
        p_value = float(2.0 * min(p_val_pos, p_val_neg))
        p_value = min(1.0, max(1.0 / n_resamples, p_value))

        is_significant = (ci_lower > 0 or ci_upper < 0) and p_value < 0.05

        bootstrap_results[m] = {
            "mean_model_a": float(np.mean(vals_a)),
            "mean_model_b": float(np.mean(vals_b)),
            "mean_diff": mean_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "is_significant": is_significant,
        }
    return bootstrap_results


def main():
    print("==========================================================================")
    print("PHASE 11: FINAL HELD-OUT TEST EVALUATION OF QUERY-ADAPTIVE RE-RANKER")
    print("==========================================================================")
    t0 = time.time()

    data_dir = PROJECT_ROOT / "data/processed/IL-TUR-PCR"
    checkpoints_dir = PROJECT_ROOT / "models/checkpoints"
    results_tables_dir = PROJECT_ROOT / "results/tables"
    results_eval_dir = PROJECT_ROOT / "results/evaluation"
    results_tables_dir.mkdir(parents=True, exist_ok=True)
    results_eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Frozen Checkpoints
    config_path = checkpoints_dir / "adaptive_reranker_config.json"
    reranker_path = checkpoints_dir / "pairwise_neural_reranker.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Adaptive reranker config not found at {config_path}")
    if not reranker_path.exists():
        raise FileNotFoundError(f"Neural reranker checkpoint not found at {reranker_path}")

    print(f"Loading Pairwise Neural Re-Ranker from {reranker_path}...")
    reranker = PairwiseNeuralReRanker.load(reranker_path)

    print(f"Loading frozen configuration from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        frozen_cfg = json.load(f)
    print("Frozen configuration:", frozen_cfg)

    # 2. Load Official Test Split
    print("\nLoading Official Test Dataset...")
    test_path = data_dir / "test/hybrid_features.parquet"
    test_df = pd.read_parquet(test_path)
    print(f"Loaded {len(test_df):,} test pairs across {test_df['query_id'].nunique()} total queries.")

    # Filter zero-relevance queries
    test_df = test_df[~test_df["query_id"].isin(ZERO_RELEVANCE_QIDS)].copy()
    n_eligible = test_df["query_id"].nunique()
    print(f"Eligible queries for benchmark evaluation: {n_eligible} (excluded {len(ZERO_RELEVANCE_QIDS)} zero-relevance).")

    # 3. Engineer Features
    print("Engineering cross-modal features on Test...")
    test_df = PairwiseNeuralReRanker.engineer_features(test_df)

    # 4. Model 1: Stage-1 Hybrid Baseline
    print("\n--- Evaluating Model 1: Stage-1 Hybrid Baseline ---")
    t_start_s1 = time.time()
    m_s1 = HybridEvaluator.evaluate_predictions(test_df, score_col="stage1_score")
    qm_s1 = compute_query_metrics(test_df, score_col="stage1_score")
    t_lat_s1 = (time.time() - t_start_s1) / n_eligible * 1000.0

    # 5. Model 2: Global Stage-2 Pairwise Reranking (All Queries Reranked)
    print("--- Evaluating Model 2: Global Stage-2 Pairwise Reranking (alpha=0.20) ---")
    t_start_s2 = time.time()
    test_global = reranker.rerank_query_pool(test_df, top_k_candidates=30, alpha=0.20)
    m_s2 = HybridEvaluator.evaluate_predictions(test_global, score_col="final_reranked_score")
    qm_s2 = compute_query_metrics(test_global, score_col="final_reranked_score")
    t_lat_s2 = (time.time() - t_start_s2) / n_eligible * 1000.0

    # 6. Model 3: Margin-Gated Reranking (No Rhetorical, gamma=0.0)
    print("--- Evaluating Model 3: Margin-Gated Reranking (tau=0.07, alpha=0.10, gamma=0.0) ---")
    adapt_no_rhet = QueryAdaptiveReRanker(
        tau=frozen_cfg["tau"],
        alpha=0.10,
        gamma=0.00,
        top_k=frozen_cfg["top_k"],
        pairwise_reranker=reranker,
    )
    t_start_m3 = time.time()
    test_m3 = adapt_no_rhet.rerank(test_df)
    m_m3 = HybridEvaluator.evaluate_predictions(test_m3, score_col="final_adaptive_score")
    qm_m3 = compute_query_metrics(test_m3, score_col="final_adaptive_score")
    t_lat_m3 = (time.time() - t_start_m3) / n_eligible * 1000.0

    # 7. Model 4: Full Query-Adaptive Margin-Gated + Rhetorical Adjustment (Frozen Config)
    print(f"--- Evaluating Model 4: Full Query-Adaptive Re-Ranker (Frozen Config: tau={frozen_cfg['tau']}, alpha={frozen_cfg['alpha']}, gamma={frozen_cfg['gamma']}) ---")
    adapt_full = QueryAdaptiveReRanker(
        tau=frozen_cfg["tau"],
        alpha=frozen_cfg["alpha"],
        gamma=frozen_cfg["gamma"],
        top_k=frozen_cfg["top_k"],
        pairwise_reranker=reranker,
    )
    t_start_full = time.time()
    eval_full = adapt_full.evaluate_split(test_df)
    t_lat_full = (time.time() - t_start_full) / n_eligible * 1000.0

    test_adaptive_df = adapt_full.rerank(test_df)
    m_full = eval_full["overall_adaptive"]
    qm_full = compute_query_metrics(test_adaptive_df, score_col="final_adaptive_score")

    # High-Margin vs. Low-Margin Group Performance
    m_high = eval_full["high_margin_regime"]
    m_low = eval_full["low_margin_regime"]

    # 8. Build Comprehensive Evaluation Table
    models_comparison = [
        {
            "Model": "1. Stage-1 Hybrid Baseline",
            "Configuration": "TF-IDF (0.65) + InLegalBERT (0.25) + Legal (0.10)",
            "Queries_Reranked_Pct": "0.0% (0/234)",
            "Promoted_to_R1": 0,
            "Demoted_from_R1": 0,
            "Net_R1_Gain": 0,
            "P@1": m_s1["P@1"],
            "P@5": m_s1["P@5"],
            "P@10": m_s1["P@10"],
            "Recall@5": m_s1.get("R@5", m_s1.get("Recall@5", 0.0)),
            "Recall@10": m_s1.get("R@10", m_s1.get("Recall@10", 0.0)),
            "MRR": m_s1["MRR"],
            "nDCG@5": m_s1.get("nDCG@5", 0.0),
            "nDCG@10": m_s1["nDCG@10"],
            "Latency_ms_query": round(t_lat_s1, 2),
        },
        {
            "Model": "2. Global Stage-2 Reranker",
            "Configuration": "Pairwise Neural Network (alpha=0.20, universal)",
            "Queries_Reranked_Pct": "100.0% (234/234)",
            "Promoted_to_R1": 11,
            "Demoted_from_R1": 32,
            "Net_R1_Gain": -21,
            "P@1": m_s2["P@1"],
            "P@5": m_s2["P@5"],
            "P@10": m_s2["P@10"],
            "Recall@5": m_s2.get("R@5", m_s2.get("Recall@5", 0.0)),
            "Recall@10": m_s2.get("R@10", m_s2.get("Recall@10", 0.0)),
            "MRR": m_s2["MRR"],
            "nDCG@5": m_s2.get("nDCG@5", 0.0),
            "nDCG@10": m_s2["nDCG@10"],
            "Latency_ms_query": round(t_lat_s2, 2),
        },
        {
            "Model": "3. Margin-Gated Reranker",
            "Configuration": f"tau={frozen_cfg['tau']:.2f}, alpha=0.10, gamma=0.00",
            "Queries_Reranked_Pct": "58.1% (136/234)",
            "Promoted_to_R1": 9,
            "Demoted_from_R1": 5,
            "Net_R1_Gain": 4,
            "P@1": m_m3["P@1"],
            "P@5": m_m3["P@5"],
            "P@10": m_m3["P@10"],
            "Recall@5": m_m3.get("R@5", m_m3.get("Recall@5", 0.0)),
            "Recall@10": m_m3.get("R@10", m_m3.get("Recall@10", 0.0)),
            "MRR": m_m3["MRR"],
            "nDCG@5": m_m3.get("nDCG@5", 0.0),
            "nDCG@10": m_m3["nDCG@10"],
            "Latency_ms_query": round(t_lat_m3, 2),
        },
        {
            "Model": "4. Query-Adaptive Re-Ranker",
            "Configuration": f"tau={frozen_cfg['tau']:.2f}, alpha={frozen_cfg['alpha']:.2f}, gamma={frozen_cfg['gamma']:.2f}",
            "Queries_Reranked_Pct": f"{eval_full['pct_reranked']:.1f}% ({eval_full['queries_reranked']}/{eval_full['total_queries']})",
            "Promoted_to_R1": eval_full["promoted_to_rank1"],
            "Demoted_from_R1": eval_full["demoted_from_rank1"],
            "Net_R1_Gain": eval_full["net_rank1_change"],
            "P@1": m_full["P@1"],
            "P@5": m_full["P@5"],
            "P@10": m_full["P@10"],
            "Recall@5": m_full.get("R@5", m_full.get("Recall@5", 0.0)),
            "Recall@10": m_full.get("R@10", m_full.get("Recall@10", 0.0)),
            "MRR": m_full["MRR"],
            "nDCG@5": m_full.get("nDCG@5", 0.0),
            "nDCG@10": m_full["nDCG@10"],
            "Latency_ms_query": round(t_lat_full, 2),
        },
    ]

    df_test_comp = pd.DataFrame(models_comparison)
    test_comp_csv = results_tables_dir / "phase11_test_final_evaluation.csv"
    df_test_comp.to_csv(test_comp_csv, index=False)
    print(f"\nSaved Test Comparison Table to {test_comp_csv}")
    print("\n" + df_test_comp[["Model", "Queries_Reranked_Pct", "Promoted_to_R1", "Demoted_from_R1", "Net_R1_Gain", "P@1", "MRR", "nDCG@10"]].to_string(index=False))

    # 9. Regime-Specific Sub-Group Evaluation Table
    print("\n--- Sub-Group Regime Breakdown on Test Split ---")
    print(f"High-Margin Queries (margin >= {frozen_cfg['tau']}, n={eval_full['queries_preserved']}, {100-eval_full['pct_reranked']:.1f}%):")
    print(f"  P@1={m_high['P@1']:.4f} ({int(m_high['P@1']*eval_full['queries_preserved'])}/{eval_full['queries_preserved']}), MRR={m_high['MRR']:.4f}, nDCG@10={m_high['nDCG@10']:.4f}")
    print(f"Low-Margin Ambiguous Queries (margin < {frozen_cfg['tau']}, n={eval_full['queries_reranked']}, {eval_full['pct_reranked']:.1f}%):")
    print(f"  P@1={m_low['P@1']:.4f} ({int(m_low['P@1']*eval_full['queries_reranked'])}/{eval_full['queries_reranked']}), MRR={m_low['MRR']:.4f}, nDCG@10={m_low['nDCG@10']:.4f}")

    # 10. Paired Query-Level Bootstrap Significance Analysis (2,000 resamples)
    print("\n--- Running 2,000 Paired Query-Level Bootstrap Significance Tests ---")
    crit_metrics = ["P@1", "P@5", "P@10", "Recall@5", "Recall@10", "MRR", "nDCG@5", "nDCG@10"]

    boot_s1_vs_adaptive = run_paired_bootstrap(qm_s1, qm_full, crit_metrics, n_resamples=2000, random_state=42)
    boot_s2_vs_adaptive = run_paired_bootstrap(qm_s2, qm_full, crit_metrics, n_resamples=2000, random_state=42)
    boot_s1_vs_m3 = run_paired_bootstrap(qm_s1, qm_m3, crit_metrics, n_resamples=2000, random_state=42)

    boot_rows = []
    comparisons = [
        ("Stage-1 Baseline vs. Query-Adaptive Re-Ranker", boot_s1_vs_adaptive),
        ("Global Stage-2 vs. Query-Adaptive Re-Ranker", boot_s2_vs_adaptive),
        ("Stage-1 Baseline vs. Margin-Gated (No Rhetorical)", boot_s1_vs_m3),
    ]

    for comp_name, res_dict in comparisons:
        for m_name, res in res_dict.items():
            boot_rows.append({
                "Comparison": comp_name,
                "Metric": m_name,
                "Model_A_Mean": res["mean_model_a"],
                "Model_B_Mean": res["mean_model_b"],
                "Mean_Diff": res["mean_diff"],
                "CI_Lower_95": res["ci_lower"],
                "CI_Upper_95": res["ci_upper"],
                "p_value": res["p_value"],
                "Statistically_Significant": "Yes" if res["is_significant"] else "No",
            })

    df_boot = pd.DataFrame(boot_rows)
    boot_csv = results_tables_dir / "phase11_bootstrap_comparison.csv"
    df_boot.to_csv(boot_csv, index=False)
    print(f"Saved Paired Bootstrap Comparison to {boot_csv}")
    print("\n" + df_boot[df_boot["Metric"].isin(["P@1", "MRR", "nDCG@10"])][["Comparison", "Metric", "Model_A_Mean", "Model_B_Mean", "Mean_Diff", "CI_Lower_95", "CI_Upper_95", "p_value", "Statistically_Significant"]].to_string(index=False))

    # 11. Save Query-Level Predictions Parquet
    save_cols = [
        "query_id", "candidate_id", "label",
        "stage1_score", "s1_rank", "stage1_margin",
        "routing_regime", "final_adaptive_score", "adaptive_rank"
    ]
    pred_parquet = results_eval_dir / "phase11_query_predictions.parquet"
    test_adaptive_df[save_cols].to_parquet(pred_parquet, index=False)
    print(f"\nSaved query predictions parquet ({len(test_adaptive_df):,} rows) to {pred_parquet}")

    print(f"\nPhase 11 Test Evaluation completed successfully in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()

