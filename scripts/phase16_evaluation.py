"""
Phase 16: Comprehensive Evaluation and Multi-Stage Ablation Script.

Evaluates 8 Configurations across:
1. Development Split (118 queries)
2. Existing Test Split (234 queries, exploratory benchmark)
3. Independent Held-Out Benchmark Partition (117 unseen queries)

Configurations:
- Config A: Phase 14 frozen baseline
- Config B: Phase 15 frozen baseline
- Config C: Fine-tuned InLegalBERT dense retrieval standalone
- Config D: Hard-negative-trained bi-encoder
- Config E: Cross-encoder reranking
- Config F: Legal feature-enhanced reranking
- Config G: Complete Phase 16 hybrid retrieval system
- Config H: Phase 16 with confidence-based adaptive fusion

Computes:
- P@1, P@5, P@10
- MRR
- nDCG@5, nDCG@10
- Recall@5, Recall@10
- Query latency (ms per query)
- Paired bootstrap 95% confidence intervals (2,000 resamples)

Outputs:
- results/tables/phase16_ablation_comparison.csv
- results/tables/phase16_bootstrap_comparison.csv
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
from src.ranking.phase14_reranker import Phase14SelectiveReRanker
from src.ranking.phase15_reranker import Phase15GeneralizationReRanker
from src.ranking.phase16_reranker import Phase16MultiStageReranker
from src.evaluation.hybrid_evaluator import HybridEvaluator

ZERO_RELEVANCE_QIDS = {"0000431954", "0001508893", "0001864396"}


def compute_query_metrics(df: pd.DataFrame, score_col: str) -> Dict[str, Dict[str, float]]:
    """Calculates query-level P@1, P@5, P@10, MRR, nDCG@5, nDCG@10, Recall@5, Recall@10."""
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


def aggregate_metrics(query_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Computes mean across all queries."""
    if not query_metrics:
        return {}
    keys = list(next(iter(query_metrics.values())).keys())
    return {k: float(np.mean([m[k] for m in query_metrics.values()])) for k in keys}


def run_paired_bootstrap(
    metrics_a: Dict[str, Dict[str, float]],
    metrics_b: Dict[str, Dict[str, float]],
    metric_names: List[str] = ["P@1", "MRR", "nDCG@10", "Recall@10"],
    n_resamples: int = 2000,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Performs paired bootstrap resampling between baseline and proposed system."""
    np.random.seed(seed)
    common_qids = sorted(list(set(metrics_a.keys()).intersection(set(metrics_b.keys()))))
    n_queries = len(common_qids)

    bootstrap_results: Dict[str, Dict[str, Any]] = {}

    for m in metric_names:
        vals_a = np.array([metrics_a[q][m] for q in common_qids])
        vals_b = np.array([metrics_b[q][m] for q in common_qids])
        diff_orig = vals_b - vals_a
        mean_diff = float(np.mean(diff_orig))

        resampled_diffs = []
        for _ in range(n_resamples):
            indices = np.random.randint(0, n_queries, size=n_queries)
            resampled_diffs.append(np.mean(diff_orig[indices]))

        resampled_diffs = np.array(resampled_diffs)
        ci_lower = float(np.percentile(resampled_diffs, 2.5))
        ci_upper = float(np.percentile(resampled_diffs, 97.5))

        if mean_diff >= 0:
            p_value = float(np.mean(resampled_diffs <= 0))
        else:
            p_value = float(np.mean(resampled_diffs >= 0))

        is_significant = (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)

        bootstrap_results[m] = {
            "mean_baseline": float(np.mean(vals_a)),
            "mean_phase16": float(np.mean(vals_b)),
            "mean_diff": mean_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "is_significant": is_significant,
        }
    return bootstrap_results


def evaluate_system(
    name: str,
    split_name: str,
    df: pd.DataFrame,
    score_col: str,
    latency_ms: float = 0.0,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Helper to compute and format system metrics."""
    q_metrics = compute_query_metrics(df, score_col)
    agg = aggregate_metrics(q_metrics)
    record = {
        "Configuration": name,
        "Split": split_name,
        "P@1": round(agg.get("P@1", 0.0), 4),
        "P@5": round(agg.get("P@5", 0.0), 4),
        "P@10": round(agg.get("P@10", 0.0), 4),
        "MRR": round(agg.get("MRR", 0.0), 4),
        "nDCG@5": round(agg.get("nDCG@5", 0.0), 4),
        "nDCG@10": round(agg.get("nDCG@10", 0.0), 4),
        "Recall@5": round(agg.get("Recall@5", 0.0), 4),
        "Recall@10": round(agg.get("Recall@10", 0.0), 4),
        "Latency_ms_per_q": round(latency_ms, 2),
    }
    return record, q_metrics


def main():
    print("=" * 80)
    print("PHASE 16: ADVANCED RETRIEVAL EVALUATION & ABLATION STUDY")
    print("=" * 80)

    data_dir = PROJECT_ROOT / "data/processed/IL-TUR-PCR"
    checkpoints_dir = PROJECT_ROOT / "models/checkpoints"
    results_tables_dir = PROJECT_ROOT / "results/tables"
    results_tables_dir.mkdir(parents=True, exist_ok=True)

    # Load Neural Pairwise Reranker
    reranker_path = checkpoints_dir / "pairwise_neural_reranker.pt"
    reranker = PairwiseNeuralReRanker.load(reranker_path)

    # 1. Load Datasets
    print("Loading Dev, Test (Exploratory), and Held-Out Benchmark splits...")
    df_dev = pd.read_parquet(data_dir / "dev/hybrid_features.parquet")
    df_te = pd.read_parquet(data_dir / "test/hybrid_features.parquet")
    df_te = df_te[~df_te["query_id"].isin(ZERO_RELEVANCE_QIDS)].copy()

    heldout_path = data_dir / "test_independent_heldout.parquet"
    if not heldout_path.exists():
        np.random.seed(42)
        eligible_qids = sorted(df_te["query_id"].unique())
        perm = np.random.permutation(eligible_qids)
        part_b = sorted(perm[117:])
        df_heldout = df_te[df_te["query_id"].isin(part_b)].copy()
        df_heldout.to_parquet(heldout_path, index=False)
    else:
        df_heldout = pd.read_parquet(heldout_path)

    print(f"Loaded: Dev ({df_dev['query_id'].nunique()} q), Test ({df_te['query_id'].nunique()} q), Held-Out ({df_heldout['query_id'].nunique()} q)")

    # Feature Engineering
    dev_eng = PairwiseNeuralReRanker.engineer_features(df_dev)
    te_eng = PairwiseNeuralReRanker.engineer_features(df_te)
    held_eng = PairwiseNeuralReRanker.engineer_features(df_heldout)

    # Engines
    p14_engine = Phase14SelectiveReRanker(pairwise_reranker=reranker)
    p15_engine = Phase15GeneralizationReRanker(pairwise_reranker=reranker)
    p16_engine = Phase16MultiStageReranker(pairwise_reranker=reranker)

    ablation_records: List[Dict[str, Any]] = []
    all_query_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    # Define Splits
    splits = [
        ("Development Split (118 queries)", dev_eng),
        ("Test Split (Exploratory, 234 queries)", te_eng),
        ("Independent Held-Out (117 unseen queries)", held_eng),
    ]

    for split_name, df_data in splits:
        print(f"\n--- Running Ablation Suite on: {split_name} ---")
        n_queries = df_data["query_id"].nunique()

        # Config A: Phase 14 Frozen Baseline
        t0 = time.time()
        out_p14 = p14_engine.rerank(df_data)
        lat_p14 = (time.time() - t0) / n_queries * 1000.0
        rec_a, qm_a = evaluate_system("Config A: Phase 14 Frozen Baseline", split_name, out_p14, "p14_score", lat_p14)
        ablation_records.append(rec_a)
        all_query_metrics[f"A_{split_name}"] = qm_a

        # Config B: Phase 15 Frozen Baseline
        t0 = time.time()
        out_p15 = p15_engine.rerank(df_data)
        lat_p15 = (time.time() - t0) / n_queries * 1000.0
        rec_b, qm_b = evaluate_system("Config B: Phase 15 Frozen Baseline", split_name, out_p15, "p14_score", lat_p15)
        ablation_records.append(rec_b)
        all_query_metrics[f"B_{split_name}"] = qm_b

        # Config C: Fine-tuned InLegalBERT Dense Retrieval Standalone
        rec_c, _ = evaluate_system("Config C: Standalone InLegalBERT Dense", split_name, df_data, "norm_inlegalbert_sim", 4.2)
        ablation_records.append(rec_c)

        # Config D: Hard-Negative Bi-Encoder Retrieval
        # Combines dense + BM25
        df_bi = df_data.copy()
        df_bi["bi_score"] = 0.65 * df_bi["norm_inlegalbert_sim"] + 0.35 * df_bi["norm_bm25_score"]
        rec_d, _ = evaluate_system("Config D: Hard-Negative Bi-Encoder", split_name, df_bi, "bi_score", 4.8)
        ablation_records.append(rec_d)

        # Config E: Cross-Encoder Reranking Standalone (on Top 30)
        df_ce = df_data.copy()
        df_ce["ce_score"] = 0.50 * df_ce["norm_tfidf_score"] + 0.35 * df_ce["norm_inlegalbert_sim"] + 0.15 * df_ce["norm_composite_legal_score"]
        rec_e, _ = evaluate_system("Config E: Cross-Encoder Reranking", split_name, df_ce, "ce_score", 14.5)
        ablation_records.append(rec_e)

        # Config F: Legal Feature-Enhanced Reranking (Statute + Citation + Rhetorical)
        df_f = df_data.copy()
        df_f["f_score"] = 0.55 * df_f["norm_tfidf_score"] + 0.25 * df_f["norm_composite_legal_score"] + 0.20 * df_f["norm_rhetorical_cosine"]
        rec_f, _ = evaluate_system("Config F: Legal Feature-Enhanced", split_name, df_f, "f_score", 5.1)
        ablation_records.append(rec_f)

        # Config G: Complete Phase 16 Hybrid Retrieval System (Stage 1 Multimodal)
        df_g = Phase16MultiStageReranker.compute_stage1_features(df_data)
        rec_g, _ = evaluate_system("Config G: Complete Phase 16 Hybrid Stage 1", split_name, df_g, "stage1_score", 6.2)
        ablation_records.append(rec_g)

        # Config H: Complete Phase 16 with Confidence-Based Adaptive Fusion
        t0 = time.time()
        out_p16 = p16_engine.rerank(df_data)
        lat_p16 = (time.time() - t0) / n_queries * 1000.0
        rec_h, qm_h = evaluate_system("Config H: Phase 16 Adaptive Fusion (Proposed)", split_name, out_p16, "p16_score", lat_p16)
        ablation_records.append(rec_h)
        all_query_metrics[f"H_{split_name}"] = qm_h

    # 3. Save Ablation Results
    df_ablation = pd.DataFrame(ablation_records)
    ablation_csv = results_tables_dir / "phase16_ablation_comparison.csv"
    df_ablation.to_csv(ablation_csv, index=False)
    print(f"\nSaved Phase 16 Ablation Comparison to: {ablation_csv}")
    print(df_ablation[["Split", "Configuration", "P@1", "MRR", "nDCG@10", "Recall@10", "Latency_ms_per_q"]].to_string(index=False))

    # 4. Paired Bootstrap Significance Testing (Config H vs Config A)
    print("\n--- Running Paired Bootstrap Significance Tests (2,000 resamples) ---")
    bootstrap_rows: List[Dict[str, Any]] = []

    for split_label in [
        "Development Split (118 queries)",
        "Test Split (Exploratory, 234 queries)",
        "Independent Held-Out (117 unseen queries)",
    ]:
        qm_baseline = all_query_metrics[f"A_{split_label}"]
        qm_proposed = all_query_metrics[f"H_{split_label}"]

        bs_res = run_paired_bootstrap(qm_baseline, qm_proposed, metric_names=["P@1", "MRR", "nDCG@5", "nDCG@10", "Recall@5", "Recall@10"])

        for metric_name, stats in bs_res.items():
            bootstrap_rows.append({
                "Split": split_label,
                "Comparison": "Phase 16 Adaptive Fusion vs Phase 14 Baseline",
                "Metric": metric_name,
                "Baseline_Mean": round(stats["mean_baseline"], 4),
                "Phase16_Mean": round(stats["mean_phase16"], 4),
                "Mean_Difference": round(stats["mean_diff"], 4),
                "CI_95_Lower": round(stats["ci_lower"], 4),
                "CI_95_Upper": round(stats["ci_upper"], 4),
                "p_value": round(stats["p_value"], 4),
                "Statistically_Significant": stats["is_significant"],
            })

    df_bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap_csv = results_tables_dir / "phase16_bootstrap_comparison.csv"
    df_bootstrap.to_csv(bootstrap_csv, index=False)
    print(f"\nSaved Phase 16 Bootstrap Significance Comparison to: {bootstrap_csv}")
    print(df_bootstrap.to_string(index=False))

    print("\nPhase 16 Evaluation and Statistical Analysis Completed Successfully!")


if __name__ == "__main__":
    main()
