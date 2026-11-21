"""
Phase 12: Comprehensive Evaluation of Legal Cross-Encoder and Rhetorical-Aware Neural Reranking.

Evaluates and compares 4 required architectures:
1. Phase 11 Frozen Baseline (Margin-Gated Pairwise Re-ranker: tau=0.07, alpha=0.12, gamma=0.04).
2. Cross-Encoder Only (Stand-alone Cross-Encoder ranking on top candidates).
3. Rhetorical-Aware Cross-Encoder (Cross-Encoder with section-aware chunking and discourse alignment).
4. Phase 11 Adaptive Plus Cross-Encoder (Full hybrid: Stage-1 + Margin-Gated Cross-Encoder + Rhetorical Adjustment).

Outputs:
- results/tables/phase12_dev_experiments.csv
- results/tables/phase12_test_final_evaluation.csv
- results/tables/phase12_bootstrap_comparison.csv
- models/checkpoints/phase12_adaptive_config.json
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.ranking.legal_cross_encoder import InLegalBERTCrossEncoder, SectionAwareChunker
from src.ranking.adaptive_cross_reranker import AdaptiveCrossReranker
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


def precompute_cross_encoder_scores_batched(
    df: pd.DataFrame,
    q_docs: pd.DataFrame,
    c_docs: pd.DataFrame,
    cross_encoder: InLegalBERTCrossEncoder,
    chunker: SectionAwareChunker,
    top_n: int = 5,
    aggregation: str = "weighted",
    batch_size: int = 32,
) -> Dict[Tuple[str, str], float]:
    """
    Batched inference across all candidates in top_n for high throughput on CPU.
    """
    top_df = df[df["s1_rank"] <= top_n]
    print(f"Precomputing Cross-Encoder scores for {len(top_df)} candidate pairs (top-{top_n})...")
    t0 = time.time()

    candidate_chunks_map: List[Tuple[str, str, List[Dict[str, Any]]]] = []
    flat_queries: List[str] = []
    flat_chunks: List[str] = []
    chunk_counts: List[int] = []
    chunk_weights_list: List[np.ndarray] = []

    for _, row in top_df.iterrows():
        qid = row["query_id"]
        cid = row["candidate_id"]

        if qid not in q_docs.index or cid not in c_docs.index:
            candidate_chunks_map.append((qid, cid, []))
            chunk_counts.append(0)
            chunk_weights_list.append(np.array([1.0]))
            continue

        q_text = str(q_docs.loc[qid, "full_text"])[:500]
        c_text = str(c_docs.loc[cid, "full_text"])
        chunks = chunker.chunk_document(c_text)
        if not chunks:
            chunks = [{"section_type": "full", "text": c_text[:500], "rhetorical_weight": 1.0}]

        candidate_chunks_map.append((qid, cid, chunks))
        chunk_counts.append(len(chunks))
        w = np.array([c.get("rhetorical_weight", 1.0) for c in chunks], dtype=np.float32)
        chunk_weights_list.append(w / np.sum(w))

        for c in chunks:
            flat_queries.append(q_text)
            flat_chunks.append(c["text"])

    print(f"Batched tokenization & forward pass over {len(flat_chunks)} total chunks...")
    # Batched score_pairs
    raw_scores = cross_encoder.score_pairs(flat_queries, flat_chunks, max_length=160, batch_size=batch_size)

    # Re-aggregate per candidate
    scores_dict: Dict[Tuple[str, str], float] = {}
    offset = 0
    for idx, (qid, cid, chunks) in enumerate(candidate_chunks_map):
        n_c = chunk_counts[idx]
        if n_c == 0:
            scores_dict[(qid, cid)] = 0.0
            continue

        c_scores = raw_scores[offset : offset + n_c]
        offset += n_c

        if aggregation == "max":
            agg_val = float(np.max(c_scores))
        elif aggregation == "mean":
            agg_val = float(np.mean(c_scores))
        elif aggregation == "weighted":
            w = chunk_weights_list[idx]
            agg_val = float(np.sum(c_scores * w))
        else:
            agg_val = float(np.max(c_scores))

        scores_dict[(qid, cid)] = agg_val

    elapsed = time.time() - t0
    print(f"Completed {len(scores_dict)} pair aggregations in {elapsed:.2f}s ({len(scores_dict)/max(1, elapsed):.1f} pairs/sec).")
    return scores_dict


def main():
    print("==========================================================================")
    print("PHASE 12: LEGAL CROSS-ENCODER AND RHETORICAL-AWARE NEURAL RERANKING")
    print("==========================================================================")
    t0 = time.time()

    data_dir = PROJECT_ROOT / "data/processed/IL-TUR-PCR"
    checkpoints_dir = PROJECT_ROOT / "models/checkpoints"
    results_tables_dir = PROJECT_ROOT / "results/tables"
    results_tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Pre-trained Models
    ce_path = checkpoints_dir / "legal_cross_encoder.pt"
    if not ce_path.exists():
        raise FileNotFoundError(f"Cross-encoder checkpoint not found at {ce_path}.")

    print(f"Loading InLegalBERTCrossEncoder from {ce_path}...")
    device = "cpu"
    cross_encoder = InLegalBERTCrossEncoder(model_name_or_path="law-ai/InLegalBERT", device=device)
    cross_encoder.load(ce_path)
    cross_encoder.eval()

    chunker = SectionAwareChunker()

    # Load Phase 11 Models
    p11_reranker = PairwiseNeuralReRanker.load(checkpoints_dir / "pairwise_neural_reranker.pt")
    p11_config = json.load(open(checkpoints_dir / "adaptive_reranker_config.json"))
    phase11_model = QueryAdaptiveReRanker(
        tau=p11_config["tau"],
        alpha=p11_config["alpha"],
        gamma=p11_config["gamma"],
        top_k=p11_config["top_k"],
        pairwise_reranker=p11_reranker,
    )

    # 2. Load Development Dataset and Texts
    print("\nLoading Development dataset and texts...")
    dev_df = pd.read_parquet(data_dir / "dev/hybrid_features.parquet")
    dev_df = PairwiseNeuralReRanker.engineer_features(dev_df)
    dev_df["s1_rank"] = dev_df.groupby("query_id")["stage1_score"].rank(ascending=False, method="first")

    dev_q_docs = pd.read_parquet(data_dir / "dev_queries/documents.parquet").set_index("id")
    dev_c_docs = pd.read_parquet(data_dir / "dev_candidates/documents.parquet").set_index("id")

    # 3. Precompute Dev Cross-Encoder Scores for Top-5
    dev_ce_scores = precompute_cross_encoder_scores_batched(
        dev_df, dev_q_docs, dev_c_docs, cross_encoder, chunker, top_n=5, aggregation="weighted", batch_size=32
    )

    # 4. Dev Calibration of Hyperparameters (tau in [0.05, 0.07, 0.08], beta in [0.10, 0.15, 0.20], gamma in [0.02, 0.04])
    print("\n--- Calibrating Phase 12 Parameters on Development Split ---")
    best_dev_p1 = 0.0
    best_dev_mrr = 0.0
    best_p12_cfg = {"tau": 0.07, "beta": 0.15, "gamma": 0.04, "top_n": 5}

    for tau in [0.05, 0.07, 0.08]:
        for beta in [0.10, 0.15, 0.20]:
            for gamma in [0.00, 0.02, 0.04]:
                acr = AdaptiveCrossReranker(tau=tau, beta=beta, gamma=gamma, top_n=5)
                dev_eval_df = acr.rerank_with_precomputed_ce_scores(dev_df, dev_ce_scores, tau=tau, beta=beta, gamma=gamma, top_n=5)
                m = HybridEvaluator.evaluate_predictions(dev_eval_df, score_col="final_cross_score")

                if (m["P@1"] > best_dev_p1) or (m["P@1"] == best_dev_p1 and m["MRR"] > best_dev_mrr):
                    best_dev_p1 = m["P@1"]
                    best_dev_mrr = m["MRR"]
                    best_p12_cfg = {"tau": tau, "beta": beta, "gamma": gamma, "top_n": 5}
                    print(f"  * New Best Dev: tau={tau:.2f}, beta={beta:.2f}, gamma={gamma:.2f} -> P@1={m['P@1']:.4f}, MRR={m['MRR']:.4f}")

    print("Selected Phase 12 Configuration:", best_p12_cfg)
    with open(checkpoints_dir / "phase12_adaptive_config.json", "w") as f:
        json.dump(best_p12_cfg, f, indent=2)

    # 5. Evaluate the 4 Required Models on Dev Split
    print("\n--- Evaluating 4 Architectures on Development Split ---")
    dev_p11_df = phase11_model.rerank(dev_df)
    m_dev_p11 = HybridEvaluator.evaluate_predictions(dev_p11_df, score_col="final_adaptive_score")

    # Cross-Encoder only (no gating, no Stage-1, stand-alone CE scores on top-5)
    dev_ce_only_df = dev_df.copy()
    raw_ce_vals = [dev_ce_scores.get((r["query_id"], r["candidate_id"]), -999.0) for _, r in dev_ce_only_df.iterrows()]
    dev_ce_only_df["ce_score"] = raw_ce_vals
    m_dev_ce_only = HybridEvaluator.evaluate_predictions(dev_ce_only_df, score_col="ce_score")

    # Rhetorical-Aware Cross-Encoder (CE + Rhetorical alignment)
    dev_rhet_ce_df = dev_df.copy()
    rhet_ce_vals = [dev_ce_scores.get((r["query_id"], r["candidate_id"]), -999.0) + 0.10 * r.get("rhetorical_cosine", 0.0) for _, r in dev_rhet_ce_df.iterrows()]
    dev_rhet_ce_df["rhet_ce_score"] = rhet_ce_vals
    m_dev_rhet_ce = HybridEvaluator.evaluate_predictions(dev_rhet_ce_df, score_col="rhet_ce_score")

    # Full Phase 12 Adaptive Cross-Reranker
    acr_best = AdaptiveCrossReranker(tau=best_p12_cfg["tau"], beta=best_p12_cfg["beta"], gamma=best_p12_cfg["gamma"], top_n=5)
    dev_p12_full_df = acr_best.rerank_with_precomputed_ce_scores(dev_df, dev_ce_scores)
    m_dev_p12_full = HybridEvaluator.evaluate_predictions(dev_p12_full_df, score_col="final_cross_score")

    dev_comparison_rows = [
        {"Model": "1. Phase 11 Frozen Baseline", "Configuration": "Margin-Gated Pairwise (tau=0.07, alpha=0.12, gamma=0.04)", "P@1": m_dev_p11["P@1"], "MRR": m_dev_p11["MRR"], "nDCG@10": m_dev_p11["nDCG@10"]},
        {"Model": "2. Cross-Encoder Only", "Configuration": "InLegalBERT Joint Token Attention (Top-5)", "P@1": m_dev_ce_only["P@1"], "MRR": m_dev_ce_only["MRR"], "nDCG@10": m_dev_ce_only["nDCG@10"]},
        {"Model": "3. Rhetorical-Aware Cross-Encoder", "Configuration": "Section-Aware Chunking + Discourse Alignment", "P@1": m_dev_rhet_ce["P@1"], "MRR": m_dev_rhet_ce["MRR"], "nDCG@10": m_dev_rhet_ce["nDCG@10"]},
        {"Model": "4. Phase 11 Adaptive + Cross-Encoder", "Configuration": f"Margin-Gated (tau={best_p12_cfg['tau']}, beta={best_p12_cfg['beta']}, gamma={best_p12_cfg['gamma']})", "P@1": m_dev_p12_full["P@1"], "MRR": m_dev_p12_full["MRR"], "nDCG@10": m_dev_p12_full["nDCG@10"]},
    ]
    df_dev_comp = pd.DataFrame(dev_comparison_rows)
    df_dev_comp.to_csv(results_tables_dir / "phase12_dev_experiments.csv", index=False)
    print("\nDevelopment Comparison:\n", df_dev_comp.to_string(index=False))

    # 6. Evaluation on Official Test Split (Exploratory / Independent Assessment)
    print("\n==========================================================================")
    print("EVALUATING ON OFFICIAL TEST SPLIT (234 ELIGIBLE QUERIES)")
    print("==========================================================================")
    test_df = pd.read_parquet(data_dir / "test/hybrid_features.parquet")
    test_df = test_df[~test_df["query_id"].isin(ZERO_RELEVANCE_QIDS)].copy()
    test_df = PairwiseNeuralReRanker.engineer_features(test_df)
    test_df["s1_rank"] = test_df.groupby("query_id")["stage1_score"].rank(ascending=False, method="first")

    test_q_docs = pd.read_parquet(data_dir / "test_queries/documents.parquet").set_index("id")
    test_c_docs = pd.read_parquet(data_dir / "test_candidates/documents.parquet").set_index("id")

    # Precompute Test Cross-Encoder Scores for Top-5 using Batched Pipeline
    t_ce_start = time.time()
    test_ce_scores = precompute_cross_encoder_scores_batched(
        test_df, test_q_docs, test_c_docs, cross_encoder, chunker, top_n=5, aggregation="weighted", batch_size=32
    )
    t_ce_total = time.time() - t_ce_start

    # Evaluate 4 Models on Test
    t1_start = time.time()
    test_p11_df = phase11_model.rerank(test_df)
    m_test_p11 = HybridEvaluator.evaluate_predictions(test_p11_df, score_col="final_adaptive_score")
    qm_test_p11 = compute_query_metrics(test_p11_df, score_col="final_adaptive_score")
    lat_p11 = (time.time() - t1_start) / 234.0 * 1000.0

    # Cross-Encoder only
    test_ce_only_df = test_df.copy()
    raw_test_ce = [test_ce_scores.get((r["query_id"], r["candidate_id"]), -999.0) for _, r in test_ce_only_df.iterrows()]
    test_ce_only_df["ce_score"] = raw_test_ce
    m_test_ce_only = HybridEvaluator.evaluate_predictions(test_ce_only_df, score_col="ce_score")
    qm_test_ce_only = compute_query_metrics(test_ce_only_df, score_col="ce_score")

    # Rhetorical-Aware Cross-Encoder
    test_rhet_ce_df = test_df.copy()
    rhet_test_ce = [test_ce_scores.get((r["query_id"], r["candidate_id"]), -999.0) + 0.10 * r.get("rhetorical_cosine", 0.0) for _, r in test_rhet_ce_df.iterrows()]
    test_rhet_ce_df["rhet_ce_score"] = rhet_test_ce
    m_test_rhet_ce = HybridEvaluator.evaluate_predictions(test_rhet_ce_df, score_col="rhet_ce_score")
    qm_test_rhet_ce = compute_query_metrics(test_rhet_ce_df, score_col="rhet_ce_score")

    # Full Phase 12 Adaptive Cross-Reranker
    t_full_start = time.time()
    test_p12_full_df = acr_best.rerank_with_precomputed_ce_scores(test_df, test_ce_scores)
    m_test_p12_full = HybridEvaluator.evaluate_predictions(test_p12_full_df, score_col="final_cross_score")
    qm_test_p12_full = compute_query_metrics(test_p12_full_df, score_col="final_cross_score")
    lat_p12 = (t_ce_total + (time.time() - t_full_start)) / 234.0 * 1000.0

    test_comparison_rows = [
        {"Model": "1. Phase 11 Frozen Baseline", "Configuration": "Margin-Gated Pairwise (tau=0.07, alpha=0.12, gamma=0.04)", "P@1": m_test_p11["P@1"], "P@5": m_test_p11["P@5"], "Recall@10": m_test_p11.get("R@10", m_test_p11.get("Recall@10", 0.0)), "MRR": m_test_p11["MRR"], "nDCG@10": m_test_p11["nDCG@10"], "Latency_ms_q": round(lat_p11, 2)},
        {"Model": "2. Cross-Encoder Only", "Configuration": "InLegalBERT Joint Token Attention (Top-5)", "P@1": m_test_ce_only["P@1"], "P@5": m_test_ce_only["P@5"], "Recall@10": m_test_ce_only.get("R@10", m_test_ce_only.get("Recall@10", 0.0)), "MRR": m_test_ce_only["MRR"], "nDCG@10": m_test_ce_only["nDCG@10"], "Latency_ms_q": round(t_ce_total / 234.0 * 1000.0, 2)},
        {"Model": "3. Rhetorical-Aware Cross-Encoder", "Configuration": "Section-Aware Chunking + Discourse Alignment", "P@1": m_test_rhet_ce["P@1"], "P@5": m_test_rhet_ce["P@5"], "Recall@10": m_test_rhet_ce.get("R@10", m_test_rhet_ce.get("Recall@10", 0.0)), "MRR": m_test_rhet_ce["MRR"], "nDCG@10": m_test_rhet_ce["nDCG@10"], "Latency_ms_q": round(t_ce_total / 234.0 * 1000.0, 2)},
        {"Model": "4. Phase 11 Adaptive + Cross-Encoder", "Configuration": f"Margin-Gated (tau={best_p12_cfg['tau']}, beta={best_p12_cfg['beta']}, gamma={best_p12_cfg['gamma']})", "P@1": m_test_p12_full["P@1"], "P@5": m_test_p12_full["P@5"], "Recall@10": m_test_p12_full.get("R@10", m_test_p12_full.get("Recall@10", 0.0)), "MRR": m_test_p12_full["MRR"], "nDCG@10": m_test_p12_full["nDCG@10"], "Latency_ms_q": round(lat_p12, 2)},
    ]
    df_test_comp = pd.DataFrame(test_comparison_rows)
    test_csv = results_tables_dir / "phase12_test_final_evaluation.csv"
    df_test_comp.to_csv(test_csv, index=False)
    print("\nTest Final Evaluation:\n", df_test_comp.to_string(index=False))

    # 7. Paired Bootstrap Significance (2,000 Resamples)
    print("\n--- Running 2,000 Paired Query-Level Bootstrap Tests ---")
    crit_metrics = ["P@1", "P@5", "Recall@10", "MRR", "nDCG@10"]
    boot_p11_vs_p12 = run_paired_bootstrap(qm_test_p11, qm_test_p12_full, crit_metrics, n_resamples=2000, random_state=42)
    boot_ce_vs_p12 = run_paired_bootstrap(qm_test_ce_only, qm_test_p12_full, crit_metrics, n_resamples=2000, random_state=42)

    boot_rows = []
    for m_name, res in boot_p11_vs_p12.items():
        boot_rows.append({
            "Comparison": "Phase 11 Baseline vs. Phase 12 Adaptive + Cross-Encoder",
            "Metric": m_name,
            "Model_A_Mean": res["mean_model_a"],
            "Model_B_Mean": res["mean_model_b"],
            "Mean_Diff": res["mean_diff"],
            "CI_Lower_95": res["ci_lower"],
            "CI_Upper_95": res["ci_upper"],
            "p_value": res["p_value"],
            "Statistically_Significant": "Yes" if res["is_significant"] else "No",
        })

    for m_name, res in boot_ce_vs_p12.items():
        boot_rows.append({
            "Comparison": "Cross-Encoder Only vs. Phase 12 Adaptive + Cross-Encoder",
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
    boot_csv = results_tables_dir / "phase12_bootstrap_comparison.csv"
    df_boot.to_csv(boot_csv, index=False)
    print(f"\nSaved Paired Bootstrap Comparison to {boot_csv}")
    print("\n" + df_boot[["Comparison", "Metric", "Mean_Diff", "CI_Lower_95", "CI_Upper_95", "p_value", "Statistically_Significant"]].to_string(index=False))

    print(f"\nPhase 12 Evaluation completed in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
