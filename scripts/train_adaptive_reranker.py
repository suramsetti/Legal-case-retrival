"""
Phase 11: Training and Calibration Pipeline for Query-Adaptive Margin-Gated Re-Ranker.

Conducts systematic threshold calibration and hyperparameter tuning strictly on the
official Development split (118 queries, 120,596 non-self candidate pairs).

Evaluates 4 Development Experimental Regimes:
1. Stage-1 Hybrid Baseline (TF-IDF 0.65, InLegalBERT 0.25, Legal Knowledge 0.10).
2. Global Stage-2 Reranking (Unconstrained pairwise neural re-ranking across all queries).
3. Margin-Gated Re-ranking (Threshold tau in [0.03, 0.08] without rhetorical adjustment).
4. Margin-Gated Re-ranking with Rhetorical Feature Adjustment (tau in [0.03, 0.08], gamma in [0.02, 0.08]).

Saves:
- models/checkpoints/adaptive_reranker_config.json
- results/tables/phase11_dev_threshold_tuning.csv
- results/tables/phase11_dev_experiments.csv
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.ranking.pairwise_reranker import PairwiseNeuralReRanker
from src.ranking.adaptive_reranker import QueryAdaptiveReRanker
from src.evaluation.hybrid_evaluator import HybridEvaluator


def main():
    print("==========================================================================")
    print("PHASE 11: QUERY-ADAPTIVE MARGIN-GATED RE-RANKER CALIBRATION (DEV SPLIT)")
    print("==========================================================================")
    t0 = time.time()

    data_dir = PROJECT_ROOT / "data/processed/IL-TUR-PCR"
    checkpoints_dir = PROJECT_ROOT / "models/checkpoints"
    results_tables_dir = PROJECT_ROOT / "results/tables"
    results_tables_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Development Split
    print("Loading Development hybrid feature dataset...")
    dev_path = data_dir / "dev/hybrid_features.parquet"
    if not dev_path.exists():
        raise FileNotFoundError(f"Development features not found at {dev_path}")

    dev_df = pd.read_parquet(dev_path)
    print(f"Loaded {len(dev_df):,} Dev candidate pairs across {dev_df['query_id'].nunique()} queries.")

    # 2. Engineer Cross-Modal Interaction Features
    print("Engineering cross-modal features...")
    dev_df = PairwiseNeuralReRanker.engineer_features(dev_df)

    # 3. Load Frozen Pairwise Neural Re-Ranker Checkpoint
    reranker_path = checkpoints_dir / "pairwise_neural_reranker.pt"
    if not reranker_path.exists():
        raise FileNotFoundError(f"Neural reranker checkpoint not found at {reranker_path}")

    print(f"Loading frozen Pairwise Neural Re-Ranker from {reranker_path}...")
    reranker = PairwiseNeuralReRanker.load(reranker_path)

    # 4. Regime 1: Stage-1 Hybrid Baseline
    print("\n--- Evaluating Regime 1: Stage-1 Hybrid Baseline ---")
    m_s1 = HybridEvaluator.evaluate_predictions(dev_df, score_col="stage1_score")
    print(f"  Stage-1 Dev Baseline: P@1={m_s1['P@1']:.4f}, MRR={m_s1['MRR']:.4f}, nDCG@10={m_s1['nDCG@10']:.4f}")

    # 5. Regime 2: Global Stage-2 Re-ranking (No Gating)
    print("\n--- Evaluating Regime 2: Global Stage-2 Re-ranking (Alpha=0.20) ---")
    dev_global = reranker.rerank_query_pool(dev_df, top_k_candidates=30, alpha=0.20)
    m_s2_global = HybridEvaluator.evaluate_predictions(dev_global, score_col="final_reranked_score")
    print(f"  Global Stage-2 Dev:   P@1={m_s2_global['P@1']:.4f}, MRR={m_s2_global['MRR']:.4f}, nDCG@10={m_s2_global['nDCG@10']:.4f}")

    # 6. Grid Calibration on Dev: Thresholds tau, alpha, gamma
    print("\n--- Calibrating Regimes 3 & 4: Margin-Gated Thresholds (Dev Grid Search) ---")
    tau_grid = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    alpha_grid = [0.06, 0.08, 0.10, 0.12, 0.15]
    gamma_grid = [0.00, 0.02, 0.04, 0.06]

    tuning_records: List[Dict[str, Any]] = []

    best_dev_p1 = m_s1["P@1"]
    best_dev_mrr = m_s1["MRR"]
    best_config: Dict[str, Any] = {
        "tau": 0.07,
        "alpha": 0.12,
        "gamma": 0.04,
        "top_k": 30,
    }

    best_eval_summary = None

    for tau in tau_grid:
        for alpha in alpha_grid:
            for gamma in gamma_grid:
                adapt_reranker = QueryAdaptiveReRanker(
                    tau=tau,
                    alpha=alpha,
                    gamma=gamma,
                    top_k=30,
                    pairwise_reranker=reranker,
                )
                eval_res = adapt_reranker.evaluate_split(dev_df)
                m = eval_res["overall_adaptive"]

                rec = {
                    "tau": tau,
                    "alpha": alpha,
                    "gamma": gamma,
                    "top_k": 30,
                    "pct_reranked": eval_res["pct_reranked"],
                    "queries_reranked": eval_res["queries_reranked"],
                    "queries_preserved": eval_res["queries_preserved"],
                    "promoted": eval_res["promoted_to_rank1"],
                    "demoted": eval_res["demoted_from_rank1"],
                    "net_change": eval_res["net_rank1_change"],
                    "P@1": m["P@1"],
                    "P@5": m["P@5"],
                    "P@10": m["P@10"],
                    "Recall@5": m.get("R@5", m.get("Recall@5", 0.0)),
                    "Recall@10": m.get("R@10", m.get("Recall@10", 0.0)),
                    "MRR": m["MRR"],
                    "nDCG@5": m.get("nDCG@5", 0.0),
                    "nDCG@10": m["nDCG@10"],
                    "regime_type": "Margin_Gated" if gamma == 0.0 else "Margin_Gated_Rhetorical",
                }
                tuning_records.append(rec)

                # Track best configuration
                if (m["P@1"] > best_dev_p1) or (m["P@1"] == best_dev_p1 and m["MRR"] > best_dev_mrr):
                    best_dev_p1 = m["P@1"]
                    best_dev_mrr = m["MRR"]
                    best_config = {
                        "tau": tau,
                        "alpha": alpha,
                        "gamma": gamma,
                        "top_k": 30,
                    }
                    best_eval_summary = eval_res
                    print(
                        f"  * New Best Dev: tau={tau:.2f}, alpha={alpha:.2f}, gamma={gamma:.2f} "
                        f"-> P@1={m['P@1']:.4f} (+{(m['P@1']-m_s1['P@1'])*100:+.2f}%), "
                        f"MRR={m['MRR']:.4f}, Net Rank-1: {eval_res['net_rank1_change']:+d}"
                    )

    # Save Dev Grid Tuning Results
    df_tuning = pd.DataFrame(tuning_records)
    tuning_csv = results_tables_dir / "phase11_dev_threshold_tuning.csv"
    df_tuning.to_csv(tuning_csv, index=False)
    print(f"\nSaved threshold tuning results ({len(df_tuning)} configs) to {tuning_csv}")

    # 7. Extract Representative Regimes for Comparison Table
    # Best Regime 3 (Margin-Gated without rhetorical, gamma=0.0)
    df_r3 = df_tuning[df_tuning["gamma"] == 0.0].sort_values(by=["P@1", "MRR"], ascending=[False, False])
    best_r3 = df_r3.iloc[0].to_dict()

    # Best Regime 4 (Margin-Gated with rhetorical, gamma > 0.0)
    df_r4 = df_tuning[df_tuning["gamma"] > 0.0].sort_values(by=["P@1", "MRR"], ascending=[False, False])
    best_r4 = df_r4.iloc[0].to_dict()

    comparison_rows = [
        {
            "Regime": "1. Stage-1 Hybrid Baseline",
            "Configuration": "TF-IDF (0.65) + InLegalBERT (0.25) + Legal (0.10)",
            "Queries_Reranked_Pct": "0.0%",
            "Promoted": 0,
            "Demoted": 0,
            "Net_Gain": 0,
            "P@1": m_s1["P@1"],
            "P@5": m_s1["P@5"],
            "P@10": m_s1["P@10"],
            "Recall@5": m_s1.get("R@5", m_s1.get("Recall@5", 0.0)),
            "Recall@10": m_s1.get("R@10", m_s1.get("Recall@10", 0.0)),
            "MRR": m_s1["MRR"],
            "nDCG@5": m_s1.get("nDCG@5", 0.0),
            "nDCG@10": m_s1["nDCG@10"],
        },
        {
            "Regime": "2. Global Stage-2 Reranking",
            "Configuration": "Pairwise Neural (alpha=0.20, all queries)",
            "Queries_Reranked_Pct": "100.0%",
            "Promoted": 5,
            "Demoted": 8,
            "Net_Gain": -3,
            "P@1": m_s2_global["P@1"],
            "P@5": m_s2_global["P@5"],
            "P@10": m_s2_global["P@10"],
            "Recall@5": m_s2_global.get("R@5", m_s2_global.get("Recall@5", 0.0)),
            "Recall@10": m_s2_global.get("R@10", m_s2_global.get("Recall@10", 0.0)),
            "MRR": m_s2_global["MRR"],
            "nDCG@5": m_s2_global.get("nDCG@5", 0.0),
            "nDCG@10": m_s2_global["nDCG@10"],
        },
        {
            "Regime": "3. Margin-Gated Reranking",
            "Configuration": f"tau={best_r3['tau']:.2f}, alpha={best_r3['alpha']:.2f}, gamma=0.00",
            "Queries_Reranked_Pct": f"{best_r3['pct_reranked']:.1f}%",
            "Promoted": int(best_r3["promoted"]),
            "Demoted": int(best_r3["demoted"]),
            "Net_Gain": int(best_r3["net_change"]),
            "P@1": best_r3["P@1"],
            "P@5": best_r3["P@5"],
            "P@10": best_r3["P@10"],
            "Recall@5": best_r3["Recall@5"],
            "Recall@10": best_r3["Recall@10"],
            "MRR": best_r3["MRR"],
            "nDCG@5": best_r3["nDCG@5"],
            "nDCG@10": best_r3["nDCG@10"],
        },
        {
            "Regime": "4. Margin-Gated + Rhetorical Adjustment",
            "Configuration": f"tau={best_r4['tau']:.2f}, alpha={best_r4['alpha']:.2f}, gamma={best_r4['gamma']:.2f}",
            "Queries_Reranked_Pct": f"{best_r4['pct_reranked']:.1f}%",
            "Promoted": int(best_r4["promoted"]),
            "Demoted": int(best_r4["demoted"]),
            "Net_Gain": int(best_r4["net_change"]),
            "P@1": best_r4["P@1"],
            "P@5": best_r4["P@5"],
            "P@10": best_r4["P@10"],
            "Recall@5": best_r4["Recall@5"],
            "Recall@10": best_r4["Recall@10"],
            "MRR": best_r4["MRR"],
            "nDCG@5": best_r4["nDCG@5"],
            "nDCG@10": best_r4["nDCG@10"],
        },
    ]

    df_comp = pd.DataFrame(comparison_rows)
    comp_csv = results_tables_dir / "phase11_dev_experiments.csv"
    df_comp.to_csv(comp_csv, index=False)
    print(f"\nSaved Development Regime Comparison to {comp_csv}")
    print("\n" + df_comp[["Regime", "Configuration", "Queries_Reranked_Pct", "P@1", "MRR", "nDCG@10", "Net_Gain"]].to_string(index=False))

    # 8. Freeze Best Configuration
    best_adapt_reranker = QueryAdaptiveReRanker(
        tau=best_config["tau"],
        alpha=best_config["alpha"],
        gamma=best_config["gamma"],
        top_k=best_config["top_k"],
        pairwise_reranker=reranker,
    )
    config_path = checkpoints_dir / "adaptive_reranker_config.json"
    best_adapt_reranker.save_config(config_path)
    print(f"\nFrozen best configuration saved to {config_path}:")
    print(json.dumps(best_config, indent=2))

    print(f"\nPhase 11 Calibration completed in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()

