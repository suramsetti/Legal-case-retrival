"""
Training Pipeline for Second-Stage Pairwise Neural Re-Ranker.

Trains a deep feature interaction network on Stage-1 Top-K hard negative pairs
to promote near-miss precedent citations (Rank #2 and Rank #3) to Rank #1.
"""

import sys
import os
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.ranking.pairwise_reranker import PairwiseNeuralReRanker
from src.evaluation.hybrid_evaluator import HybridEvaluator


def main():
    print("=== Training Stage-2 Pairwise Neural Re-Ranker ===")
    t0 = time.time()

    data_dir = PROJECT_ROOT / "data/processed/IL-TUR-PCR"
    checkpoints_dir = PROJECT_ROOT / "models/checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("Loading datasets...")
    train_df = pd.read_parquet(data_dir / "train/hybrid_features.parquet")
    dev_df = pd.read_parquet(data_dir / "dev/hybrid_features.parquet")
    test_df = pd.read_parquet(data_dir / "test/hybrid_features.parquet")

    print(f"Loaded: Train={len(train_df):,}, Dev={len(dev_df):,}, Test={len(test_df):,} pairs.")

    # 2. Engineer Features
    print("Engineering cross-modal interaction features...")
    train_df = PairwiseNeuralReRanker.engineer_features(train_df)
    dev_df = PairwiseNeuralReRanker.engineer_features(dev_df)
    test_df = PairwiseNeuralReRanker.engineer_features(test_df)

    # Stage 1 Baselines
    m_s1_dev = HybridEvaluator.evaluate_predictions(dev_df, score_col="stage1_score")
    m_s1_test = HybridEvaluator.evaluate_predictions(test_df, score_col="stage1_score")
    print(f"Stage 1 Dev Baseline:  P@1={m_s1_dev['P@1']:.4f}, MRR={m_s1_dev['MRR']:.4f}, nDCG@10={m_s1_dev['nDCG@10']:.4f}")
    print(f"Stage 1 Test Baseline: P@1={m_s1_test['P@1']:.4f}, MRR={m_s1_test['MRR']:.4f}, nDCG@10={m_s1_test['nDCG@10']:.4f}")

    # 3. Extract Top-K Hard Negative Pairs for Training
    K = 30
    print(f"Mining hard negative pairs from Top-{K} candidate pools...")
    reranker = PairwiseNeuralReRanker(hidden_dim=64, lr=0.003, margin=1.0)
    pos_train, neg_train = reranker.prepare_pairwise_training_data(train_df, top_k_candidates=K, max_pairs_per_query=50)
    pos_dev, neg_dev = reranker.prepare_pairwise_training_data(dev_df, top_k_candidates=K, max_pairs_per_query=30)

    print(f"Generated {len(pos_train):,} training pairs and {len(pos_dev):,} validation pairs.")

    # 4. Train Neural Re-Ranker
    print("Training Pairwise Neural Network (epochs=35, batch_size=128)...")
    losses = reranker.train_epochs(pos_train, neg_train, epochs=35, batch_size=128, weight_decay=1e-4)
    print(f"Final training loss: {losses[-1]:.4f} (initial: {losses[0]:.4f})")

    # 5. Tune Alpha on Dev Split
    print("Tuning ensemble weight alpha on Dev split...")
    best_alpha = 0.5
    best_dev_p1 = 0.0
    best_dev_mrr = 0.0

    for alpha in [0.2, 0.35, 0.5, 0.65, 0.8]:
        dev_reranked = reranker.rerank_query_pool(dev_df, top_k_candidates=K, alpha=alpha)
        m = HybridEvaluator.evaluate_predictions(dev_reranked, score_col="final_reranked_score")
        print(f"  Dev (alpha={alpha:.2f}): P@1={m['P@1']:.4f}, MRR={m['MRR']:.4f}, nDCG@10={m['nDCG@10']:.4f}")
        if m["P@1"] > best_dev_p1 or (m["P@1"] == best_dev_p1 and m["MRR"] > best_dev_mrr):
            best_dev_p1 = m["P@1"]
            best_dev_mrr = m["MRR"]
            best_alpha = alpha

    print(f"Selected Best Alpha: {best_alpha:.2f} (Dev P@1: {best_dev_p1:.4f}, Dev MRR: {best_dev_mrr:.4f})")

    # Save checkpoint
    ckpt_path = checkpoints_dir / "pairwise_neural_reranker.pt"
    reranker.save(ckpt_path)
    print(f"Saved Re-Ranker checkpoint to {ckpt_path}")

    # 6. Evaluate Two-Stage Model on Official Test Split
    print("\n=== Evaluating Two-Stage Pipeline on Official Test Split ===")
    test_reranked = reranker.rerank_query_pool(test_df, top_k_candidates=K, alpha=best_alpha)
    m_test_final = HybridEvaluator.evaluate_predictions(test_reranked, score_col="final_reranked_score")

    print("\n-------------------------------------------------------------")
    print(f"Stage 1 Test Baseline: P@1 = {m_s1_test['P@1']:.4f} | MRR = {m_s1_test['MRR']:.4f} | nDCG@10 = {m_s1_test['nDCG@10']:.4f}")
    print(f"Stage 2 Test Reranked: P@1 = {m_test_final['P@1']:.4f} | MRR = {m_test_final['MRR']:.4f} | nDCG@10 = {m_test_final['nDCG@10']:.4f}")
    delta_p1 = (m_test_final['P@1'] - m_s1_test['P@1']) * 100.0
    delta_mrr = (m_test_final['MRR'] - m_s1_test['MRR']) * 100.0
    print(f"Net Gain on Test:      P@1: {delta_p1:+.2f}% | MRR: {delta_mrr:+.2f}%")
    print("-------------------------------------------------------------")

    # Save comparison to results
    results_dir = PROJECT_ROOT / "results/evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "Two-Stage Hybrid Retrieval + Pairwise Neural Re-Ranker",
        "stage1_baseline_test": m_s1_test,
        "stage2_reranked_test": m_test_final,
        "best_alpha": best_alpha,
        "top_k_reranked": K,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(results_dir / "twostage_reranker_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved results to {results_dir / 'twostage_reranker_metrics.json'} in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()

