"""
Phase 10 – Reproducibility Script.

Re-runs Stage-1 and Stage-2 metrics from saved checkpoint to confirm
the numbers reported in twostage_reranker_metrics.json.

Expected values (from saved JSON):
  Stage 1 Dev:  P@1=0.7881, MRR=0.8461
  Stage 2 Dev:  P@1=0.8051, MRR=0.8532
  Stage 1 Test: P@1=0.6410, MRR=0.7472
  Stage 2 Test: P@1=0.6453, MRR=0.7480  (saved file values)

NOTE: The task brief quoted Phase-9 dev metrics of P@1=0.7881/0.8051.
      Those originate from the training run that produced the checkpoint;
      this script re-derives the test numbers from the saved weights and
      checks them against the persisted JSON.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ranking.pairwise_reranker import PairwiseNeuralReRanker
from src.evaluation.hybrid_evaluator import HybridEvaluator

ZERO_RELEVANCE_QUERIES = {"0000431954", "0001508893", "0001864396"}
RESULTS_FILE = PROJECT_ROOT / "results/evaluation/twostage_reranker_metrics.json"


def load_and_engineer(split: str) -> pd.DataFrame:
    """Load hybrid_features.parquet and compute Stage-1 features."""
    path = PROJECT_ROOT / "data/processed/IL-TUR-PCR" / split / "hybrid_features.parquet"
    df = pd.read_parquet(path)
    df = PairwiseNeuralReRanker.engineer_features(df)
    return df


def evaluate_stage1(df: pd.DataFrame, label: str) -> dict:
    m = HybridEvaluator.evaluate_predictions(df, score_col="stage1_score")
    print(f"  {label}: P@1={m['P@1']:.4f}, MRR={m['MRR']:.4f}, nDCG@10={m['nDCG@10']:.4f}  "
          f"(n_queries={m['evaluated_queries']})")
    return m


def evaluate_stage2(reranker: PairwiseNeuralReRanker, df: pd.DataFrame, alpha: float, label: str) -> dict:
    reranked = reranker.rerank_query_pool(df, top_k_candidates=30, alpha=alpha)
    m = HybridEvaluator.evaluate_predictions(reranked, score_col="final_reranked_score")
    print(f"  {label}: P@1={m['P@1']:.4f}, MRR={m['MRR']:.4f}, nDCG@10={m['nDCG@10']:.4f}  "
          f"(n_queries={m['evaluated_queries']})")
    return m


def check(label: str, computed: float, expected: float, tol: float = 5e-4) -> str:
    ok = abs(computed - expected) <= tol
    status = "MATCH" if ok else f"MISMATCH (expected {expected:.4f}, got {computed:.4f})"
    return f"  {label}: {status}"


def main():
    print("=" * 65)
    print("Phase 10 – Stage-1 & Stage-2 Reproducibility Check")
    print("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────
    print("\n[1/4] Loading datasets …")
    dev_df  = load_and_engineer("dev")
    test_df = load_and_engineer("test")

    # Confirm zero-relevance queries are excluded by evaluator (it skips
    # queries with 0 positives automatically; we log what we see).
    test_qids = set(test_df["query_id"].unique())
    zero_found = test_qids & ZERO_RELEVANCE_QUERIES
    print(f"      Zero-relevance queries in test data: {zero_found}")

    # ── 2. Stage-1 metrics ───────────────────────────────────────
    print("\n[2/4] Stage-1 baseline metrics …")
    m_s1_dev  = evaluate_stage1(dev_df,  "Stage-1 Dev ")
    m_s1_test = evaluate_stage1(test_df, "Stage-1 Test")

    # ── 3. Load checkpoint + best_alpha ──────────────────────────
    print("\n[3/4] Loading pairwise reranker checkpoint …")
    ckpt_path = PROJECT_ROOT / "models/checkpoints/pairwise_neural_reranker.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    reranker = PairwiseNeuralReRanker.load(ckpt_path)
    print(f"      Checkpoint loaded: {ckpt_path}")

    # Load best_alpha from saved JSON
    with open(RESULTS_FILE) as f:
        saved = json.load(f)
    best_alpha = saved.get("best_alpha", 0.2)
    print(f"      Using best_alpha={best_alpha} (from saved JSON)")

    # ── 4. Stage-2 metrics ───────────────────────────────────────
    print("\n[4/4] Stage-2 re-ranked metrics …")
    m_s2_dev  = evaluate_stage2(reranker, dev_df,  best_alpha, "Stage-2 Dev ")
    m_s2_test = evaluate_stage2(reranker, test_df, best_alpha, "Stage-2 Test")

    # ── 5. Reproducibility comparison ────────────────────────────
    print("\n" + "=" * 65)
    print("REPRODUCIBILITY SUMMARY")
    print("=" * 65)

    # The saved JSON contains test metrics; dev metrics come from training logs.
    # We compare test against saved JSON; dev targets from task brief are
    # from the original training run (non-deterministic) so dev comparison uses
    # a relaxed tolerance of 0.02.
    TARGETS = {
        "Stage1_Dev_P@1":   0.7881,
        "Stage1_Dev_MRR":   0.8461,
        "Stage2_Dev_P@1":   0.8051,
        "Stage2_Dev_MRR":   0.8532,
        "Stage1_Test_P@1":  saved["stage1_baseline_test"]["P@1"],
        "Stage1_Test_MRR":  saved["stage1_baseline_test"]["MRR"],
        "Stage2_Test_P@1":  saved["stage2_reranked_test"]["P@1"],
        "Stage2_Test_MRR":  saved["stage2_reranked_test"]["MRR"],
    }

    results = {
        "Stage1_Dev_P@1":  m_s1_dev["P@1"],
        "Stage1_Dev_MRR":  m_s1_dev["MRR"],
        "Stage2_Dev_P@1":  m_s2_dev["P@1"],
        "Stage2_Dev_MRR":  m_s2_dev["MRR"],
        "Stage1_Test_P@1": m_s1_test["P@1"],
        "Stage1_Test_MRR": m_s1_test["MRR"],
        "Stage2_Test_P@1": m_s2_test["P@1"],
        "Stage2_Test_MRR": m_s2_test["MRR"],
    }

    all_ok = True
    for key, target in TARGETS.items():
        line = check(key, results[key], target)
        print(line)
        if "MISMATCH" in line:
            all_ok = False

    print("\n" + ("ALL METRICS REPRODUCED" if all_ok else "SOME METRICS DIFFER") )
    print("=" * 65)

    # ── 6. Save reproducibility report ───────────────────────────
    report = {
        "status": "CONFIRMED" if all_ok else "PARTIAL",
        "stage1_dev":   {"P@1": m_s1_dev["P@1"],  "MRR": m_s1_dev["MRR"]},
        "stage2_dev":   {"P@1": m_s2_dev["P@1"],  "MRR": m_s2_dev["MRR"]},
        "stage1_test":  {"P@1": m_s1_test["P@1"], "MRR": m_s1_test["MRR"]},
        "stage2_test":  {"P@1": m_s2_test["P@1"], "MRR": m_s2_test["MRR"]},
        "targets":       TARGETS,
        "best_alpha":    best_alpha,
    }

    out_path = PROJECT_ROOT / "results/evaluation/phase10_reproducibility.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
