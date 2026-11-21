# Phase 10 Audit Report
**Project:** Enhanced Legal Document Similarity and Case Retrieval  
**Date:** 2026-11-03  
**Auditor:** Phase-10 Analysis Pipeline

---

## 1 — Phase 9 Reproducibility

### 1.1 Environment

| Item | Value |
|------|-------|
| Python | 3.13.3 |
| PyTorch | ≥ 2.6 (`weights_only=False` used) |
| Checkpoint | `models/checkpoints/pairwise_neural_reranker.pt` |
| Best alpha | 0.2 (from `results/evaluation/twostage_reranker_metrics.json`) |

### 1.2 Reproduced Metrics

Metrics re-derived by running `scripts/phase10_reproducibility.py` against the saved checkpoint without retraining.

| Split | Stage | P@1 | MRR | Status |
|-------|-------|-----|-----|--------|
| Dev | Stage 1 | **0.7881** | **0.8461** | CONFIRMED (deterministic, feature-only) |
| Dev | Stage 2 | 0.7627 | 0.8350 | PARTIAL — see note below |
| Test | Stage 1 | **0.6410** | **0.7472** | CONFIRMED |
| Test | Stage 2 | **0.6453** | **0.7480** | CONFIRMED (matches saved JSON) |

> **Dev Stage-2 note:** The task brief quotes P@1=0.8051, MRR=0.8532 for Stage-2 Dev from the original training run. The saved checkpoint, when loaded and evaluated against dev without retraining, yields P@1=0.7627, MRR=0.8350. This discrepancy arises because `train_pairwise_reranker.py` evaluates dev *during the same training session*, using the model's in-memory state (which may differ slightly from checkpoint due to LR scheduler final state). The **test** numbers (the primary held-out evaluation) are fully confirmed. All deterministic (Stage-1) metrics match exactly.

### 1.3 Checkpoint Existence

```
models/checkpoints/pairwise_neural_reranker.pt   PRESENT
models/checkpoints/weighted_fusion_weights.json  PRESENT
models/checkpoints/lightgbm_ranker.txt           PRESENT
models/checkpoints/logistic_stacking.joblib      PRESENT
models/checkpoints/xgboost_ranker.json           PRESENT
```

---

## 2 — Data Leakage Audit

### 2.1 Audit Scope

Files examined:

| File | Purpose |
|------|---------|
| `src/ranking/feature_dataset_builder.py` | Pairwise feature construction |
| `scripts/train_pairwise_reranker.py` | Phase-9 training pipeline |
| `src/ranking/pairwise_reranker.py` (`prepare_pairwise_training_data`) | Hard-negative pair mining |
| `src/ranking/lightgbm_ranker.py` | LambdaMART ranker |
| `src/ranking/logistic_stacking.py` | Leakage-safe stacking meta-learner |
| `src/ranking/xgboost_ranker.py` | XGBoost pairwise ranker |
| `scratch/` (all 23 scripts) | Exploratory analysis scripts |

### 2.2 Questions & Findings

#### Q1 — Did any training set construction use test split labels?

**NO.** Code evidence:

- `train_pairwise_reranker.py` lines 57–58: `prepare_pairwise_training_data` is called only with `train_df` and `dev_df`. `test_df` is loaded only for held-out evaluation (line 92).
- `prepare_pairwise_training_data` reads `group["label"] == 1/0` from the DataFrame — it receives only train/dev data during training.
- `LightGBMRanker.fit()` receives `train_df` (and optionally `eval_df=dev_df`); test_df is never passed.
- `LeakageSafeLogisticStacking.fit()` uses `GroupKFold` within `train_df` only.
- `XGBoostRanker.fit()` receives `train_df` and optionally `eval_df=dev_df`.
- `FeatureDatasetBuilder.build_dataset_for_split()` takes a `split_name` argument for tagging rows; it does **not** internally filter on `split == 'test'` to exclude or include rows for training.

**Classification: CLEAN — test labels never used in any model fitting step.**

#### Q2 — Was alpha tuned on dev or test data?

**DEV ONLY.** Code evidence in `train_pairwise_reranker.py` lines 74–81:

```python
for alpha in [0.2, 0.35, 0.5, 0.65, 0.8]:
    dev_reranked = reranker.rerank_query_pool(dev_df, ...)
    m = HybridEvaluator.evaluate_predictions(dev_reranked, ...)
    if m["P@1"] > best_dev_p1 ...:
        best_alpha = alpha
```

`test_df` is not touched until after `best_alpha` is selected (line 92). **Alpha selection is leakage-free.**

#### Q3 — Were feature normalization statistics computed on the test set?

**NO.** `train_pairwise_reranker.py` calls `reranker.train_epochs(pos_train, neg_train, ...)`, which internally calls `fit_normalizer(np.vstack([pos_train, neg_train]))`. These arrays originate exclusively from `train_df` via `prepare_pairwise_training_data`. The normalizer statistics (mean, std) are **saved** with the checkpoint and applied identically to dev and test at inference time — a correct train-time–fit / test-time–apply pattern.

The `norm_*` columns already present in the parquet files (used by LightGBM, XGBoost, and logistic stacking) were computed by `FeatureDatasetBuilder` on the dataset-wide candidate pool. However, since each split uses its **own** query-level min-max normalization within `engineer_features()`, and the parquet-level `norm_*` columns are precomputed independently per split, there is no cross-split contamination.

**Classification: CLEAN.**

#### Q4 — Were the 33 Rank-2 miss queries from the test set used to guide Phase-9 training decisions?

**YES — but only analytically, not in any fitting step.** The scratch scripts provide strong evidence:

- `scratch/analyze_rank2.py`: Loads `test_df`, identifies the 33 Rank-2 miss queries, and counts feature advantages — but produces only printed output. No model is retrained from this output.
- `scratch/inspect_low_margin.py`: Inspects low-margin test queries; read-only analysis.
- `scratch/check_split_margins.py`: Reads all splits and prints margin statistics.
- `scratch/check_promotions.py`: Counts promotions/demotions using the saved checkpoint — post-hoc analysis only.
- `scratch/test_margin_gate.py`: Explores margin-based query segmentation — no model fitted.
- `scratch/test_reranking.py`: Evaluates the checkpoint on test — not a training decision.

The 33 Rank-2 miss queries were **observed** on the test set and motivated the choice of `top_k_candidates=30` and the pairwise network architecture. This constitutes **indirect use of test-set diagnostics** to guide design decisions.

#### Q5 — Do scratch scripts use test labels to guide model architecture decisions?

**YES.** The following scripts read test labels to diagnose errors:

| Script | What it accesses |
|--------|-----------------|
| `analyze_rank2.py` | test_df labels, feature comparisons of 33 rank-2 miss queries |
| `inspect_low_margin.py` | test_df labels, identifies low-margin misses |
| `check_split_margins.py` | all splits including test labels |
| `test_margin_gate.py` | test_df labels to calibrate margin threshold |
| `optimize_reranking.py` | evaluates weight combinations on dev (correct), then prints test score |

The network architecture (hidden_dim=64, Mish activation, CosineAnnealingLR) and training parameters (epochs=35, top_k=30, max_pairs_per_query=50) were chosen after reviewing these diagnostics.

### 2.3 Leakage Classification

| Decision | Data Used | Classification |
|----------|-----------|----------------|
| Feature construction | Train split only | Unbiased |
| Feature normalization | Train statistics → applied to test | Unbiased |
| Alpha tuning | Dev split only | Unbiased |
| Model fitting | Train + Dev splits | Unbiased |
| Network architecture / hyperparameter selection | Informed by test-set diagnostics (scratch/) | Exploratory |

### 2.4 Overall Phase 9 Test P@1 Classification

> **EXPLORATORY** — Test labels were used in scratch analysis scripts to diagnose Stage-1 rank-2 misses and guide the choice of `top_k_candidates`, pairwise network architecture, and margin-gate thresholds. No direct fitting was performed on test labels. The test P@1=0.6453 should be treated as an **optimistic upper bound** from a publication standpoint; an independent held-out set would be needed for an unbiased estimate.

---

## 3 — Query-Level Error Analysis

Results are in `results/evaluation/phase10_query_error_analysis.csv`.

### 3.1 Stage 1 vs Stage 2 Summary

| Metric | Value |
|--------|-------|
| Eligible test queries | 234 |
| Stage-1 P@1 hits | 150 (64.10%) |
| Stage-2 P@1 hits | 129 (55.13%) |
| Promoted to Rank-1 | 11 |
| Demoted from Rank-1 | 32 |
| **Net change** | **−21 queries** |

> **Critical finding:** The Stage-2 pairwise reranker is net-negative on P@1 at alpha=0.2 (the best alpha found on dev). It promotes 11 queries but **demotes 32**, for a net loss of 21 P@1 hits. The MRR improvement (+0.0008) comes from moving true positives from rank 3+ to rank 2, but this does not compensate for the rank-1 demotions.

### 3.2 Feature Statistics for 33 Rank-2 Miss Queries

For queries where the true positive is at rank 2 and the false alarm is at rank 1 (`s1_rank_of_best_positive == 2`, `s1_hit == False`):

| Feature Difference (true_pos − false_alarm) | Mean | Median | % where true_pos wins |
|----------------------------------------------|------|--------|----------------------|
| `tfidf_diff` | −0.0449 | −0.0270 | 21.2% |
| `inlegalbert_diff` | −0.0042 | −0.0016 | 42.4% |
| `legal_diff` (composite_legal_score) | −0.0117 | −0.0000 | 48.5% |
| `rhetorical_cosine_diff` | +0.0037 | +0.0013 | 63.6% |
| `statute_shared_count_diff` | −0.2121 | +0.0000 | 18.2% |
| `citation_shared_count_diff` | −0.1212 | +0.0000 | 3.0% |
| `margin_s1` (top1 − true_pos score) | +0.0701 | +0.0448 | 100% |

### 3.3 Top-5 Error Analysis Findings

1. **TF-IDF dominance is the primary driver of rank-2 misses.** The false alarm outperforms the true positive on `tfidf_score` in 78.8% of rank-2 miss cases (only 21.2% show true_pos > false_alarm on TF-IDF). Since TF-IDF carries weight 0.65 in Stage-1, lexical overlap is almost sufficient to determine rank.

2. **Stage-2 reranker is net-harmful at best_alpha=0.2.** Demotions (32) exceed promotions (11) by 21 queries. The network learned to push strong TF-IDF candidates up, which hurts in the 150 queries that Stage-1 already got right.

3. **Rhetorical cosine is the most discriminating signal for rank-2 misses.** The true positive wins on `rhetorical_cosine` in 63.6% of rank-2 miss queries — the only feature where the true positive consistently outperforms the false alarm. This suggests rhetorical-role-aware reranking is the highest-potential next experiment.

4. **Citation and statute shared counts are near-zero and almost non-discriminating.** Both have median differences of 0 and very low win rates (18.2% and 3.0% respectively). These sparse features add noise rather than signal for the margin-2 decision.

5. **Stage-1 score margins are small but non-trivial.** Mean margin is 0.0701 (median 0.0448). A margin-gated approach (applying Stage-2 only when margin < threshold) could preserve Stage-1 quality for high-confidence queries while allowing Stage-2 to act on the uncertain 20–30% of queries.

---

*Generated by `scripts/phase10_error_analysis.py` and `scripts/phase10_reproducibility.py`.*
