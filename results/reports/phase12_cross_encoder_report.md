# Phase 12 Final Report: Legal Cross-Encoder and Rhetorical-Aware Neural Reranking

**Project:** Enhanced Legal Document Similarity and Case Retrieval Using Hybrid BERT, Legal Knowledge, and Ensemble Learning  
**Date:** November 2026  
**Status:** Completed & Validated  
**Test Pool:** 234 Eligible Test Queries (1,727 Candidates; official non-self pools)  

---

## 1. Executive Summary

Phase 12 developed and evaluated a deep transformer **Cross-Encoder reranking model** based on `law-ai/InLegalBERT` with joint query-candidate token-level self-attention and discourse-aware chunking. We investigated whether direct cross-attention over legal text could surpass or enhance our frozen Phase 11 **Margin-Gated Adaptive Neural Reranker** (`tau=0.07`).

### Key Findings:
1. **Stand-alone Cross-Encoder Efficacy**: The fine-tuned `InLegalBERTCrossEncoder` achieves **48.29% P@1, 0.6423 MRR, and 0.4496 nDCG@10** on the official untouched test set. This surpasses the published ACL 2024 benchmark baseline (MRR 0.6081), demonstrating that joint token-level interaction captures nuanced precedent relevance.
2. **Adaptive Cross-Encoder Integration**: Integrating the cross-encoder into our margin-gated routing architecture (`AdaptiveCrossReranker`) yields **65.38% P@1, 0.7522 MRR, and 0.5857 nDCG@10**. A 2,000-sample paired bootstrap test reveals that Phase 12 Adaptive is statistically indistinguishable from the Phase 11 baseline ($p = 0.266$ for MRR, $p = 0.808$ for P@1; both 95% CIs encompass zero).
3. **Massive Efficiency Disparity**: The Phase 11 Pairwise Neural reranker executes in **11.48 ms/query** on CPU, whereas the full InLegalBERT Cross-Encoder requires **2,603.94 ms/query** (a **226.8× latency penalty**).
4. **Architectural Recommendation**: Given equivalent retrieval accuracy at 227× lower latency, the **Phase 11 Margin-Gated Pairwise Neural architecture** remains the optimal, frozen production system. The Phase 12 Cross-Encoder provides a complementary, highly explainable semantic verification tool for ambiguous, contested borderline cases.

---

## 2. Cross-Encoder Architecture and Section-Aware Chunking

### 2.1 Model Architecture (`InLegalBERTCrossEncoder`)
Standard bi-encoders encode query and candidate representations independently into fixed-dimensional vectors:
$$\mathbf{s}_{\text{bi}} = \cos(\mathbf{e}_q, \mathbf{e}_c)$$
While computationally efficient, bi-encoders lose fine-grained token-level interactions. In contrast, our Phase 12 Cross-Encoder concatenates query and candidate text into a joint sequence:
$$[\text{CLS}] \; q_1, q_2, \dots, q_m \; [\text{SEP}] \; c_1, c_2, \dots, c_n \; [\text{SEP}]$$
This allows every query token to attend to every candidate token across all self-attention layers of `law-ai/InLegalBERT`.

```
                    +---------------------------+
                    | Relevance Score s(q, c)   |
                    +-------------^-------------+
                                  | Linear + Tanh
                    +-------------+-------------+
                    | Pooler / [CLS] Token Head |
                    +-------------^-------------+
                                  |
               +--------------------------------------+
               |  InLegalBERT Transformer Layers      |
               |  (Joint Token Self-Attention)        |
               +------------------^-------------------+
                                  |
      +---------------------------+---------------------------+
      |                                                       |
[CLS] Query Tokens [SEP]                    Candidate Chunk Tokens [SEP]
```

To enable stable CPU fine-tuning and inference, lower transformer layers (0–9) were frozen, fine-tuning the top 2 transformer layers and the binary classification relevance projection head.

### 2.2 Section-Aware Chunking (`SectionAwareChunker`)
Indian Supreme Court judgments frequently exceed 10,000 words, far surpassing the 512-token context window of transformer models. Blind head/tail truncation truncates critical legal rationale. We implemented `SectionAwareChunker` which parses judgment text into functional legal discourse units:
- **Ratio Decidendi / Legal Principle**: Extracted using judicial holding markers (`"we hold"`, `"settled legal position"`, `"ratio"`, `"rule of law"`). Weight: $1.25\times$.
- **Final Ruling / Disposal**: Extracted using dispositive disposition patterns (`"appeal is allowed"`, `"appeal dismissed"`, `"order accordingly"`). Weight: $1.15\times$.
- **Headnotes & Case Primer**: Extracted from initial formal headers, procedural citations, and framing issues. Weight: $1.00\times$.
- **Factual Background**: Procedural narrative and factual background. Weight: $0.90\times$.

Candidate chunk scores are aggregated via normalized rhetorical weighting:
$$s_{\text{candidate}}(q, c) = \frac{\sum_{i=1}^K w_i \cdot s(q, \text{chunk}_i)}{\sum_{i=1}^K w_i}$$

---

## 3. Training and Optimization

### 3.1 Pairwise Margin Ranking Loss
The cross-encoder was trained on the official training split using pairwise ranking loss with hard negatives:
$$\mathcal{L}(\theta) = \max\left(0, \Delta - (s_\theta(q, c^+) - s_\theta(q, c^-))\right)$$
where $\Delta = 0.15$ is the pairwise margin, $c^+$ is a gold precedent, and $c^-$ is a top-ranked non-precedent retrieved by the Stage-1 hybrid system.

### 3.2 Training Trajectory
- **Mined Triples**: 240 hard-negative triples across 120 training queries.
- **Optimizer**: AdamW ($\text{lr} = 2 \times 10^{-5}$, weight decay $0.01$).
- **Batch Size**: 4 triples per batch (16 chunks per step with gradient accumulation).
- **Epochs**: 3 epochs.
- **Loss Convergence**: Initial loss $0.9870 \to$ Final loss $0.9127$ (monotonic decrease).
- **Checkpoint**: Saved to `models/checkpoints/legal_cross_encoder.pt` (440 MB).

---

## 4. Development Tuning and Routing Calibration

Tuning was conducted exclusively on the 59 development queries. We swept margin gating threshold $\tau \in [0.03, 0.08]$, cross-encoder fusion weight $\beta \in [0.05, 0.30]$, and rhetorical alignment weight $\gamma \in [0.0, 0.05]$ over candidate pool cutoffs Top-3, Top-5, and Top-10.

### Table 1: Development Set Calibration Experiments (59 Queries)

| Model Architecture | Configuration | Dev P@1 | Dev MRR | Dev nDCG@10 |
| :--- | :--- | :---: | :---: | :---: |
| **1. Phase 11 Frozen Baseline** | Margin-Gated Pairwise ($\tau=0.07, \alpha=0.12, \gamma=0.04$) | **79.66%** | **0.8556** | **0.6434** |
| **2. Cross-Encoder Only** | InLegalBERT Joint Attention (Top-5) | 42.37% | 0.6092 | 0.4562 |
| **3. Rhetorical-Aware Cross-Encoder** | Section Chunks + Discourse Alignment | 42.37% | 0.6122 | 0.4574 |
| **4. Phase 12 Adaptive + Cross-Encoder** | Margin-Gated ($\tau=0.05, \beta=0.10, \gamma=0.0$) | 77.97% | 0.8415 | 0.6349 |

**Optimal Frozen Phase 12 Hyperparameters**:
- Threshold $\tau^* = 0.05$ (queries with Stage-1 margin $< 0.05$ routed to cross-encoder).
- Cross-encoder blend $\beta^* = 0.10$.
- Candidate rerank depth: Top-5 candidates.
- Serialized to `models/checkpoints/phase12_adaptive_config.json`.

---

## 5. Untouched Final Test Evaluation

The four architectures were evaluated on the complete, untouched official IL-TUR PCR test split of **234 eligible queries** across **1,727 non-self candidates** (1,170 candidate pairs evaluated across top ranks).

### Table 2: Final Test Evaluation Benchmark (234 Test Queries)

| Model Architecture | P@1 | P@5 | Recall@10 | MRR | nDCG@10 | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 11 Baseline (Frozen)** | **66.24%** | **0.4444** | **0.5583** | **0.7660** | **0.5904** | **11.48 ms/query** |
| **Cross-Encoder Only** | 48.29% | 0.4393 | 0.4545 | 0.6423 | 0.4496 | 2,603.94 ms/query |
| **Rhetorical-Aware Cross-Encoder** | 48.29% | 0.4393 | 0.4529 | 0.6416 | 0.4477 | 2,603.94 ms/query |
| **Phase 12 Adaptive + Cross-Encoder** | 65.38% | 0.4393 | 0.5563 | 0.7522 | 0.5857 | 2,613.64 ms/query |

---

## 6. Paired Bootstrap Statistical Significance (2,000 Resamples)

To verify whether metric differences are statistically significant or attributable to sampling variance, we conducted query-level paired bootstrap testing with 2,000 resamples on the test set.

### Table 3: Paired Bootstrap Statistical Comparison

| Comparison | Metric | Mean Diff | 95% Confidence Interval | $p$-value | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 11 vs Phase 12 Adaptive** | **P@1** | -0.0085 | [-0.0470, +0.0342] | 0.808 | **No** |
| | **P@5** | -0.0051 | [-0.0154, +0.0043] | 0.313 | **No** |
| | **Recall@10** | -0.0020 | [-0.0189, +0.0119] | 0.835 | **No** |
| | **MRR** | -0.0138 | [-0.0377, +0.0103] | 0.266 | **No** |
| | **nDCG@10** | -0.0047 | [-0.0169, +0.0068] | 0.437 | **No** |
| **Cross-Encoder Only vs Phase 12 Adaptive** | **P@1** | **+0.1709** | **[+0.0983, +0.2436]** | **0.0005** | **Yes ($p < 0.001$)** |
| | **Recall@10** | **+0.1019** | **[+0.0837, +0.1213]** | **0.0005** | **Yes ($p < 0.001$)** |
| | **MRR** | **+0.1099** | **[+0.0663, +0.1579]** | **0.0005** | **Yes ($p < 0.001$)** |
| | **nDCG@10** | **+0.1360** | **[+0.1141, +0.1607]** | **0.0005** | **Yes ($p < 0.001$)** |

### Statistical Conclusions:
1. **Adaptive Integration Dominates Pure Cross-Encoding**: Gating the cross-encoder behind Stage-1 confidence delivers dramatic, statistically significant improvements ($+17.09\%$ P@1, $+0.1099$ MRR, $p = 0.0005$). Pure cross-encoding without multi-modal anchor signals suffers from false-positive semantic drift.
2. **Phase 11 Baseline Equivalence**: The difference between Phase 11 Pairwise Reranking and Phase 12 Adaptive Cross-Encoder is statistically non-significant across all five retrieval metrics ($p \in [0.266, 0.835]$), with confidence intervals spanning zero.

---

## 7. Computational Complexity and Latency Analysis

| Architecture | Forward Pass Type | Compute Device | Latency / Query | Throughput | Relative Speed |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 11 Pairwise MLP** | Feature Vector (11 dims) | CPU | **11.48 ms** | 87.1 queries/sec | **226.8× faster** |
| **Phase 12 Cross-Encoder** | Joint Self-Attention (512 tokens × chunks) | CPU | **2,603.94 ms** | 0.38 queries/sec | 1.0× (baseline) |

### Computational Insights:
- The Phase 11 model leverages precomputed dense embeddings and lexical statistics, performing reranking via an ultra-lightweight 11-dimensional neural interaction layer.
- The Phase 12 Cross-Encoder requires full quadratic self-attention over token pairs for multiple document chunks per query. While feasible for offline analysis or high-stakes borderline arbitration, it introduces prohibitive latency for real-time interactive search without GPU acceleration.

---

## 8. Data Integrity & Reproducibility Sign-Off

1. **Zero Data Leakage**:
   - The cross-encoder was trained exclusively on the 120 training queries and gold labels.
   - All routing hyperparameters ($\tau=0.05, \beta=0.10$) were selected on the 59 development queries.
   - Test labels were used solely for final evaluation reporting.
2. **Deterministic Reproducibility**:
   - Random seed 42 set across PyTorch and NumPy.
   - Batch size, learning rate, and margin parameters fully documented.
   - Pre-computed scores and predictions persisted in `results/tables/`.
3. **Comprehensive Test Suite**:
   - 93/93 unit tests passing (`tests/unit/test_phase12_cross_encoder.py` and prior phase test suites).

---

## 9. Phase 12 Deliverables Checklist

- [x] **`src/ranking/legal_cross_encoder.py`**: `SectionAwareChunker` and `InLegalBERTCrossEncoder` with batched chunk inference.
- [x] **`src/ranking/adaptive_cross_reranker.py`**: `AdaptiveCrossReranker` supporting margin gating and precomputed cross-encoder scores.
- [x] **`scripts/train_legal_cross_encoder.py`**: Pairwise training pipeline with hard-negative mining from candidate rankings.
- [x] **`scripts/evaluate_phase12_reranker.py`**: Comprehensive comparative evaluation across baselines with paired bootstrap statistics.
- [x] **`models/checkpoints/legal_cross_encoder.pt`**: Trained model weights (440 MB).
- [x] **`models/checkpoints/phase12_adaptive_config.json`**: Frozen hyperparameter configuration.
- [x] **`results/tables/phase12_dev_experiments.csv`**: Development grid search metrics.
- [x] **`results/tables/phase12_test_final_evaluation.csv`**: Final untouched test benchmark metrics.
- [x] **`results/tables/phase12_bootstrap_comparison.csv`**: 2,000 paired bootstrap resamples and p-values.
- [x] **`tests/unit/test_phase12_cross_encoder.py`**: Unit tests verifying chunking, scoring, and margin gating (7/7 passed).
- [x] **`results/reports/phase12_cross_encoder_report.md`**: Complete technical and experimental report.

