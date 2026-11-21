# Phase 10 — Recommended Next-Experiment Plan
**Project:** Enhanced Legal Document Similarity and Case Retrieval  
**Date:** 2026-11-03  
**Basis:** Phase-10 error analysis, leakage audit, and Phase-9 reproducibility results

---

## Preamble: What the Error Analysis Tells Us

Before prescribing experiments, the key findings from Phase 10 must drive the design:

| Finding | Implication |
|---------|-------------|
| TF-IDF is correct 78.8% of the time in rank-2 miss queries | The bottleneck is semantic, not lexical |
| Rhetorical cosine wins for true positives 63.6% of the time | Rhetorical role is the most promising discriminating signal |
| Stage-2 reranker demotes 32, promotes 11 (net −21 P@1) | The current reranker hurts; next-generation must fix this |
| `citation_shared_count` wins only 3% of the time | Sparse legal signals need richer encoding, not raw counts |
| Stage-1 margin < 0.05 in ~20% of test queries | A margin gate can protect high-confidence queries |

---

## Experiment 1 — InLegalBERT Cross-Encoder Reranking

### Motivation

InLegalBERT as a bi-encoder produces cosine similarities between fixed embeddings. A cross-encoder reads the query and candidate jointly (full attention over concatenation), capturing fine-grained interaction patterns. This is the strongest known signal for near-miss disambiguation.

### Data

| Split | Role | Preparation |
|-------|------|-------------|
| Train (817 queries × top-5 Stage-1 candidates) | Training pairs | For each train query, extract top-5 Stage-1 candidates by `stage1_score`. Form (query_text, candidate_text, label) triples. |
| Dev (118 queries × top-10 Stage-1 candidates) | Hyperparameter tuning | Used to select learning rate, margin, and number of training epochs. |
| Test (234 queries) | Final evaluation | Held out; touch only once after all hyperparameters are locked. |

**Pair counts (estimate):**
- Train positives: 817 queries, typically 1–5 positives each → ~1,500–3,000 positive pairs.
- Train hard negatives: top-5 minus positives → ~2,000–3,500 negative pairs.
- Total training triples: ~5,000–6,500 pairs.

### Features to Include

The cross-encoder produces a score `s_ce(q, c)`. Combine with Stage-1 score in a convex blend:

```
final_score = (1 - beta) * stage1_score + beta * s_ce(q, c)
```

Tune `beta ∈ {0.2, 0.3, 0.4, 0.5, 0.6}` on dev. Do **not** include raw feature columns — the cross-encoder learns all interactions implicitly.

### Training Objective

**Pairwise margin loss** (same as Phase 9 but now applied to the full text cross-encoder):

```
L = mean( log(1 + exp(-(s_pos - s_neg))) + ReLU(margin - (s_pos - s_neg)) )
```

with `margin = 1.0`. This encourages `s_ce(q, pos) - s_ce(q, neg) > 1`.

Alternative: **ListNet** (listwise softmax loss over the top-K candidate pool). If the pairwise objective produces frequent ties (equal predicted scores), switch to ListNet.

### Model Specification

- Base: `law-ai/InLegalBERT` (cached at `C:\Users\mohan\.cache\huggingface\hub`)
- Head: single linear layer on `[CLS]` token → scalar score
- Max sequence length: 512 tokens (concatenate query first 200 tokens + `[SEP]` + candidate first 300 tokens)
- Optimizer: AdamW, lr=2e-5, weight decay=0.01
- Scheduler: Linear warmup (10% of steps) then linear decay
- Epochs: 3 (cross-encoders overfit quickly on small legal corpora)

### Success/Failure Criteria on Dev

| Outcome | Decision |
|---------|----------|
| Dev P@1 ≥ 0.820 (> current 0.7881 Stage-1) | Success → evaluate on test |
| Dev P@1 0.790–0.820 | Marginal → adjust beta, try longer fine-tuning |
| Dev P@1 < 0.790 | Failure → investigate training pair construction; likely too few pairs or sequence truncation |

### What Can Go Wrong

- **Sequence truncation:** Long legal documents truncated to 512 tokens lose structural signal. Mitigation: use the first 200 tokens of query and first 300 of candidate; or use chunked cross-attention.
- **Overfitting to training query topics:** 817 train queries is small for fine-tuning. Use strong dropout (0.15+) and early stopping on dev.
- **Score collapse:** If all candidates receive nearly identical scores, the ranking is random. Detect via score variance per query; if std < 0.01, training failed.
- **How to detect:** Compute dev P@1 after each epoch; if it plateaus or declines after epoch 1, stop.

---

## Experiment 2 — Contrastive Bi-Encoder Fine-Tuning

### Motivation

The current InLegalBERT embeddings are domain-adapted but **not** fine-tuned for the PCR task. Fine-tuning with contrastive loss on (query, positive, hard_negative) triples will push the embedding space to reflect legal precedent similarity.

### Available Positive Pairs

From `data/processed/IL-TUR-PCR/train/hybrid_features.parquet`:

```python
train_df = pd.read_parquet("data/processed/IL-TUR-PCR/train/hybrid_features.parquet")
positives = train_df[train_df["label"] == 1][["query_id", "candidate_id"]]
```

Train split has ~827 queries. The raw text files for queries and candidates are required; locate them in the original dataset directory used by `build_hybrid_feature_dataset.py`.

**Pair budget:** ~827 unique (query, positive) pairs from train.

### Hard Negative Mining Strategy

Stage 1 (lexical):
1. Rank all candidates for each training query by `stage1_score`.
2. Select the top-5 non-positive candidates as **lexical hard negatives** (they fool TF-IDF but are not relevant).

Stage 2 (embedding-based, after first epoch):
3. After 1 epoch, recompute embeddings with the partially fine-tuned encoder.
4. Re-rank and select the new top-5 non-positive candidates as **semantic hard negatives**.
5. Repeat for epoch 2.

**Do not** use all 1,726 candidates as in-batch negatives (memory constraint; GPU VRAM).

### Base Model

`law-ai/InLegalBERT` — already cached at `C:\Users\mohan\.cache\huggingface\hub`. Load with:

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoModel.from_pretrained("law-ai/InLegalBERT")
model = AutoModel.from_pretrained("law-ai/InLegalBERT")
```

### Training Objective

**MultipleNegativesRankingLoss** (InfoNCE / NT-Xent):

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, p^+)/\tau)}{\sum_{j} \exp(\text{sim}(q, p^-_j)/\tau)}$$

where τ = 0.05 (temperature) and negatives include the batch hard negatives.

### Expected Training Time Estimate

- 817 train queries × ~4,320 candidates → ~3.5M candidate-level operations, but training is done on (query, positive, negative) triples not all pairs.
- With batch size 16 triples and ~3,000 training triples: ~188 steps per epoch.
- At ~1 s/step on GPU (BERT-base forward+backward): **~3 minutes per epoch**.
- Without GPU (CPU inference): ~15–20 minutes per epoch.
- Planned: 2–3 epochs with hard negative refresh between epochs.

### Success/Failure Criteria on Dev

| Outcome | Decision |
|---------|----------|
| Dev P@1 ≥ 0.820 | Success → replace InLegalBERT column in parquet, re-run Stage-1 baseline |
| Dev P@1 0.800–0.820 | Marginal → try larger margin, more epochs |
| Dev MRR improved even if P@1 flat | Partial success → combine with cross-encoder |
| Dev P@1 < current baseline | Failure → check if positives are too few; try frozen encoder with MLP head only |

### What Can Go Wrong

- **Catastrophic forgetting:** Fine-tuning may degrade general InLegalBERT capabilities. Use low LR (1e-5) and gradual unfreezing (unfreeze top 4 layers first).
- **Collapse:** All embeddings converge to similar vectors. Detect via cosine similarity distribution of negative pairs — if mean > 0.95, training collapsed. Add temperature warmup.
- **Insufficient positives:** 827 positive pairs is borderline for BERT fine-tuning. Augment by treating dev positives as additional training positives (only for encoder fine-tuning, not for evaluation).
- **How to detect:** Monitor dev P@1 and embedding cosine histograms after each epoch.

---

## Experiment 3 — Rhetorical-Role-Aware Chunk Scoring

### Motivation

Error analysis shows `rhetorical_cosine` is the **only** feature where the true positive wins in 63.6% of rank-2 miss cases. Currently, rhetorical role is used as a single aggregate cosine over 11 role categories. The `RatioOfTheDecision` role (index 8) is specifically the holding of a case — directly relevant for precedent citation. Weighting embeddings by `RatioOfTheDecision` mass at query time should amplify the discriminating signal.

### Rhetorical Distribution Usage

Each document has a `rhetorical_distribution` vector of length 11 (one entry per role, from `src/legal_features/legal_feature_encoder.py`). Role index 8 is `RatioDecidendi`.

**Proposed RatioDecidendi-weighted dense scoring:**

Instead of computing `InLegalBERT(doc_full_text)`, compute:

```
weighted_embed(doc) = ratio_weight * embed(RatioDecidendi_chunks)
                    + (1 - ratio_weight) * embed(full_doc)
```

where `ratio_weight = rhetorical_distribution[8]` (fraction of sentences labeled RatioDecidendi).

For inference, query-document similarity becomes:

```
sim(q, c) = alpha * cosine(embed_q, embed_c_weighted)
           + (1 - alpha) * cosine(embed_q, embed_c_full)
```

### Modification to `src/retrieval/dense_retriever.py`

1. Add a method `encode_with_rhetorical_weight(texts, rhetorical_dists, role_idx=8)`:
   - For each document, extract the sentences/chunks labeled with `role_idx`.
   - Embed those chunks separately.
   - Return a weighted combination: `ratio_weight * chunk_embed + (1 - ratio_weight) * full_embed`.
   - Both embeddings are L2-normalized before combination.

2. At dataset build time (`build_hybrid_feature_dataset.py`), produce a new column `inlegalbert_ratio_weighted_sim` alongside the existing `inlegalbert_sim`.

3. Replace or augment the `qnorm_inlegalbert_sim` feature in Stage-1 with the new column.

### Split Assignment

| Split | Role |
|-------|------|
| Train | Used to fit any new normalizer statistics |
| Dev | Tune the `alpha` mixing weight for chunk vs. full embedding |
| Test | Final evaluation; touch once |

### Success/Failure Criteria on Dev

| Outcome | Decision |
|---------|----------|
| Dev P@1 ≥ 0.800 | Success |
| Dev P@1 0.790–0.800 | Investigate which queries improved; combine with cross-encoder |
| Dev P@1 < 0.7881 (Stage-1 baseline) | Failure → check if RatioDecidendi chunks are too short for meaningful embedding |

### What Can Go Wrong

- **Empty RatioDecidendi chunks:** Some documents may have zero RatioDecidendi sentences. Fall back to full-document embedding when `rhetorical_distribution[8] < 0.05`.
- **Chunk too short for BERT:** RatioDecidendi sections may be 1–2 sentences, giving low-variance embeddings. Use `min_chunk_tokens=30` filter.
- **Annotation noise:** If `rhetorical_distribution` values come from an imperfect classifier, the signal may be noisy. Cross-check with human-annotated cases from the dataset.
- **How to detect:** Compare `rhetorical_cosine` distribution between rank-2 miss promoted vs. demoted queries.

---

## Experiment 4 — Query-Adaptive Fusion (Margin-Gated Ensemble)

### Motivation

The error analysis shows that Stage-2 reranking hurts (−21 net P@1). But this does not mean reranking is universally bad — it promotes 11 queries. The problem is applying reranking indiscriminately. A **margin-based gate** routes high-confidence queries (large Stage-1 margin between rank-1 and rank-2) directly to Stage-1 results, and applies Stage-2 reranking only to low-confidence queries.

### Margin-Based Gate Definition

For each query, compute:

```
margin = max(stage1_score) - second_max(stage1_score)
```

If `margin >= threshold`: **high-confidence** → use Stage-1 score directly.  
If `margin < threshold`: **low-confidence** → apply Stage-2 reranking.

### Calibration of Threshold

From Phase-10 error analysis (`test_margin_gate.py` patterns):

| Threshold | High-conf queries | High-conf P@1 | Low-conf queries | Low-conf P@1 |
|-----------|------------------|---------------|-----------------|--------------|
| 0.04 | ~197 | ~69% | ~37 | ~42% |
| 0.06 | ~185 | ~71% | ~49 | ~39% |
| 0.10 | ~162 | ~74% | ~72 | ~34% |

Tune `threshold` on **dev** (not test). The optimal threshold is where the weighted harmonic mean of both subgroup P@1s is maximized on dev.

### Different Weight Vectors per Regime

| Regime | Recommended fusion weights |
|--------|--------------------------|
| High-confidence | `w_tfidf=0.70, w_dense=0.20, w_legal=0.10` (amplify the signal that's already working) |
| Low-confidence | `w_tfidf=0.50, w_dense=0.30, w_legal=0.20` + Stage-2 reranker at alpha=0.3 (seek richer signal) |

These are starting points; tune both weight vectors jointly on dev using grid search.

### Features to Distinguish Regimes

The gate decision uses only `margin` (computed from Stage-1 scores). Additional features that could improve gate quality:

| Feature | Reasoning |
|---------|-----------|
| `margin` | Primary signal |
| `statute_shared_count` of rank-1 candidate | High statute overlap → high confidence |
| `inlegalbert_sim` of rank-1 candidate | High semantic match → high confidence |
| `rhetorical_cosine` difference (rank-1 − rank-2) | Large rhetorical gap → high confidence |

A logistic classifier trained on **train queries** (with dev for threshold selection) can produce a soft gate probability instead of a hard threshold.

### Split Assignment

| Split | Role |
|-------|------|
| Train | Train the gate classifier (features: margin, statute overlap, etc. → label: was Stage-1 correct?) |
| Dev | Tune threshold and weight vectors; select best configuration |
| Test | Final held-out evaluation; **do not touch until configuration is fixed** |

### Success/Failure Criteria on Dev

| Outcome | Decision |
|---------|----------|
| Dev P@1 ≥ 0.820 | Success — outperforms both Stage-1 and Stage-2 alone |
| Dev P@1 0.800–0.820 | Marginal — investigate which regime fails; may need better gate features |
| Dev P@1 < 0.7881 | Failure — gate is misclassifying queries; check margin distribution overlap between correct/incorrect |

### What Can Go Wrong

- **Dev-test distribution shift in margin:** If dev queries systematically have different margin distributions than test, the threshold selected on dev will be suboptimal on test. Detect by plotting margin histograms for dev vs. test.
- **Gate overfits to dev:** A complex gate classifier trained to optimize dev P@1 may memorize dev query characteristics. Keep the gate simple (logistic regression or threshold on `margin` only) and avoid > 5 features.
- **Low-confidence regime too small:** If only 10–15% of queries fall into low-confidence, the reranker has little opportunity to improve. Check gate size; if too small, lower the threshold.
- **How to detect:** After fixing the configuration on dev, run a single evaluation pass on test. Any further adjustment constitutes test-set overfitting.

---

## Cross-Experiment Constraints

| Constraint | Applies to |
|-----------|-----------|
| Feature normalization statistics must be fitted on train split only, never dev or test | All experiments |
| Hyperparameter selection (alpha, beta, threshold, weights, epochs) must use dev exclusively | All experiments |
| Test is evaluated exactly once per experiment, after all hyperparameters are locked | All experiments |
| Dev metrics must be reported alongside test metrics to detect overfit | All experiments |
| All exploratory analysis on test set (like `scratch/analyze_rank2.py`) must be disclosed | Future publication |

---

## Priority Ordering

Based on the Phase-10 error analysis, recommended priority:

1. **Experiment 4 (Query-Adaptive Fusion)** — highest expected P@1 gain with least implementation risk; uses existing features.
2. **Experiment 3 (Rhetorical-Role-Aware Chunk Scoring)** — targets the strongest error signal; moderate implementation complexity.
3. **Experiment 1 (Cross-Encoder)** — strongest theoretical gain but highest compute and training risk.
4. **Experiment 2 (Contrastive Bi-Encoder)** — longer-term investment; requires full re-embedding of the corpus.

---

*This plan was produced based on Phase-10 audit and error analysis results.*  
*All cited metric values come from `results/evaluation/phase10_query_error_analysis.csv`.*
