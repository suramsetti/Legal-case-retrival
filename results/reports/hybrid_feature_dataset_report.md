# Phase 6A: Hybrid Feature Alignment and Training Dataset Construction Report

**Project**: Enhanced Legal Document Similarity and Case Retrieval Using Hybrid BERT, Legal Knowledge, and Ensemble Learning  
**Phase**: Phase 6A – Feature Alignment, Negative Sampling & Normalization  
**Date**: November 2026  

---

## 1. Executive Summary

In Phase 6A, we constructed the multi-modal training and evaluation datasets that unify Lexical (BM25, TF-IDF), Dense Semantic (InLegalBERT, BERT-Base), and Legal Knowledge (Statutes, Citations, Legal Entities, Rhetorical Roles) features. Strict train/dev/test isolation was maintained, query self-exclusion was enforced, and normalization parameters were fitted strictly on the training partition to ensure zero data leakage.

---

## 2. Dataset Dimensions and Split Statistics

| Partition | Queries | Candidate Pool | Total Pairs | Positive Pairs (Precedents) | Negative Pairs | Pos/Neg Imbalance Ratio | Construction Methodology |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train** | 827 | 4,320 | **15,232** | **5,308** (34.8%) | **9,924** (65.2%) | 1 : 1.87 | 100% Positives + Top-10 BM25 Hard Negatives + 2 Random Negatives |
| **Dev** | 118 | 1,023 | **120,596** | **973** (0.81%) | **119,623** (99.19%) | 1 : 122.9 | Full Cross-Product (Unconstrained Official Evaluation Pool) |
| **Test** | 237 | 1,727 | **409,062** | **1,727** (0.42%) | **407,335** (99.58%) | 1 : 235.9 | Full Cross-Product (Unconstrained Official Evaluation Pool) |

---

## 3. Negative Sampling Methodology (Training Set)

Because the complete training cross-product comprises 3,572,640 pairs with only 5,308 positives (an extreme 1 : 672 class imbalance), training rankers directly on the full matrix causes gradient dilution on millions of uninformative zero-overlap documents. We addressed this through deterministic negative sampling:
- **100% Positive Retention**: All 5,308 valid annotated precedent links in the training pool were retained.
- **Hard Negative Mining**: For each query, the top-10 non-relevant candidates ranked highest by BM25 were sampled (8,270 pairs, 54.3%). These penalize superficial lexical overlap.
- **Random Negative Regularization**: 2 uniform random candidates per query (1,654 pairs, 10.9%) were sampled to anchor global non-relevance.
- **Deterministic Random Seed**: `random_seed = 42`.
- **Evaluation Pools Preserved**: Dev (120,596 pairs) and Test (409,062 pairs) use the full candidate pools without sampling.

---

## 4. Feature Space Specifications

The aligned feature space includes **20 primary features** spanning three distinct retrieval paradigms:

### A. Lexical Modality (4 Features)
- `bm25_score`: Raw BM25 score of candidate given query.
- `bm25_rank`: 1-based rank of candidate under BM25.
- `tfidf_score`: TF-IDF cosine similarity.
- `tfidf_rank`: 1-based rank of candidate under TF-IDF.

### B. Dense Semantic Modality (2 Primary Features + 2 Dev Baselines)
- `inlegalbert_sim`: Cosine similarity of 768-D masked mean pooled InLegalBERT embeddings.
- `inlegalbert_rank`: 1-based rank of candidate under InLegalBERT.
- `bert_base_sim` & `bert_base_rank`: General BERT embeddings (available on dev partition for comparative ablation).

### C. Legal Knowledge Modality (14 Features)
- Statutory: `statute_jaccard`, `statute_containment_query`, `statute_shared_count` (log1p-transformed).
- Citations: `citation_jaccard`, `citation_shared_count` (log1p-transformed).
- Legal Actors: `entity_jaccard`, `entity_shared_count`, `judge_overlap`, `court_overlap`, `org_overlap`.
- Rhetorical Structure: `rhetorical_cosine`, `rhetorical_js_sim`, `rhetorical_ratio_diff`.
- Prior Score: `composite_legal_score`.

---

## 5. Normalization, Skew Handling, and Zero-Leakage Guarantee

1. **Raw Preservation**: All raw values are permanently retained as `raw_{col}` alongside normalized `norm_{col}`.
2. **Domain-Specific Transformations**:
   - Highly skewed count features (`statute_shared_count`, `citation_shared_count`, `entity_shared_count`) are smoothed via $\log(1 + x)$.
   - Ordinal rank features (`bm25_rank`, `tfidf_rank`, `inlegalbert_rank`) are converted to reciprocal ranks: $1 / \text{rank}$.
3. **Split Discipline**:
   - Normalization statistics (means, standard deviations, mins, maxs) were fitted **strictly on the training partition**.
   - Transformations were frozen and applied to `dev` and `test` without re-estimating statistics.
   - Zero missing values (all 20 features have 100% non-null numeric coverage).

---

## 6. Feature Correlation and Alignment Analysis

Top correlations with target relevance label (`label` = 1 for true precedents) on the training set:

| Feature | Modality | Pearson Correlation with Label | Description |
| :--- | :---: | :---: | :--- |
| `norm_bm25_score` | Lexical | **+0.1165** | Normalized indicator of candidate precedent |
| `norm_citation_shared_count` | Legal Knowledge | **+0.1146** | Normalized indicator of candidate precedent |
| `norm_composite_legal_score` | Legal Knowledge | **+0.0960** | Normalized indicator of candidate precedent |
| `norm_statute_jaccard` | Legal Knowledge | **+0.0913** | Normalized indicator of candidate precedent |
| `norm_citation_jaccard` | Legal Knowledge | **+0.0711** | Normalized indicator of candidate precedent |
| `norm_tfidf_score` | Lexical | **+0.0691** | Normalized indicator of candidate precedent |
| `norm_statute_shared_count` | Legal Knowledge | **+0.0605** | Normalized indicator of candidate precedent |
| `norm_inlegalbert_sim` | Dense Semantic | **+0.0434** | Normalized indicator of candidate precedent |
| `norm_inlegalbert_rank` | Dense Semantic | **+0.0392** | Normalized indicator of candidate precedent |
| `norm_rhetorical_ratio_diff` | Legal Knowledge | **+0.0345** | Normalized indicator of candidate precedent |

---

## 7. Downstream Readiness for Phase 6B

The datasets have been persisted in Apache Parquet format under:
- `data/processed/IL-TUR-PCR/train/hybrid_features.parquet`
- `data/processed/IL-TUR-PCR/dev/hybrid_features.parquet`
- `data/processed/IL-TUR-PCR/test/hybrid_features.parquet`

They are immediately ready for training fusion algorithms in Phase 6B, including Reciprocal Rank Fusion (RRF), Weighted Linear Combination, Gradient Boosted Decision Trees (LightGBM Ranker / XGBoost), and Stacking Classifiers.