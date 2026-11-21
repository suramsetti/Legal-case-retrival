# Phase 13 Initial Architecture Audit Report

**Project:** Enhanced Legal Document Similarity and Case Retrieval Using Hybrid BERT, Legal Knowledge, and Ensemble Learning  
**Date:** November 2026  
**Auditor:** Antigravity Autonomous Research Assistant  
**Objective:** Baseline architectural audit prior to Gemini API integration (Preserving Phase 11 frozen production pipeline).  

---

## 1. System Overview & Repository Layout

The repository implements a state-of-the-art Two-Stage Legal Prior Case Retrieval (PCR) architecture tailored for the Indian Legal domain (specifically the IL-TUR PCR benchmark from Supreme Court of India judgments).

```
legal-case-retrieval/
├── backend/                       # Web service and REST API
│   ├── app.py                     # Flask application factory
│   ├── api/routes.py              # REST API blueprints (/api/search, /api/case, /api/explanation, /api/metrics)
│   ├── services/
│   │   ├── retrieval_service.py   # Hybrid case retrieval service (Candidate pools, TF-IDF, InLegalBERT, Knowledge)
│   │   └── explanation_service.py # Feature attribution, SHAP-inspired explanations, modality breakdown
│   └── schemas/                   # Data contract definitions
├── frontend/
│   └── index.html                 # Single-Page React 18 application with Tailwind CSS (Babel CDN)
├── src/
│   ├── ranking/
│   │   ├── adaptive_reranker.py   # Phase 11 Query-Adaptive Margin-Gated Re-Ranker (Production Baseline)
│   │   ├── legal_cross_encoder.py # Phase 12 InLegalBERT Joint Cross-Encoder & SectionAwareChunker
│   │   ├── adaptive_cross_reranker.py # Phase 12 Margin-Gated Cross-Encoder pipeline
│   │   ├── pairwise_reranker.py   # Phase 9 Pairwise Neural Feature Network (MLP)
│   │   ├── weighted_fusion.py     # Phase 6B.1 Frozen Linear Fusion (0.65 Lexical, 0.25 Dense, 0.10 Legal)
│   │   ├── lightgbm_ranker.py     # LambdaMART Ranker
│   │   ├── xgboost_ranker.py      # Pairwise XGBoost Ranker
│   │   └── logistic_stacking.py   # Leakage-Safe Stacking Meta-Learner
│   ├── legal_features/
│   │   ├── rhetorical_analyzer.py # 11-role Indian Legal Discourse Parser (Ratio Decidendi, Ruling, Facts)
│   │   ├── statute_extractor.py   # Indian Penal Code & statutory provisions extractor
│   │   └── legal_kg_builder.py    # Legal Knowledge Graph & citation network builder
│   ├── dense/
│   │   └── inlegalbert_retriever.py # InLegalBERT 768-dim mean-pooled dense embeddings
│   ├── lexical/
│   │   ├── tfidf_retriever.py     # Sublinear TF-IDF retrieval
│   │   └── bm25_retriever.py      # BM25Okapi baseline
│   └── evaluation/
│       ├── hybrid_evaluator.py    # Query-level P@1, P@5, P@10, Recall, MRR, nDCG evaluator
│       └── bootstrap_evaluator.py # 2,000 paired bootstrap resamples with 95% CIs and p-values
├── models/checkpoints/            # Frozen checkpoints & configs
│   ├── weighted_fusion_weights.json # Phase 6B.1 weights (TF-IDF: 0.65, Dense: 0.25, Legal: 0.10)
│   ├── pairwise_neural_reranker.pt  # Phase 9 trained MLP weights
│   ├── phase11_adaptive_config.json # Phase 11 frozen config (tau=0.07, alpha=0.12, gamma=0.04)
│   ├── legal_cross_encoder.pt       # Phase 12 trained Cross-Encoder weights (440 MB)
│   └── phase12_adaptive_config.json # Phase 12 config (tau=0.05, beta=0.10, top_n=5)
├── data/processed/IL-TUR-PCR/     # Official IL-TUR PCR benchmark
│   ├── train/                     # 120 queries, hybrid_features.parquet
│   ├── dev/                       # 118 queries, hybrid_features.parquet
│   ├── test/                      # 237 queries (234 eligible non-zero relevance), hybrid_features.parquet
│   ├── test_candidates/           # 1,727 candidate judgments (documents.parquet, chunks.parquet, legal_features.json)
│   └── test_queries/              # 237 query judgments (documents.parquet, chunks.parquet, legal_features.json)
└── results/                       # Empirical reports, tables, figures, explainability samples
```

---

## 2. Frozen Production Baseline: Phase 11 Architecture

### 2.1 Core Routing Mechanism (`QueryAdaptiveReRanker`)
- **Stage 1**: Query-level min-max scaled score fusion:
  $$s_{\text{Stage1}}(q, c) = 0.65 \cdot s_{\text{tfidf\_norm}} + 0.25 \cdot s_{\text{dense\_norm}} + 0.10 \cdot s_{\text{legal\_norm}}$$
- **Confidence Margin Calculation**:
  $$\text{margin}(q) = s_{\text{Stage1}}(q, \text{Rank 1}) - s_{\text{Stage1}}(q, \text{Rank 2})$$
- **Margin Gating Rule**:
  - If $\text{margin}(q) \ge \tau$ ($\tau = 0.07$): Query is classified as **High Confidence**. The verified Stage-1 ordering is strictly preserved.
  - If $\text{margin}(q) < \tau$: Query is classified as **Low Margin / Ambiguous**. Stage-2 pairwise reranking is applied over Top-$K$ ($K=30$):
    $$s_{\text{adaptive}}(q, c) = (1 - \alpha - \gamma) \cdot s_{\text{Stage1}} + \alpha \cdot s_{\text{pairwise\_norm}} + \gamma \cdot s_{\text{rhetorical}}$$
- **Frozen Hyperparameters**:
  `{"tau": 0.07, "alpha": 0.12, "gamma": 0.04, "top_k": 30}`
- **Test Set Performance (234 Eligible Queries)**:
  - **P@1**: 66.24% (+2.14% over Stage 1)
  - **MRR**: 0.7660 (+0.0188 over Stage 1, $p = 0.033$, statistically significant)
  - **nDCG@10**: 0.5904
  - **Latency**: 11.48 ms/query on CPU

---

## 3. Phase 12 Cross-Encoder & Discourse Chunking

- **Model**: `InLegalBERTCrossEncoder` (`law-ai/InLegalBERT` backbone).
- **Discourse Parser**: `SectionAwareChunker` partitions judgments into functional units:
  - **Ratio Decidendi**: Judicial holdings (`"we hold"`, `"rule of law"`), weight $1.25\times$.
  - **Ruling**: Dispositions (`"appeal is allowed"`, `"appeal dismissed"`), weight $1.15\times$.
  - **Headnotes**: Procedural citations, framing issues, weight $1.00\times$.
  - **Facts**: Factual background, weight $0.90\times$.
- **Test Metric**: P@1: 65.38%, MRR: 0.7522; Latency: 2,603.94 ms/query.

---

## 4. Existing Web Application Integration Points

### 4.1 REST API Routes (`backend/api/routes.py`)
- `GET /api/health`: Health check, returns current models, candidate count, normalization type.
- `POST /api/search`: Accepts `{"query": str, "top_k": int, "filter_statute": Optional[str]}`. Runs `RetrievalService.search()`.
- `GET /api/case/<case_id>`: Returns full judgment text, word count, sentence count, detected statutes, citations, and judges.
- `GET /api/explanation/<query_id>/<case_id>`: Returns score decomposition, feature attribution, and summary.
- `GET /api/metrics`: Serves official test evaluation benchmark.
- `GET /api/sample-queries`: Serves sample query cases for one-click testing.

### 4.2 Frontend Application (`frontend/index.html`)
- Interactive single-page React 18 app with navigation tabs: `search`, `results`, `case_details`, `explainability`, `metrics`.
- Displays candidate rankings, hybrid scores, ground-truth badges, statute filter pills, and modality contribution breakdowns.

---

## 5. Gemini API Integration Points for Phase 13

| Component | Target Location | Input Interface | Output Interface | Fallback Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Configuration** | `backend/config/gemini_config.py` | Env vars (`GEMINI_API_KEY`, `GEMINI_MODEL`) | Validated Config Object | Disable LLM gracefully if key missing |
| **Service Layer** | `backend/services/gemini_service.py` | Text prompts, schema constraints | Structured Dict / Pydantic JSON | Cached result or None |
| **Module A: Query Expansion** | `backend/services/legal_query_expansion.py` | Raw query string | Issues, concepts, statutes, synonyms | Fallback to raw query |
| **Module B: Ratio Extraction** | `backend/services/legal_reasoning_extractor.py` | Judgment chunks & text | Grounded facts, issues, ratio, ruling | Fallback to regex/rhetorical parser |
| **Module C: Precedent Reranking** | `backend/services/gemini_precedent_reranker.py` | Query + Top-K candidate pairs | Structured alignment scores + evidence | Fallback to Phase 11 baseline ranks |
| **Module D: Grounded Explanation** | `backend/services/legal_explanation_service.py` | Query + Top-1 Precedent | 8-point grounded explanation | Fallback to SHAP/Linear decomposition |
| **Frontend Integration** | `frontend/index.html` | REST API responses | UI badges, comparison cards, AI toggle | Render Phase 11 baseline if Gemini off |

---

## 6. Data Integrity & Leakage Prevention Rules

1. **Test Set Isolation**:
   - The official 237-query test set has been historically inspected. It is designated as an **exploratory benchmark**.
   - No Gemini prompts, scoring weights, or routing thresholds will be fitted, selected, or tuned using test set labels.
   - All prompt engineering, threshold calibration ($\tau$), and fusion weight sweeps ($\alpha, \beta, \gamma$) are conducted strictly on the **118 development queries**.
2. **Deterministic Evidence Grounding**:
   - Gemini cannot invent citations, statutes, or holdings.
   - Every generated legal claim must carry chunk ID pointers to the source text.
3. **Phase 11 Preservation**:
   - Phase 11 baseline code, weights, and routes remain 100% executable and unmodified.

