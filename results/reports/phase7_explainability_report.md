# Phase 7: Explainable AI and Legal Retrieval Interpretability Report

**Project**: Enhanced Legal Document Similarity and Case Retrieval Using Hybrid BERT, Legal Knowledge, and Ensemble Learning  
**Target Architecture**: Validated Weighted Linear Score Fusion ($w_{\text{bm25}} = 0.00, w_{\text{tfidf}} = 0.65, w_{\text{dense}} = 0.25, w_{\text{legal}} = 0.10$)  
**Evaluation Scope**: Development Split (118 queries × 1,022 non-self candidates = 120,596 pairs, 973 true citations)  
**Strict Protocol**: Frozen checkpoint weights; zero model retuning; strictly zero test-set access.

---

## 1. Executive Summary & Attribution Overview

Phase 7 delivers an interpretability and Explainable AI (XAI) framework for the validated hybrid prior-case retrieval system. By design, the selected architecture provides two complementary levels of explanation:
1. **Exact Mathematical Score Decomposition**: Additive decomposition where each modality's contribution equals $w_i \cdot \tilde{S}_i(d)$, summing identically to $\hat{S}_{\text{hybrid}}(d)$.
2. **SHAP Feature Attribution**: Model-agnostic Shapley explanations measuring how much each signal drives a candidate's score above or below the expected baseline score across the candidate pool.

### Global Attribution Summary Across Retrieval Slices:
| Candidate Pool Slice | TF-IDF Mean Contrib | TF-IDF Pct Share | Dense Mean Contrib | Dense Pct Share | Legal Mean Contrib | Legal Pct Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Pairs** | 0.2167 | **48.3%** | 0.2038 | **45.5%** | 0.0277 | **6.2%** |
| **Top1 Candidates** | 0.6439 | **68.7%** | 0.2347 | **25.0%** | 0.0588 | **6.3%** |
| **Top5 Candidates** | 0.5520 | **66.4%** | 0.2307 | **27.7%** | 0.0490 | **5.9%** |
| **Top10 Candidates** | 0.5068 | **64.8%** | 0.2294 | **29.3%** | 0.0464 | **5.9%** |
| **True Positives** | 0.4148 | **61.3%** | 0.2206 | **32.6%** | 0.0410 | **6.1%** |

---

## 2. Mathematical Score Contribution vs. SHAP Explainability

### Mathematical Score Attribution (Additive Decomposition)
The hybrid score is an exact additive function of normalized component similarities:
$$\hat{S}_{\text{hybrid}}(d) = 0.65 \cdot \tilde{S}_{\text{tfidf}}(d) + 0.25 \cdot \tilde{S}_{\text{inlegalbert}}(d) + 0.10 \cdot \tilde{S}_{\text{legal}}(d)$$

- **TF-IDF Contribution**: Accounts for **68.7%** of the total score for top-1 ranked candidates, acting as the primary statutory anchor.
- **InLegalBERT Contribution**: Accounts for **25.0%** of the top-1 score, bridging vocabulary gaps through masked mean pooled contextual representations.
- **Legal Knowledge Contribution**: Accounts for **6.3%** of the top-1 score, serving as an authority filter (shared IPC sections, reported citations, coram overlap).

### SHAP Explainability (Linear Closed-Form Shapley Values)
Because the model is a linear score fusion architecture rather than a decision tree ensemble, TreeSHAP was rejected. Instead, we implemented exact closed-form linear Shapley attributions against a background reference sample of 500 development candidate pairs:
$$\phi_i(x) = w_i \cdot (x_i - \mathbb{E}_{X}[X_i])$$
$$\text{Base Value (Expected Score)} \mathbb{E}[f(X)] = 0.44480$$

| Feature Name | Model Weight | Mean Absolute SHAP ($E[|\phi_i|]$) | Mean SHAP Direction | Primary Role |
| :--- | :---: | :---: | :---: | :--- |
| **tfidf_score** | 0.65 | **0.07565** | +0.00280 | Dominant rank discriminator |
| **inlegalbert_sim** | 0.25 | **0.01902** | -0.00023 | Semantic smoothing & recall expansion |
| **composite_legal_score** | 0.10 | **0.01416** | +0.00085 | Substantive legal authority anchor |
| **bm25_score** | 0.00 | **0.00000** | +0.00000 | Excluded (zero weight) |

---

## 3. Case-Level Explanation Generator: Real-World Case Studies

The system generates factual, auditable structured explanations for retrieved candidates. All legal citations, statutes, and judicial coram overlaps are pulled directly from pre-extracted knowledge structures without hallucination.

### Case Study: Query `0000012637` -> Top-Ranked Candidate `0001853481`
- **Candidate Status**: **Ground-Truth Precedent Citation** (Rank #1)
- **Overall Hybrid Score**: **0.8969**
- **Additive Score Breakdown**:
  - TF-IDF Lexical Match: **0.6500** (72.5%)
  - InLegalBERT Dense Semantics: **0.2126** (23.7%)
  - Legal Knowledge Overlap: **0.0343** (3.8%)
- **Factual Legal Entities Extracted**:
  - Shared Statutory Provisions: `None detected in formal title block`
  - Shared Formal Citations: `None directly cited`
  - Shared Judicial Coram: `Different Benches`
  - Jurisdictional Forum: `['SUPREME_COURT_OF_INDIA']`
  - Rhetorical Role Similarity: **0.9884** (Cosine similarity across 11 rhetorical discourse roles)
- **Substantive Factors**:
  - *Matching Factors*: Strong lexical phrasing overlap (TF-IDF score 1.00/1.00); High semantic similarity via InLegalBERT (0.85/1.00); Jurisdictional forum: SUPREME_COURT_OF_INDIA; Highly compatible rhetorical discourse structure (similarity 0.99)
  - *Non-Matching Factors*: No reported precedent citations shared with query

---

## 4. Explainable Ranking Movement Analysis

By comparing rankings under TF-IDF alone, InLegalBERT alone, Legal Knowledge alone, and Full Hybrid fusion, the system reveals why candidates shift positions:

### 1. Precedent Promotion Example (Query `0000012637`):
- **Candidate `0001043522`** (True Precedent):
  - **TF-IDF Rank**: #3
  - **InLegalBERT Dense Rank**: #1 (Strong semantic match!)
  - **Legal Knowledge Rank**: #30
  - **Full Hybrid Rank**: **#2** (Promoted by **+1** positions!)
  - **Explainability Driver**: *Promoted due to strong dense semantic alignment (Dense Rank #1).*

### 2. False Positive Demotion Example (Query `0000012637`):
- **Candidate `0000676779`** (Non-Precedent with superficial lexical match):
  - **TF-IDF Rank**: #6 (Top-10 false alarm)
  - **InLegalBERT Dense Rank**: #464 (Severe semantic mismatch)
  - **Legal Knowledge Rank**: #37
  - **Full Hybrid Rank**: **#8** (Demoted by **-2** positions!)
  - **Explainability Driver**: *Correctly demoted due to poor semantic similarity (Dense Rank #464) and lack of statutory/citation overlap (Legal Rank #37).*

---

## 5. Artifact and Figure Manifest

The following deliverables have been generated and validated:

- **Core Explainability Modules**:
  - `src/explainability/feature_attribution.py`: Exact mathematical score decomposition.
  - `src/explainability/shap_explainer.py`: Closed-form linear SHAP explainer with efficiency verification.
  - `src/explainability/case_explanation.py`: Domain-specific case-level structured explanation generator.
  - `src/explainability/ranking_comparison.py`: Multi-model rank transition and movement analyzer.
- **Master Pipeline & Notebook**:
  - `scripts/run_explainability.py`: Full explainability generation script.
  - `notebooks/08_explainable_legal_retrieval.ipynb`: Interactive Jupyter analysis notebook.
- **Unit Tests**:
  - `tests/unit/test_explainability.py`: 6/6 tests passing in 0.36s.
- **Persisted JSON & CSV Data**:
  - `results/explainability/global_feature_attribution.json`
  - `results/explainability/modality_contributions.csv`
  - `results/explainability/shap_summary_metrics.json`
  - `results/explainability/case_explanations_sample.json`
  - `results/explainability/ranking_comparison_examples.json`
- **High-Resolution Figures (300 DPI)**:
  - `results/figures/global_feature_importance.png`
  - `results/figures/modality_contributions.png`
  - `results/figures/shap_summary_plot.png`
  - `results/figures/score_decomposition_examples.png`
  - `results/figures/ranking_movement_slope.png`

---
*Report generated and validated under Phase 7 Explainable AI standards.*
