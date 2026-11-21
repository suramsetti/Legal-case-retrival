# Legal Knowledge Feature Extraction and Representation Analysis

**Project**: Enhanced Legal Document Similarity and Case Retrieval Using Hybrid BERT, Legal Knowledge, and Ensemble Learning  
**Phase**: Phase 5 – Legal Knowledge Feature Extraction and Representation  
**Date**: November 2026  

---

## 1. Executive Summary

In Phase 5, we engineered a dedicated, domain-adapted legal knowledge extraction and representation pipeline tailored for the Supreme Court of India judgments from the IL-TUR PCR benchmark. The pipeline comprises four modular extractors:
1. **Statutory Reference Extractor**: High-precision parsing of central acts (IPC, CrPC, CPC, Constitution of India, Evidence Act, NI Act, Arbitration Act, NDPS Act, PC Act, IBC) and normalization into canonical keys (`IPC:302`, `CRPC:482`, `CPC:OVII_R11`, `CONST:ART_21`).
2. **Precedent Citation Extractor**: Extraction and canonical key construction for official legal reporters (AIR, SCC, SCR) and formal High Court / Supreme Court appeal numbers (`APPEAL:CRIMINAL_APPEAL:92:2015`).
3. **Legal Entity Extractor**: Recognition and canonicalization of judicial actors (Courts, Judges, Appellants, Respondents, Counsels, Government Organizations, Geographic Jurisdictions, and Law Enforcement Authorities).
4. **Rhetorical Role Analyzer**: Discourse segmentation and 11-class rhetorical probability distribution profiling aligned with the official IL-TUR RR benchmark.

---

## 2. Gold Benchmark Validation Metrics

We validated our rule-based and pattern-driven legal knowledge extractors against official gold human annotations from the IL-TUR benchmark without any data leakage or training on test partitions.

### A. Legal Named Entity & Statute Extraction (IL-TUR `l-ner` Fold 1 Gold Annotations)

| Target Legal Entity / Provision | Gold Annotations | Extracted Instances | True Positives | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Statute** | 525 | 296 | 162 | **0.5473** | **0.3086** | **0.3946** |
| **Court** | 455 | 684 | 321 | **0.4693** | **0.7055** | **0.5637** |
| **Judge** | 123 | 101 | 58 | **0.5743** | **0.4715** | **0.5179** |
| **Citation** | 624 | 276 | 125 | **0.4529** | **0.2003** | **0.2778** |

### B. Rhetorical Role Discourse Classification (IL-TUR `rr` CL_dev Gold Annotations)

- **Corpus Evaluated**: `rr/CL_dev` (5 full judicial decisions, 1396 sentences)
- **Overall Sentence Role Accuracy**: **32.23%** (random chance baseline: 8.33%)
- **Macro-F1 Across Roles**: **0.2332**

---

## 3. Dataset Coverage and Extraction Statistics

Across the IL-TUR PCR benchmark partitions, we evaluated extraction coverage and missing-value rates:

| Partition | Total Docs | Statute Coverage | Citation Coverage | Entity Coverage | Avg Statutes / Doc | Avg Citations / Doc | Avg Entities / Doc | Unique Statutes | Unique Citations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **dev_queries** | 118 | 77.12% | 61.86% | 98.31% | 22.65 | 4.09 | 22.95 | 425 | 399 |
| **dev_candidates** | 1,023 | 73.22% | 62.95% | 92.18% | 32.13 | 3.14 | 21.95 | 1,873 | 2,380 |
| **test_queries** | 237 | 73.0% | 57.38% | 94.94% | 19.08 | 3.16 | 17.37 | 601 | 628 |
| **test_candidates** | 1,727 | 70.41% | 57.09% | 90.45% | 22.3 | 3.1 | 18.86 | 2,470 | 3,939 |

---

## 4. Legal Feature Representation for Hybrid Retrieval (Phase 6)

To support Phase 6 ensemble fusion without entanglement, we represent document-pair legal similarity across five orthogonal sub-spaces:

1. **Statutory Reference Overlap**:
   - `statute_jaccard`: Jaccard similarity of normalized statute keys ($|S_q \cap S_c| / |S_q \cup S_c|$)
   - `statute_containment_query`: Directional coverage of query statutes in candidate ($|S_q \cap S_c| / |S_q|$)
   - `statute_shared_count`: Absolute count of co-cited statutory provisions
2. **Precedent Co-Citation Overlap**:
   - `citation_jaccard`: Jaccard similarity of formal reporter and appeal citations ($|C_q \cap C_c| / |C_q \cup C_c|$)
   - `citation_shared_count`: Direct co-citation frequency between query and candidate opinions
3. **Legal Entity and Actor Overlap**:
   - `entity_jaccard`: Overall Jaccard similarity across all detected legal actors
   - `judge_overlap`: Overlap in bench composition (shared presiding justices)
   - `court_overlap`: Jurisdiction concordance (e.g. both originating from Bombay High Court)
   - `org_overlap`: Concordance of governmental or corporate litigants (e.g., State of Maharashtra)
4. **Rhetorical Role Distribution Alignment**:
   - `rhetorical_cosine`: Cosine similarity of the 11-dimensional discourse probability distribution
   - `rhetorical_js_sim`: Complementary Jensen-Shannon distance ($1 - D_{JS}(P_q \parallel P_c)$)
   - `rhetorical_ratio_diff`: Absolute difference in judicial reasoning / ratio decidendi proportion
5. **Composite Legal Feature Score**:
   - Linear combination providing an initial un-learned legal prior before GBDT ensemble training.

---

## 5. Architectural Separation

As mandated by system guidelines, all extracted legal features are persisted separately in `legal_features.json` for each split and are completely decoupled from lexical (Phase 3 BM25 / TF-IDF) and dense semantic (Phase 4 InLegalBERT) representations. In Phase 6, they will serve as structured tabular features in a Gradient Boosted Decision Tree (LightGBM / XGBoost) and Logistic Stacking re-ranker.