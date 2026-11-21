# Phase 18: Jud-IPL Forensic Audit & Research Foundation Report

**Project Title:** Enhanced Legal Document Similarity and Case Retrieval Using Hybrid BERT, Legal Knowledge, and Ensemble Learning  
**Phase:** 18A & 18B — Standalone Jud-IPL Forensic Audit & Research Problem Definition  
**Date:** November 2026  
**Scope:** Standalone Jud-IPL Research System (Strictly Independent of IL-TUR)  
**Status:** Audit & Feasibility Complete; Ready for Dataset Preparation & Modeling

---

## 1. Executive Summary & Strict Scope Compliance

In accordance with Phase 18 instructions:
- **Zero IL-TUR Dependency:** No data, queries, splits, checkpoints, or relevance pairs from IL-TUR PCR (Phases 1–17) are imported or utilized.
- **No Relevance Label Fabrication:** Jud-IPL outcome labels (`Accepted`, `Rejected`, `Other`) are **not** treated as prior case retrieval relevance labels. Precedent retrieval is explicitly rejected as a supervised benchmark due to the absence of query-precedent ground truth.
- **Standalone Asset:** The research system operates exclusively on:
  - `data/processed/judipl/documents.parquet` (42,006 unique Indian Supreme Court judgments)
  - `data/processed/judipl/chunks.parquet` (40,000 rhetorical section chunks)
  - `data/processed/judipl/legal_features.json` (42,006 legal feature records)

---

## 2. Phase 18A: Dataset Forensic Audit

### 2.1 Complete Dataset Schema & Column Completeness

The dataset consists of **42,006 unique Indian Kanoon judgment records** with 19 columns:

| Column Name | Data Type | Completeness (%) | Non-Empty Count | Unique Values | Description |
|---|---|---:|---:|---:|---|
| `id` | `str` | 100.0% | 42,006 | 42,006 | Standardized 10-digit Kanoon ID |
| `raw_id` | `int64` | 100.0% | 42,006 | 42,006 | Numeric Kanoon document identifier |
| `raw_name` | `str` | 100.0% | 42,006 | 42,006 | Original filename (e.g. `74634371.txt`) |
| `case_category` | `str` | 82.51% | 34,658 | 3 | Legal domain (`civil`, `criminal`, or unassigned) |
| `case_type` | `str` | 98.72% | 41,469 | 13 | Procedural type (`appeal`, `writ petition`, `SLP`, etc.) |
| `label` | `str` | 100.0% | 42,005 | 4 | Outcome label (`Accepted`, `Rejected`, `Other`) |
| `petitioner` | `str` | 34.40% | 14,452 | 11,899 | Extracted petitioner party name |
| `respondent` | `str` | 71.14% | 29,882 | 20,836 | Extracted respondent party name |
| `date_of_judgment` | `str` | 40.10% | 16,845 | 6,319 | Parsed judgment date (DD/MM/YYYY) |
| `bench` | `str` | 42.32% | 17,775 | 7,856 | Coram / presiding judges |
| `citation_meta` | `str` | 20.90% | 8,779 | 8,679 | Law report citation in case header |
| `proof_sentence` | `str` | 100.0% | 42,005 | 27,152 | Dispositive holding / ruling sentence |
| `char_length` | `int64` | 100.0% | 42,006 | 19,807 | Total document character count |
| `word_length` | `int64` | 100.0% | 42,006 | 5,905 | Total word token count |
| `sentence_count` | `int64` | 100.0% | 42,006 | 1,079 | Number of sentences in judgment |
| `statute_count` | `int64` | 100.0% | 42,006 | 13 | Number of detected statutory provisions |
| `citation_count` | `int64` | 100.0% | 42,006 | 30 | Number of detected judicial citations |
| `full_text` | `str` | 100.0% | 42,006 | 40,978 | Complete normalized judgment text |
| `source` | `str` | 100.0% | 42,006 | 1 | Corpus origin identifier (`Jud-IPL`) |

### 2.2 Text Statistics & Structural Quality

- **Document Lengths:**
  - **Characters:** Median = 13,504 | Mean = 15,727.8 | Min = 53 | Max = 32,765
  - **Words:** Median = 2,309 | Mean = 2,688.2 | Min = 7 | Max = 6,591
  - **Sentences:** Median = 82 | Mean = 125.7 | Min = 1 | Max = 13,639
- **Truncation Artifact:** 6,867 judgments (16.35%) are truncated at exactly **32,765 characters**. This stems from the Excel/CSV cell length limit present in the original Kaggle raw export (`case_files_total.csv`). The remaining 35,139 judgments (83.65%) are complete.
- **Short Judgments:** Only 214 judgments (<0.5%) contain fewer than 500 characters (primarily brief order sheets, administrative adjournments, or procedural dismissals).

### 2.3 Exact and Near-Duplicate Analysis

- **Unique IDs:** All 42,006 Kanoon IDs are unique.
- **Exact Full-Text Duplicates:** 1,028 documents (2.45%) share identical text with another case ID.
  - *Root Cause:* In Indian Supreme Court jurisprudence, batch appeals (connected appeals, cross-appeals, or companion writ petitions) are disposed of via a single joint judgment published under multiple docket/Kanoon numbers.
  - *Data Integrity Constraint:* **Crucial.** Train, validation, and test splits must be grouped by full-text hash to prevent joint-appeal leakage between partitions.
- **Prefix Duplicates (First 500 characters):** 1,153 documents (2.74%) share identical header boilerplate.

### 2.4 Label Semantics and Distribution

The target labels represent judicial outcome / disposition:

| Label | Definition in Indian Jurisprudence | Count | Percentage |
|---|---|---:|---:|
| `Rejected` | Appeal/petition dismissed; lower court decree/order upheld | 18,836 | 44.84% |
| `Accepted` | Appeal/petition allowed; lower court order set aside or remanded | 16,684 | 39.72% |
| `Other` | Disposed with directions, withdrawn, infructuous, or partially allowed | 6,485 | 15.44% |
| *(Empty)* | Unlabeled record | 1 | <0.01% |
| **Total** | | **42,006** | **100.00%** |

#### Cross-Tabulation by Legal Category:
- **Civil Cases (27,650):** `Rejected` = 12,462 (45.1%) | `Accepted` = 11,246 (40.7%) | `Other` = 3,941 (14.3%)
- **Criminal Cases (7,008):** `Rejected` = 3,261 (46.5%) | `Accepted` = 2,862 (40.8%) | `Other` = 885 (12.6%)
- **Unassigned Category (7,348):** `Rejected` = 3,113 (42.4%) | `Accepted` = 2,576 (35.1%) | `Other` = 1,659 (22.6%)

> [!NOTE]
> The binary subset (`Accepted` vs `Rejected`) contains **35,520 cases** with a 53.0% : 47.0% distribution. This provides a balanced, statistically sound benchmark for binary legal outcome prediction.

### 2.5 Proof Sentence Audit & The "Dispositive Leakage" Challenge

- **Coverage:** 42,005 / 42,006 records (100.0%) contain a `proof_sentence`.
- **Nature of `proof_sentence`:** It is the court's definitive holding statement (e.g., *"We are of the view that this appeal must be allowed and the judgment of the High Court is set aside..."* or *"Consequently, this appeal is dismissed"*).
- **Position:** In 89.4% of non-truncated cases where the sentence is located, it appears in the final 20% of the document.
- **Dispositive Terms in Full Text:** Phrases like `"appeal is allowed"` occur in 15.2% of cases, and `"appeal is dismissed"` in 4.7%.
- **Critical Risk:** If an NLP model is trained on unmasked full judgment text, it simply memorizes final disposition phrases rather than learning legal reasoning. **To ensure scientific validity, all disposition sentences and concluding ruling paragraphs must be stripped/masked during outcome prediction.**

### 2.6 Provenance and Licensing

- **Origin:** Hosted on Kaggle ("Jud-IPL: Indian Legal Judgment Prediction dataset").
- **Source Authority:** Judgments from the Supreme Court of India obtained via Indian Kanoon.
- **License:** Research-only usage under academic fair use. Not approved for commercial redistribution without independent source licensing.

---

## 3. Phase 18B: Research Problem Definition & Task Feasibility

We evaluated five candidate research tasks against available supervision and data integrity:

| Task ID | Candidate Task | Ground Truth Available? | Coverage | Feasibility Verdict | Selected Role |
|---|---|---|---:|---|:---:|
| **T1** | **Legal Outcome Prediction (LJP)** | **Yes** (`Accepted` / `Rejected` / `Other`) | 100.0% | **HIGHLY FEASIBLE** | **Primary Benchmark** |
| **T2** | **Legal Holding Extraction** | **Yes** (`proof_sentence` holding target) | 83.7% | **FEASIBLE** | **Secondary Task** |
| **T3** | **Legal Category Classification** | **Yes** (`case_category`: Civil vs Criminal) | 82.5% | **FEASIBLE** | **Auxiliary Task** |
| **T4** | **Unsupervised Semantic Clustering** | **Unsupervised** (Extrinsic metadata alignment) | 100.0% | **FEASIBLE** | **Exploratory Analysis** |
| **T5** | **Prior Case Precedent Retrieval** | **NO** (Zero query-candidate relevance pairs) | 0.0% | **NOT DEFENSIBLE** | **REJECTED** |

### 3.1 Task Selection & Justification

1. **Selected Primary Task: Leakage-Free Legal Outcome Prediction (LJP)**
   - *Task:* Given the factual background, procedural history, legal arguments, and statutory references of an Indian Supreme Court case (with dispositive holding sentences strictly masked), predict whether the appeal/petition will be `Accepted` or `Rejected` (binary), or `Accepted` / `Rejected` / `Other` (3-class).
   - *Scientific Value:* Evaluates whether domain-adapted transformers (InLegalBERT) and legal feature extractors can infer judicial outcomes from legal reasoning and factual merits.
   - *Target Metrics:* Accuracy, Macro-F1, Weighted-F1, Precision, Recall, ROC-AUC.

2. **Selected Secondary Task: Legal Holding / Ratio Extraction**
   - *Task:* Given the body of a judgment, identify and rank the sentence that embodies the court's core holding (`proof_sentence`).
   - *Target Metrics:* Sentence Top-1 Accuracy, MRR, Sentence F1.

3. **Explicitly Rejected Task: Supervised Precedent Retrieval (T5)**
   - *Rationale:* Jud-IPL contains **no** query-candidate relevance matrices. Outcome labels (`Accepted`/`Rejected`) are appeal dispositions, not precedent relevance judgments.
   - *Integrity Rule:* Fabricating relevance pairs from outcome labels or co-citations would violate experimental integrity.

---

## 4. Phase 18C–18E Roadmap

### 4.1 Phase 18C: Dataset Preparation (Leakage-Free Partitioning)

- **Grouping by Text Hash:** Group all 1,028 duplicate and connected judgments together.
- **Stratified Partition:** Split data 70% Train (29,404), 15% Validation (6,301), 15% Test (6,301), stratified by `label` and `case_category`.
- **Temporal Splitting Analysis:** For the subset of 16,845 records with confirmed dates, evaluate chronological holdout (pre-2000 train vs. post-2000 test) to test temporal generalization.
- **Masking Protocol:** Implement an automated pipeline that strips the `proof_sentence` and truncates the trailing dispositive paragraph to create `masked_text` for fair LJP evaluation.

### 4.2 Phase 18D: Model Development Architecture

1. **Baseline 1:** Majority class baseline & Random prior.
2. **Baseline 2:** Lexical TF-IDF + Logistic Regression / Linear SVM on masked text.
3. **Baseline 3:** Legal Feature Baseline (statute counts, citation counts, case type, coram size, document length) + LightGBM.
4. **Baseline 4:** Hybrid Shallow Classifier (TF-IDF + Legal Features + LightGBM).
5. **Neural Model 1:** InLegalBERT Domain Adapter (fine-tuned on masked legal facts and ratio chunks).
6. **Neural Model 2:** Section-Aware Bi-Encoder with Rhetorical Chunk Attention (leveraging `chunks.parquet`).

### 4.3 Phase 18E: Rigorous Evaluation Protocol

- Report 5-fold cross-validation or frozen train/val/test splits.
- Compute Accuracy, Precision, Recall, Macro-F1, Weighted-F1, and ROC-AUC for both Binary and 3-Class tasks.
- Perform ablation studies:
  - Impact of dispositive sentence masking (quantifying the "leakage gap").
  - Contribution of statutory/citation metadata vs. raw text representations.
  - Section-level ablations (`facts` only vs. `facts` + `headnote` vs. full masked body).
- Validate with unit tests and statistical significance tests.

---

## 5. Artifact Manifest

| Artifact | File Path | Status |
|---|---|:---:|
| Column Audit CSV | [`results/tables/phase18_judipl_column_audit.csv`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/phase18_judipl_column_audit.csv) | ✅ Generated |
| Label Distribution CSV | [`results/tables/phase18_judipl_label_distribution.csv`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/phase18_judipl_label_distribution.csv) | ✅ Generated |
| Task Feasibility CSV | [`results/tables/phase18_judipl_task_feasibility.csv`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/phase18_judipl_task_feasibility.csv) | ✅ Generated |
| Audit Script | [`scripts/audit_phase18_judipl.py`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/scripts/audit_phase18_judipl.py) | ✅ Verified |
| Unit Tests | [`tests/unit/test_phase18_judipl_audit.py`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/tests/unit/test_phase18_judipl_audit.py) | ✅ 11/11 Passed |
| Research Foundation Report | [`results/reports/phase18_judipl_research_foundation.md`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/reports/phase18_judipl_research_foundation.md) | ✅ Complete |

