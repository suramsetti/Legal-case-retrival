# Phase 18D: Baseline Models Execution & Empirical Validation Report

**Date:** November 2026  
**Status:** Completed & Validated  
**Scope:** Jud-IPL Standalone Legal Intelligence Research System (Strictly Jud-IPL dataset, 0% IL-TUR dependency)  
**Dataset:** Dispositive-Remediated Jud-IPL Binary Dataset (`Accepted` vs. `Rejected`)  
**Data Integrity:** Leakage remediated (Ruling phrase detection <= 0.02%, 100% proof sentence masking)  
**Evaluation Protocol:** Evaluated strictly on the Validation set (`val_masked.parquet`, 5,320 documents); Test set labels remain completely unaccessed and frozen.

---

## 1. Executive Summary

In Phase 18D Task 2, three classical and n-gram linear baseline models were trained on the sanitized Jud-IPL training corpus (24,849 documents) and evaluated on the independent validation split (5,320 documents). 

These models establish the true, uninflated performance floor for Indian legal outcome prediction in the complete absence of dispositive leakage (i.e. without access to final holding sentences, concluding paragraphs, or dispositive phrases such as *"appeal is dismissed"* or *"petition allowed"*).

### Key Empirical Findings:
1. **Unigram TF-IDF + Logistic Regression:** Achieves **65.68% Validation Accuracy**, **0.6548 Macro-F1**, and **0.7188 ROC-AUC**.
2. **Unigram TF-IDF + Linear SVM:** Achieves **64.53% Validation Accuracy**, **0.6434 Macro-F1**, and **0.6971 ROC-AUC**.
3. **Word N-gram (1-2) TF-IDF + Logistic Regression:** Achieves **69.32% Validation Accuracy**, **0.6913 Macro-F1**, and **0.7534 ROC-AUC**. Bigram collocations capture essential multi-word legal doctrines and standard procedural phrases (e.g. *"preponderance of probability"*, *"beyond reasonable doubt"*, *"error apparent on the face of the record"*), yielding a **+3.65% Macro-F1 gain** over unigrams.

---

## 2. Validation Set Results Comparison

The table below presents the verified validation results for all three baseline models on the remediated binary classification task:

| Model Name | Features / Vocab Size | Val Accuracy | Val Macro-F1 | Val Weighted-F1 | Val Precision (Macro) | Val Recall (Macro) | Val ROC-AUC | Val PR-AUC | Training Time | Inference Latency |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **TF-IDF + Logistic Regression** | Word TF-IDF (Unigrams, 12,000) | 65.68% | 0.6548 | 0.6570 | 0.6546 | 0.6551 | 0.7188 | 0.7467 | 56.3s | 0.003 ms/doc |
| **TF-IDF + Linear SVM** | Word TF-IDF (Unigrams, 12,000) | 64.53% | 0.6434 | 0.6456 | 0.6432 | 0.6438 | 0.6971 | 0.7249 | 4.2s | 0.002 ms/doc |
| **Word N-gram TF-IDF + LR** | Word TF-IDF (1-2 N-grams, 16,000) | **69.32%** | **0.6913** | **0.6934** | **0.6911** | **0.6915** | **0.7534** | **0.7777** | 142.3s | 0.003 ms/doc |

*Data source: `results/tables/phase18d_baseline_results.csv`*

---

## 3. Detailed Model Analysis

### Model 1: TF-IDF (Unigram) + Logistic Regression
- **Architecture:** Term Frequency-Inverse Document Frequency vectorizer with sublinear term-frequency scaling ($1 + \log(tf)$), English stop-word filtration, min-document frequency threshold of 3, and max-document frequency cutoff of 0.90. The classifier is an $L_2$-regularized Logistic Regression ($C=1.0$) with class-balanced weighting.
- **Performance:** Balanced precision and recall (Macro Precision: 65.46%, Macro Recall: 65.51%), with an ROC-AUC of 0.7188.
- **Strengths & Weaknesses:** Fast to train and score; effectively captures isolated legal terms but is insensitive to phrase compositionality and negation.

### Model 2: TF-IDF (Unigram) + Linear Support Vector Machine
- **Architecture:** Linear Support Vector Classifier (`LinearSVC`) using squared hinge loss with $C=0.5$ and balanced class weighting, operating on the exact unigram feature space as Model 1.
- **Performance:** Macro-F1 of 0.6434 with an ROC-AUC (decision function rank) of 0.6971.
- **Strengths & Weaknesses:** Extremely rapid convergence (4.16 seconds on 24,849 documents); slightly underperforms logistic regression because margin maximization without probabilistic calibration is more sensitive to high-variance outliers in legal arguments.

### Model 3: Word N-gram (1-2) TF-IDF + Logistic Regression
- **Architecture:** Expanded feature space (16,000 dimensions) incorporating contiguous word bigrams in addition to unigrams, coupled with class-balanced Logistic Regression.
- **Performance:** Best-performing baseline across all metrics (69.32% Accuracy, 0.6913 Macro-F1, 0.7534 ROC-AUC).
- **Substantive Impact:** Bigrams enable the model to identify compound legal standards, statutory headings, and jurisdictional markers that unigrams decompose into ambiguous separate tokens.

---

## 4. Leakage-Free Validation Realism

In un-remediated datasets where final orders and holding sentences are present, simple string-matching models easily achieve artificial accuracy (>90%) by memorizing explicit dispositive keywords (such as *"appeal allowed"* or *"petition dismissed"*).

By enforcing:
1. Complete removal of the final concluding holding sentence (100% masking of `proof_sentence`),
2. Systematic truncation of concluding disposition paragraphs, and
3. Comprehensive regex masking of 14 dispositive structures across the body,

the baseline validation performance of **69.32% Accuracy / 0.6913 Macro-F1** represents genuine statistical predictability derived from factual narratives, lower court proceedings, statutory invocations, and judicial deliberation.

---

## 5. Serialized Checkpoints

All three trained baseline pipelines (including vectorizers, scalers, models, and label encoders) have been serialized to disk:

| Checkpoint File | File Size | Description |
|:---|---:|:---|
| `models/checkpoints/phase18d/binary_tfidf_lr.joblib` | 542,277 bytes | Model 1: Unigram TF-IDF (12k) + Logistic Regression |
| `models/checkpoints/phase18d/binary_tfidf_svm.joblib` | 542,145 bytes | Model 2: Unigram TF-IDF (12k) + Linear SVM |
| `models/checkpoints/phase18d/binary_ngram_lr.joblib` | 775,269 bytes | Model 3: Word N-gram TF-IDF (16k) + Logistic Regression |

---

## 6. Next Steps: Neural Transformer Architectures

With baseline benchmarks established:
- **Floor Benchmark:** 69.32% Accuracy, 0.6913 Macro-F1.
- **Neural Target:** Leverage hierarchical and attention-pooled transformer architectures (`InLegalBERTSequenceClassifier`, `InLegalBERTBiLSTM`, `InLegalBERTAttentionPooling`, `InLegalBERTBiLSTMAttention`) to capture long-range contextual discourse and statutory-factual interactions.
- **Training Protocol:** Detailed in `scripts/phase18d_training.py` with GPU requirements and compute cost projections documented in `results/reports/phase18d_gpu_cost_estimation.md`.
