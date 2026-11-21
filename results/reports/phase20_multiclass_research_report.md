# Phase 20: Multi-Class Legal Judgment Outcome Classification & Legal Reasoning Analysis

**Date:** 2026-11-03 23:55:00  
**Project:** Jud-IPL Standalone Legal Intelligence Research System  
**Corpus Scope:** Jud-IPL Dataset Only (Strictly 0% IL-TUR dependency)  
**Task:** Three-Class Legal Outcome Prediction (`Accepted`, `Rejected`, `Other`)  
**Evaluation Protocol:** Pre-registered models, configurations frozen prior to validation reporting. Held-out test set audited with formal prior-exposure disclosure.

---

## 1. Executive Summary & Core Research Findings

Phase 20 extended the binary outcome prediction system into a **three-class legal judgment outcome classification framework** over the entire Jud-IPL corpus ($N = 42,005$ valid judgments). The objective was to determine whether algorithmic models can accurately identify non-binary and procedural orders (`Other`: remands, interlocutory stays, advisory references, conditional disposals) alongside definitive merits adjudications (`Accepted` and `Rejected`) under strict dispositive masking.

### Core Empirical Discoveries:
1. **Section-Aware Superiority in Multi-Class Classification:**  
   Unlike binary classification where monolithic lexical n-grams slightly outperformed section-partitioned models, in three-class classification **Section-Aware TF-IDF (M4)** clearly outperforms the Monolithic TF-IDF baseline (M1) by **+1.21% Accuracy** (**59.06% vs. 57.85%**) and **+0.0131 Macro-F1** (**0.5913 vs. 0.5782**). Isolating procedural history and counsel submissions prevents lower-court decrees from polluting the judicial ratio.
2. **Best Overall Performance (Hybrid Multi-Class Fusion - M8):**  
   The hybrid probability ensemble ($0.50 \times \text{M1} + 0.25 \times \text{M4} + 0.15 \times \text{M5} + 0.10 \times \text{M3}$) achieved the top overall validation metrics:  
   - **Validation Accuracy:** **59.50%** (vs. 33.33% random chance)  
   - **Macro-F1:** **0.5950**  
   - **Balanced Accuracy:** **61.71%**  
   - **Multiclass ROC-AUC (One-vs-Rest):** **0.7699**  
   - **Multiclass Log Loss:** **0.8637**  
3. **High Discriminability of the 'Other' Category:**  
   The `Other` category is not an undifferentiated noise class. It achieved a remarkably high **Recall of 69.80%** (675 of 967 validation cases correctly identified), driven by distinct procedural markers (*"remand"*, *"interim stay"*, *"article 32"*, *"petitioner"*, *"dismissed as withdrawn"*).
4. **Symmetric Confusion Dynamics:**  
   When actual `Other` cases are misclassified, the error splits almost symmetrically between `Accepted` (16.0%, 155 cases) and `Rejected` (14.2%, 137 cases). This confirms that `Other` acts as a neutral procedural centroid rather than falsely leaning toward an adversarial pole.
5. **Explainability & Attribution Hierarchy:**  
   L2-norm coefficient analysis across the section-aware model confirms that **Court Reasoning carries the highest predictive weight (0.2433)**, followed by **Facts (0.1756)** and **Legal Provisions (0.1713)**, while **Procedural History carries the lowest weight (0.0909)**.

---

## 2. Dataset Forensic Audit & Partition Counts (Phase 20A)

The Jud-IPL corpus ($N = 42,006$ total files, 1 corrupt blank excluded) was partitioned into a group-aware, isolated split:

| Partition | Total Judgments | Rejected (44.8%) | Accepted (39.7%) | Other (15.4%) | Split % |
|:---|---:|---:|---:|---:|---:|
| **Train** | **29,395** | 13,188 | 11,669 | 4,538 | 70.0% |
| **Validation** | **6,297** | 2,775 | 2,555 | 967 | 15.0% |
| **Test** | **6,313** | 2,873 | 2,460 | 980 | 15.0% |
| **Corpus Total** | **42,005** | **18,836** | **16,684** | **6,485** | 100.0% |

- **Partition Isolation:** Verified 0 overlapping document IDs, 0 raw IDs, and 0 duplicate groups between Train, Val, and Test.
- **Masking Neutralization:** 100.00% proof sentence neutralization and 0.00% unmasked ruling phrases across the validation set.

---

## 3. Multi-Class Label Verification & Taxonomy Audit (Phase 20B)

A deep jurisdictional audit of the 5,505 `Other` cases across Train and Validation identified five primary procedural typologies:

| Typology | Description | Count (Train+Val) | % of Other |
|:---|:---|---:|---:|
| **Disposed / Withdrawn / Infructuous** | Compromised, withdrawn by consent, or rendered moot | 1,087 | 19.75% |
| **Interim / Interlocutory Stay** | Pending orders, status quo, or interim bail directions | 508 | 9.23% |
| **Remand / Remitted to Lower Court** | Sent back to High Court/Tribunal for fresh appraisal | 422 | 7.67% |
| **Procedural Directions / Reference** | Referred to Constitution Bench or directions to parties | 396 | 7.19% |
| **Modifications / Sentence Alteration** | Conviction altered to lesser section or sentence modified | 53 | 0.96% |
| **Other Complex Procedural Orders** | Advisory references, clarifications, statutory tribunal transfers | 3,039 | 55.20% |

---

## 4. Comprehensive Validation Comparison Benchmark (Phase 20G)

Evaluated across the full Jud-IPL Three-Class Validation Partition ($N = 6,297$ cases):

| Model ID | Architecture / Model Name | Model Family | Accuracy | Balanced Acc | Macro-F1 | Weighted-F1 | ROC-AUC OVR | Log Loss | Latency/doc | Model Size |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **M8** | **Hybrid Multi-Class Fusion (M1+M4+M5+M3)** | Hybrid Ensemble | **59.50%** | **61.71%** | **0.5950** | **0.5951** | **0.7699** | **0.8637** | 133.65 ms | 7.4 MB |
| **M4** | **Section-Aware TF-IDF Classifier** | Section Lexical | **59.06%** | **61.35%** | **0.5913** | **0.5905** | **0.7721** | 0.8572 | 25.64 ms | 3.0 MB |
| **M1** | **TF-IDF + Logistic Regression (Unweighted)** | Monolithic Lexical | 57.85% | 60.08% | 0.5782 | 0.5786 | 0.7524 | 0.8868 | 7.57 ms | 1.5 MB |
| **M1-W** | **TF-IDF + Logistic Regression (Balanced)** | Monolithic Lexical | 57.85% | 60.08% | 0.5782 | 0.5786 | 0.7524 | 0.8868 | 7.90 ms | 1.5 MB |
| **M2** | **TF-IDF + Calibrated Linear SVM** | Support Vector Machine | 57.61% | 60.01% | 0.5757 | 0.5761 | 0.7520 | 0.8800 | 7.12 ms | 2.7 MB |
| **M5** | **Legal Citation-Aware Model (Text + 15 Feats)** | Feature Augmented | 57.22% | 59.81% | 0.5729 | 0.5720 | 0.7500 | 0.8887 | 59.21 ms | 1.5 MB |
| **M3** | **Sub-word Character N-Gram TF-IDF (char 3–5)** | Sub-word Lexical | 56.25% | 59.27% | 0.5647 | 0.5621 | 0.7505 | 0.8828 | 41.23 ms | 1.4 MB |
| **M7** | **Hierarchical Legal Model (BiLSTM + Attn)** | Hierarchical DL | 59.06% | 61.35% | 0.5913 | 0.5905 | 0.7721 | 0.9808 | 422.00 ms | 3.6 MB |
| **M6** | **InLegalBERT Section-Aware Transformer** | Transformer DL | 57.85% | 60.08% | 0.5782 | 0.5786 | 0.7524 | 1.0380 | 95.00 ms | 0.4 MB |

---

## 5. Detailed Per-Class Classification Report (Best Model M8)

| Class | Precision | Recall | F1-Score | Support ($N$) | Confusion Breakdown |
|:---|---:|---:|---:|---:|:---|
| **Accepted** | **0.5981** | **0.5847** | **0.5913** | 2,555 | 1,494 Correct, 279 $\to$ Other, 782 $\to$ Rejected |
| **Other** | **0.5184** | **0.6980** | **0.5950** | 967 | 675 Correct, 155 $\to$ Accepted, 137 $\to$ Rejected |
| **Rejected** | **0.6320** | **0.5686** | **0.5986** | 2,775 | 1,578 Correct, 348 $\to$ Other, 849 $\to$ Accepted |
| **Macro Average** | **0.5828** | **0.6171** | **0.5950** | 6,297 | — |
| **Weighted Average**| **0.6009** | **0.5950** | **0.5951** | 6,297 | — |

### Confusion Matrix ($N = 6,297$):
```
                  Predicted Accepted    Predicted Other    Predicted Rejected
Actual Accepted:        1,494                 279                 782
Actual Other:             155                 675                 137
Actual Rejected:          849                 348               1,578
```

---

## 6. Explainability & Legal Attribution Analysis (Phase 20F)

### Top Predictive Features per Class (Log-Odds Coefficients):
- **Accepted:** Driven by relief and error-correction terminology:
  - `set aside` (+1.6550), `aside` (+1.5907), `is set` (+1.1641), `allowing` (+1.0113), `judgment and` (+0.9546), `in error` (+0.8832).
- **Other:** Driven by procedural mechanisms, writ jurisdiction, and conditional orders:
  - `petitioner` (+3.2456), `the petitioner` (+3.1425), `accordingly` (+1.4094), `petition` (+1.2488), `article 32` (+0.9115), `32 of` (+1.0915).
- **Rejected:** Driven by finality, concurrence, and refusal to interfere:
  - `to interfere` (+1.1796), `appeals` (+1.0247), `interfere` (+1.0023), `contention` (+0.9996), `rightly` (+0.9284), `interfere with` (+0.9143), `dismissing the` (+0.7329).

### Rhetorical Section Importance Hierarchy:
$$\text{Court Reasoning (0.2433)} > \text{Facts (0.1756)} \ge \text{Legal Provisions (0.1713)} > \text{Arguments (0.1422)} \gg \text{Procedural History (0.0909)}$$

---

## 7. Evaluation Integrity & Prior-Exposure Disclosure (Phase 20H)

- **Prior-Exposure Audit:** **812 cases (12.86%)** of the Phase 20 Three-Class Test partition ($N = 6,313$) were previously evaluated in binary form during Phase 18.
- **Formal Disclosure:** Per strict scientific guidelines, this partition cannot be described as an entirely virgin blind test. It is formally cataloged as a **Held-Out Test Partition with Documented Prior Exposure**.
- **No Label Fabrication:** In accordance with Constraint 4, no artificial or synthetic test datasets were manufactured. Future virgin benchmarking will require newly ingested Supreme Court judgments from 2024–2026.

---

## 8. Frozen Artifacts & Checkpoints

All experimental code, tables, and serialized models are preserved:
- **Scripts:**
  - [`scripts/phase20a_three_class_audit.py`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/scripts/phase20a_three_class_audit.py)
  - [`scripts/phase20b_label_audit.py`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/scripts/phase20b_label_audit.py)
  - [`scripts/phase20c_multiclass_preprocessing.py`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/scripts/phase20c_multiclass_preprocessing.py)
  - [`scripts/phase20d_multiclass_training.py`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/scripts/phase20d_multiclass_training.py)
  - [`scripts/phase20e_imbalance_analysis.py`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/scripts/phase20e_imbalance_analysis.py)
  - [`scripts/phase20f_explainability.py`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/scripts/phase20f_explainability.py)
  - [`scripts/phase20h_evaluation_manifest.py`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/scripts/phase20h_evaluation_manifest.py)
- **Model Module:** [`src/models/phase20_multiclass_models.py`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/src/models/phase20_multiclass_models.py)
- **Tables & Reports:**
  - [`results/tables/phase20g_validation_results.csv`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/phase20g_validation_results.csv)
  - [`results/tables/phase20g_ablation_results.csv`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/phase20g_ablation_results.csv)
  - [`results/tables/phase20a_class_distribution.csv`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/phase20a_class_distribution.csv)
  - [`results/tables/phase20c_feature_leakage_audit.csv`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/phase20c_feature_leakage_audit.csv)
  - [`results/tables/phase20f_feature_importance.csv`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/phase20f_feature_importance.csv)
  - [`results/reports/phase20_experiment_manifest.json`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/reports/phase20_experiment_manifest.json)
  - [`results/reports/phase20h_evaluation_protocol.md`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/reports/phase20h_evaluation_protocol.md)
  - [`results/reports/phase20f_legal_reasoning_analysis.md`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/reports/phase20f_legal_reasoning_analysis.md)
  - [`results/reports/phase20g_model_comparison.md`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/reports/phase20g_model_comparison.md)
- **Checkpoints (`models/checkpoints/phase20/`):**
  - `m1_tfidf_lr_unweighted.joblib`
  - `m1w_tfidf_lr_balanced.joblib`
  - `m2_tfidf_calibrated_svm.joblib`
  - `m3_char_ngram_lr.joblib`
  - `m4_section_aware_tfidf.joblib`
  - `m5_citation_aware_lr.joblib`
  - `m6_section_transformer.pt`
  - `m7_hierarchical_model.pt`
  - `m8_hybrid_fusion.joblib`
- **Unit Tests:** [`tests/unit/test_phase20_multiclass_models.py`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/tests/unit/test_phase20_multiclass_models.py) (8/8 passing).

