# Phase 18D-C: Calibration Audit, Model Freeze & Final Held-Out Test Evaluation

**Generated:** 2026-11-03 21:20:00  
**Status:** Completed, Audited, Reconciled & Evaluated on Held-Out Test Data  
**Scope:** Jud-IPL Standalone Legal Intelligence Research System (Strictly 0% IL-TUR dependency)  
**Evaluation Protocol:** Pre-registered models, ensemble weights, and threshold frozen prior to test evaluation. No test labels were used for training, calibration, tuning, or threshold selection.

---

## 1. Executive Summary & Critical Reconciliation

Phase 18D-C conducted a forensic audit of calibration claims, audited the validation-derived ensemble configuration, resolved model nomenclature, and performed final held-out test evaluations across both the full held-out test partition ($N=5,351$) and the balanced benchmark subset ($N=1,000$).

### Core Reconciliations & Scientific Corrections:
1. **Model Nomenclature & No Hybrid Claim (Critical Issue 1):**  
   The frozen ensemble grid search over validation data assigned **transformer weight $\alpha = 0.00$** and **baseline weight $1.00$** because the standalone Word 1-2 N-Gram TF-IDF Logistic Regression achieved higher Macro-F1 (0.7063 at $\tau=0.49$) than any linear blend with transformer representations. Therefore, the proposed model is **not a hybrid model**; it is officially designated as:  
   **"TF-IDF Logistic Regression with Validation-Selected Decision Threshold" ($\tau = 0.49$)**.  
   No transformer representation contributes to this model's predictions, and no hybrid superiority is claimed.
2. **Full Test Partition Evaluation ($N=5,351$, Critical Issue 2):**  
   On the complete held-out test partition of 5,351 unseen judgments:
   - **Baseline ($\tau = 0.50$):** **69.56% Accuracy**, **0.6949 Macro-F1**, **0.6950 Balanced Accuracy**, **0.7619 ROC-AUC**, **0.7259 PR-AUC**.
   - **Proposed ($\tau = 0.49$):** **69.80% Accuracy**, **0.6978 Macro-F1**, **0.6986 Balanced Accuracy**, **0.7619 ROC-AUC**, **0.7259 PR-AUC**.
   - **Calibrated Baseline (OOF Platt, $\tau = 0.50$):** **69.73% Accuracy**, **0.6953 Macro-F1**, **0.6950 Balanced Accuracy**, **0.7619 ROC-AUC**, **0.2004 Brier Score**.
3. **Balanced Benchmark Separation ($N=1,000$, Critical Issue 3):**  
   Evaluated on the exact same 1,000 document IDs (500 Accepted, 500 Rejected):
   - **Baseline ($\tau = 0.50$):** **70.80% Accuracy**, **0.7080 Macro-F1**, **0.7602 ROC-AUC**.
   - **Proposed ($\tau = 0.49$):** **71.30% Accuracy**, **0.7129 Macro-F1**, **0.7602 ROC-AUC**.
   - **Calibrated Baseline (OOF Platt, $\tau = 0.50$):** **70.40% Accuracy**, **0.7040 Macro-F1**, **0.7602 ROC-AUC**, **0.2004 Brier Score**.
   - **InLegalBERT Attention Pooling:** **52.70% Accuracy**, **0.5269 Macro-F1**, **0.5505 ROC-AUC**.
   - **InLegalBERT BiLSTM Attention:** **51.60% Accuracy**, **0.5140 Macro-F1**, **0.5312 ROC-AUC**.
4. **Calibration Impact on Zero-Transformer Weight (Critical Issue 4):**  
   Because $\alpha = 0.00$, Platt scaling fits solely on the baseline log-odds:
   $$P_{\text{cal}} = \frac{1}{1 + \exp(-(1.3511 \cdot z - 0.0921))}$$
   This transformation is strictly monotonic, preserving identical ROC-AUC and PR-AUC. It improves probability calibration (Brier score decreases from 0.2026 to 0.2004). At a fixed decision threshold of 0.50, it implicitly shifts the raw decision threshold from 0.50 to approximately 0.517, resulting in a conservative shift toward predicting Rejected.
5. **Statistical Significance Testing (Critical Issue 5):**  
   Paired bootstrap confidence intervals (1,000 iterations, seed=42) confirm:
   - **Full Test Set ($N=5,351$):** $\Delta \text{Acc} = +0.24\%$ [95% CI: $-0.13\%, +0.67\%$]; $\Delta \text{Macro-F1} = +0.0029$ [95% CI: $-0.0009, +0.0072$]. The interval spans zero; the $+0.24\%$ gain is within sampling variation.
   - **Balanced Benchmark ($N=1,000$):** $\Delta \text{Acc} = +0.50\%$ [95% CI: $-0.30\%, +1.40\%$]; $\Delta \text{Macro-F1} = +0.0050$ [95% CI: $-0.0029, +0.0139$]. The interval also spans zero.

---

## 2. Forensic Audit of Previous Calibration Claims

In Phase 18D-B, a 70.83% accuracy had been reported using Platt scaling fit directly on the validation dataset. The forensic audit confirmed:

| Audit Item | Verification | Finding | Impact & Action Taken |
|:---|:---|:---|:---|
| **Validation Overfitting** | In-Sample Logit Fitting | Confirmed | Platt calibrator was previously fit directly on validation labels, creating optimism. **Retracted.** |
| **Out-of-Fold Calibration** | 5-Fold Stratified Group K-Fold | Verified | Calibrator re-fit strictly on OOF training predictions with zero validation or test exposure. |
| **Ranking Invariance** | Monotonicity Check | Verified | OOF Platt calibrator produces strictly monotonic probability transforms; ROC-AUC is invariant ($|\Delta\text{AUC}| = 0.000000$). |
| **Split Isolation** | Document & Duplicate Group Hash Check | Verified | 0 overlapping IDs, 0 overlapping raw IDs, and 0 overlapping duplicate groups across Train, Val, and Test. |

---

## 3. Comprehensive Held-Out Test Evaluation Table

All models and evaluations are recorded in [`results/tables/phase18dc_test_results.csv`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/phase18dc_test_results.csv).

| Population | Configuration | Model Type | Threshold ($\tau$) | Sample Size ($N$) | Accuracy | Balanced Acc | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC | Brier Score | Latency/doc | Model Size |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full Test Set** | **TF-IDF + LR Baseline** | Linear Baseline | 0.50 | 5,351 | 69.56% | 69.50% | 0.6949 | 0.6957 | 0.7619 | 0.7259 | 0.2026 | 3.85 ms | 0.77 MB |
| **Full Test Set** | **TF-IDF + LR with Val Threshold** | Proposed Baseline | 0.49 | 5,351 | 69.80% | 69.86% | 0.6978 | 0.6982 | 0.7619 | 0.7259 | 0.2026 | 3.85 ms | 0.77 MB |
| **Full Test Set** | **TF-IDF + LR Calibrated (OOF Platt)** | Calibrated Baseline | 0.50 | 5,351 | 69.73% | 69.50% | 0.6953 | 0.6966 | 0.7619 | 0.7259 | 0.2004 | 3.85 ms | 0.77 MB |
| **Balanced Benchmark** | **TF-IDF + LR Baseline** | Linear Baseline | 0.50 | 1,000 | 70.80% | 70.80% | 0.7080 | 0.7080 | 0.7602 | 0.7349 | 0.2026 | 3.98 ms | 0.77 MB |
| **Balanced Benchmark** | **TF-IDF + LR with Val Threshold** | Proposed Baseline | 0.49 | 1,000 | 71.30% | 71.30% | 0.7129 | 0.7129 | 0.7602 | 0.7349 | 0.2026 | 3.98 ms | 0.77 MB |
| **Balanced Benchmark** | **TF-IDF + LR Calibrated (OOF Platt)** | Calibrated Baseline | 0.50 | 1,000 | 70.40% | 0.7040 | 0.7040 | 0.7040 | 0.7602 | 0.7349 | 0.2004 | 3.98 ms | 0.77 MB |
| **Balanced Benchmark** | **InLegalBERT Attention Pooling** | Transformer | 0.50 | 1,000 | 52.70% | 52.70% | 0.5269 | 0.5269 | 0.5505 | 0.5324 | 0.2486 | 330.90 ms | 440.38 MB |
| **Balanced Benchmark** | **InLegalBERT BiLSTM Attention** | Transformer | 0.50 | 1,000 | 51.60% | 51.60% | 0.5140 | 0.5140 | 0.5312 | 0.5201 | 0.2514 | 352.10 ms | 453.78 MB |

---

## 4. Confusion Matrices & Classification Reports

### A. Full Test Set ($N=5,351$) Confusion Matrices:

1. **TF-IDF + LR Baseline ($\tau = 0.50$):**
   ```
                     Predicted Rejected (0)    Predicted Accepted (1)
   Actual Rejected:          1,990                      831
   Actual Accepted:            798                    1,732
   ```

2. **Proposed: TF-IDF + LR with Validation Threshold ($\tau = 0.49$):**
   ```
                     Predicted Rejected (0)    Predicted Accepted (1)
   Actual Rejected:          1,941                      880
   Actual Accepted:            736                    1,794
   ```
   *Net change at $\tau = 0.49$: 62 additional true Accepted cases captured at the cost of 49 additional false positives.*

3. **Calibrated Baseline ($\tau = 0.50$, effective threshold $\approx 0.517$):**
   ```
                     Predicted Rejected (0)    Predicted Accepted (1)
   Actual Rejected:          2,078                      743
   Actual Accepted:            877                    1,653
   ```

### B. Balanced Benchmark ($N=1,000$, 500 Accepted / 500 Rejected):

1. **Proposed Model ($\tau = 0.49$):**
   ```
                     Predicted Rejected (0)    Predicted Accepted (1)
   Actual Rejected:            348                      152
   Actual Accepted:            135                      365
   ```

2. **InLegalBERT Attention Pooling ($\tau = 0.50$):**
   ```
                     Predicted Rejected (0)    Predicted Accepted (1)
   Actual Rejected:            258                      242
   Actual Accepted:            231                      269
   ```

3. **InLegalBERT BiLSTM Attention ($\tau = 0.50$):**
   ```
                     Predicted Rejected (0)    Predicted Accepted (1)
   Actual Rejected:            264                      236
   Actual Accepted:            248                      252
   ```

---

## 5. Paired Bootstrap Hypothesis Testing

To assess whether the proposed validation threshold ($\tau = 0.49$) provides a statistically significant improvement over default ($\tau = 0.50$), 1,000 paired bootstrap iterations were run with random seed 42:

### Full Held-Out Test Set ($N=5,351$):
- **$\Delta$ Accuracy:** $+0.0024$ ($+0.24\%$)
  - **95% Confidence Interval:** $[-0.0013, +0.0067]$ ($-0.13\%$ to $+0.67\%$)
- **$\Delta$ Macro-F1:** $+0.0029$
  - **95% Confidence Interval:** $[-0.0009, +0.0072]$
- **Conclusion:** Both intervals cross zero. The observed gain is within sampling variation on the full test set.

### Balanced Benchmark Subset ($N=1,000$):
- **$\Delta$ Accuracy:** $+0.0050$ ($+0.50\%$)
  - **95% Confidence Interval:** $[-0.0030, +0.0140]$ ($-0.30\%$ to $+1.40\%$)
- **$\Delta$ Macro-F1:** $+0.0050$
  - **95% Confidence Interval:** $[-0.0029, +0.0139]$
- **Conclusion:** Both intervals cross zero. The benchmark improvement should not be overstated as statistically significant.

---

## 6. Granular Error Analysis

Evaluated on the standardized 1,000-case test sample across document and legal dimensions:

| Dimension | Stratum | Cases | Accuracy | Error Rate | Primary Observation |
|:---|:---|---:|---:|---:|:---|
| **Class Label** | **Accepted** | 500 | 73.00% | 27.00% | Lower false-negative rate with threshold 0.49. |
| **Class Label** | **Rejected** | 500 | 69.60% | 30.40% | Moderate false-positive rate due to slight threshold relaxation. |
| **Document Length** | **Q1 (Shortest 25%)** | 250 | 74.80% | 25.20% | Concise factual records yield strongest lexical signals. |
| **Document Length** | **Q2 (Medium-Short)** | 250 | 73.20% | 26.80% | Stable performance across intermediate judgments. |
| **Document Length** | **Q3 (Medium-Long)** | 250 | 71.60% | 28.40% | Slight degradation as procedural background grows. |
| **Document Length** | **Q4 (Longest 25%)** | 250 | 65.60% | 34.40% | Multiple procedural stages and dense citations obscure core ratio. |
| **Character Truncation** | **Truncated ($\ge$32k chars)** | 171 | 63.74% | 36.26% | Long cases suffer primarily from narrative complexity, not truncation. |
| **Character Truncation** | **Untruncated (<32k chars)** | 829 | 72.86% | 27.14% | Complete texts yield consistent lexical representations. |
| **Case Type** | **Civil** | 666 | 70.27% | 29.73% | Civil appeals exhibit balanced accuracy. |
| **Case Type** | **Criminal** | 179 | 71.51% | 28.49% | Criminal judgments show strong evidentiary lexical indicators. |
| **Case Type** | **Unspecified / Other** | 155 | 75.48% | 24.52% | High accuracy in short procedural orders. |

---

## 7. Frozen Artifacts & Checkpoints

All experimental artifacts are strictly tracked and preserved:

- **Manifest:** [`results/reports/phase18dc_experiment_manifest.json`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/reports/phase18dc_experiment_manifest.json)
- **Test Table:** [`results/tables/phase18dc_test_results.csv`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/phase18dc_test_results.csv)
- **Validation Table:** [`results/tables/phase18dc_validation_results.csv`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/phase18dc_validation_results.csv)
- **Classification Report:** [`results/tables/phase18dc_classification_report.csv`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/phase18dc_classification_report.csv)
- **Error Analysis Table:** [`results/tables/phase18dc_error_analysis.csv`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/phase18dc_error_analysis.csv)
- **Baseline Model Checkpoint:** [`models/checkpoints/phase18d/binary_ngram_lr.joblib`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/models/checkpoints/phase18d/binary_ngram_lr.joblib) (SHA-256: `51df9b82...`)
- **OOF Platt Calibrator Checkpoint:** [`models/checkpoints/phase18d/platt_calibrator.joblib`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/models/checkpoints/phase18d/platt_calibrator.joblib) (SHA-256: `24b2f7ec...`)
- **Transformer Checkpoint:** [`models/checkpoints/phase18d/inlegalbert_attention_pooling_best.pt`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/models/checkpoints/phase18d/inlegalbert_attention_pooling_best.pt) (SHA-256: `c6bb14a2...`)
