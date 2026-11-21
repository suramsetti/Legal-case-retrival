# Phase 21B: Final Three-Class Test Evaluation & Comparative Performance Analysis

**Project:** Jud-IPL Standalone Legal Intelligence Research System  
**Audit Protocol:** Frozen-Model Comparative Test Evaluation  
**Evaluated Models:**  
- **Model M8:** Hybrid Multi-Class Fusion Ensemble (`m8_hybrid_fusion.joblib`)  
- **Model M4:** Section-Aware TF-IDF Classifier (`m4_section_aware_tfidf.joblib`)  
**Evaluation Population:** $N = 6,311$ eligible held-out judgments (from $6,313$ total test partition)  
**Scientific Integrity Constraints:** 0% IL-TUR Dependency | Zero Model Retraining | Zero Weight Tuning | Zero Threshold Optimization | Explicit Prior Exposure Disclosure  

---

## 1. Executive Summary & Comparative Matrix

This evaluation establishes the definitive held-out test performance of the two leading models from Phase 20/21A under strict frozen-model conditions:
- **Model M8 (Hybrid Late Fusion):** Combines M1 (50%), M4 (25%), M5 (15%), and M3 (10%).
- **Model M4 (Section-Aware TF-IDF):** 5-section rhetorical decomposition (50,000 features).

Both models were evaluated on the identical eligible held-out test partition ($N = 6,311$), excluding the two cross-partition duplicate cases identified in Phase 21A.

### Final Held-Out Test Evaluation Matrix ($N = 6,311$):

| Evaluation Metric | Model M4 (Section-Aware) | Model M8 (Hybrid Ensemble) | Difference ($\Delta = M8 - M4$) | Statistical Significance |
|---|---|---|---|---|
| **Accuracy** | **58.66%** | **59.20%** | **+0.54%** | Not Significant ($p = 0.2670$, McNemar) |
| **Balanced Accuracy** | **61.03%** | **61.26%** | **+0.23%** | Not Significant (95% CI: [-0.70%, +1.08%]) |
| **Macro-F1** | **0.5898** | **0.5919** | **+0.0021** | Not Significant (95% CI: [-0.0068, +0.0106]) |
| **Weighted-F1** | **0.5864** | **0.5922** | **+0.0058** | Descriptive (+0.58%) |
| **Macro Precision** | 0.5780 | 0.5805 | +0.0025 | Descriptive |
| **Macro Recall** | 0.6103 | 0.6126 | +0.0023 | Descriptive |
| **Multiclass ROC-AUC (OvR)**| **0.7685** | **0.7651** | **-0.0034** | M4 superior |
| **Macro PR-AUC** | 0.6500 | 0.6526 | +0.0026 | Descriptive |
| **Multiclass Log Loss** | **0.8630** | **0.8689** | **+0.0059** | M4 superior (lower loss) |
| **Expected Calibration Error**| **0.0278** | **0.0540** | **+0.0262** | M4 superior (better calibrated) |
| **Inference Latency** | **15.03 ms/doc** | **>130 ms/doc** | **~8.7x slower** | M4 superior (single model) |
| **Model Footprint** | **3.01 MB** | **7.45 MB** (4 models) | **+4.44 MB** | M4 superior |

---

## 2. Pre-Evaluation Integrity Audit (Phase 21B-1)

All checkpoints and serialization manifests were forensically audited and verified prior to test prediction generation:
- **M4 Checkpoint (`m4_section_aware_tfidf.joblib`):** SHA-256 `1d5963eec3d4002112e61fe79f73f96391738ed2bc8e56dc97ed8743ac13b460` (Confirmed frozen).
- **M8 Checkpoint (`m8_hybrid_fusion.joblib`):** SHA-256 `552e0819080d560c1bf82b5a2eeba8a2e805fe2fed88907ca004eed16f2b4db2` (Confirmed frozen).
- **M8 Component Checkpoints:**
  - M1: `26c62266ad9a2d59...` (1.52 MB)
  - M3: `1ce55ad05ceddde1...` (1.39 MB)
  - M5: `53b592c95352fea1...` (1.52 MB)
- **M8 Late Fusion Weights:** $\{M1: 0.50, M4: 0.25, M5: 0.15, M3: 0.10\}$ summing to $1.000000$.
- **Class Ordering:** Strictly standardized to `['Accepted', 'Other', 'Rejected']`.

---

## 3. Cross-Split Duplicate Resolution (Phase 21B-2)

In accordance with Phase 21A findings, exact MD5 text hashes across partition boundaries were audited. Exactly three cross-partition collisions were detected:
1. **Train–Validation:** 1 pair (Train `0054290303` vs Val `0000289593`, hash `e39219f573fd665ad5fb2af1b111bce9`).
2. **Train–Test:** 1 pair (Train `0001664346` vs Test `0001116667`, hash `55e2943267035597e0f5f2c26a528585`).
3. **Validation–Test:** 1 pair (Val `0103682128` vs Test `0001505760`, hash `c264cff91bf065c02a55e5cda0e32028`).

### Deterministic Exclusion Manifest:
- To eliminate any cross-partition text leakage without altering split files, the two test documents involved in collisions were excluded from the evaluation set:
  - Excluded Test IDs: `['0001116667', '0001505760']`
- **Original Test Size:** $6,313$
- **Excluded Cases:** $2$
- **Eligible Test Size:** **$6,311$**
- Manifest persisted at `results/reports/phase21b_duplicate_exclusion_manifest.json` and `results/tables/phase21b_duplicate_exclusion_audit.csv`.

---

## 4. Frozen Model Predictions (Phase 21B-3)

Predictions were generated strictly on masked text without inspecting or loading test labels:
- `m4_test_predictions.csv`: 6,311 rows.
- `m8_test_predictions.csv`: 6,311 rows.
- **Verification:** Both files have identical document IDs in identical sequence. Zero NaNs, zero infs. Probabilities sum to $1.000000 \pm 10^{-5}$ across all rows.

---

## 5. Granular Class-Wise Performance (Phase 21B-4)

### Class Distribution in Eligible Test Partition ($N = 6,311$):
- **Rejected:** 2,872 cases (45.51%)
- **Accepted:** 2,459 cases (38.96%)
- **Other:** 980 cases (15.53%)

### Model M4 Per-Class Performance:
- **Accepted:** Precision = 0.5657 | Recall = 0.5763 | **F1 = 0.5709** (Support: 2,459, FP: 1,088, FN: 1,042)
- **Other:** Precision = 0.5353 | Recall = 0.6969 | **F1 = 0.6055** (Support: 980, FP: 593, FN: 297)
- **Rejected:** Precision = 0.6332 | Recall = 0.5578 | **F1 = 0.5931** (Support: 2,872, FP: 928, FN: 1,270)
- **Confusion Matrix (M4):**
  $$\begin{bmatrix} 1417 & 246 & 796 \\ 165 & 683 & 132 \\ 923 & 347 & 1602 \end{bmatrix}$$

### Model M8 Per-Class Performance:
- **Accepted:** Precision = 0.5774 | Recall = 0.5872 | **F1 = 0.5823** (Support: 2,459, FP: 1,057, FN: 1,015)
- **Other:** Precision = 0.5217 | Recall = 0.6867 | **F1 = 0.5930** (Support: 980, FP: 617, FN: 307)
- **Rejected:** Precision = 0.6425 | Recall = 0.5637 | **F1 = 0.6005** (Support: 2,872, FP: 901, FN: 1,253)
- **Confusion Matrix (M8):**
  $$\begin{bmatrix} 1444 & 259 & 756 \\ 162 & 673 & 145 \\ 895 & 358 & 1619 \end{bmatrix}$$

---

## 6. Comparative Statistical Analysis (Phase 21B-5)

1. **Paired Bootstrap Confidence Intervals ($B = 1,000$, seed = 42):**
   - $\Delta \text{Macro-F1} = +0.0021$, 95% CI: **$[-0.0068, +0.0106]$**, $p = 0.3190$.
   - $\Delta \text{Accuracy} = +0.54\%$, 95% CI: **$[-0.40\%, +1.43\%]$**, $p = 0.2670$.
   - $\Delta \text{Balanced Accuracy} = +0.23\%$, 95% CI: **$[-0.70\%, +1.08\%]$**, $p = 0.2950$.
   - **Conclusion:** Because zero is centrally contained within all 95% bootstrap confidence intervals, the performance difference between M8 and M4 is **statistically indistinguishable from random noise**.
2. **McNemar's Paired Error Test:**
   - Both correct ($n_{11}$): 3,366 (53.34%)
   - Both incorrect ($n_{00}$): 2,241 (35.51%)
   - M4 incorrect, M8 correct ($n_{01}$): 370 (5.86%)
   - M4 correct, M8 incorrect ($n_{10}$): 334 (5.29%)
   - Net difference: $+36$ cases in favor of M8 out of 6,311 cases.
   - Continuity-corrected $\chi^2 = 1.7401$ ($df = 1$).
   - $p\text{-value} = \mathbf{0.267038} > 0.05$.
   - **Conclusion:** The null hypothesis of equal error rates cannot be rejected.

---

## 7. Prior Exposure Disclosure & Performance Breakdown (Phase 21B-6)

### Transparent Scientific Disclosure:
- The three-class test partition contains **812 cases (12.87%)** that overlap with the held-out test partition previously evaluated in Phase 18 binary classification experiments.
- Neither of the two cross-split duplicate cases was part of this overlap.
- Therefore, in the final eligible population of $6,311$ judgments:
  - **Exposed Cases:** 812 (12.87%)
  - **Unexposed (Virgin) Cases:** 5,499 (87.13%)
- This evaluation is explicitly documented as **Held-Out Test Evaluation with Disclosed Historical Binary Overlap**, not an unexposed blind test.

### Performance by Exposure Category:
- **Unexposed Cases ($N = 5,499$):**
  - M4: Accuracy = **59.07%** | Macro-F1 = **0.5974**
  - M8: Accuracy = **59.63%** | Macro-F1 = **0.5999**
- **Exposed Cases ($N = 812$):**
  - M4: Accuracy = **55.91%** | Macro-F1 = **0.3966** (No "Other" cases present in binary partition)
  - M8: Accuracy = **56.28%** | Macro-F1 = **0.3991**
- **Observation:** Test accuracy on unexposed judgments is slightly *higher* than on previously exposed cases, confirming that prior Phase 18 exposure did not inflate the evaluation metrics.

---

## 8. Robustness & Subgroup Analysis (Phase 21B-7)

1. **Document Length:**
   - **Short ($\le 1,500$ words, $N = 2,180$):** M4 = 63.12% | M8 = **64.72%** ($\Delta = +1.61\%$)
   - **Medium ($1,501-4,500$ words, $N = 2,642$):** M4 = **59.05%** | M8 = 58.89% ($\Delta = -0.15\%$)
   - **Long ($> 4,500$ words, $N = 1,489$):** M4 = 51.44% | M8 = **51.65%** ($\Delta = +0.20\%$)
2. **Reasoning Section Completeness:**
   - **Reasoning Present ($N = 5,727$):** M4 = 58.29% | M8 = **58.70%**
   - **Reasoning Missing / Brief ($N = 584$):** M4 = 62.33% | M8 = **64.04%**

---

## 9. Calibration & Probability Analysis (Phase 21B-8)

| Metric | Model M4 (Section TF-IDF) | Model M8 (Hybrid Ensemble) | Advantage |
|---|---|---|---|
| **Brier Score** | **0.5209** | 0.5219 | M4 |
| **Multiclass Log Loss** | **0.8630** | 0.8689 | M4 |
| **Expected Calibration Error (ECE)** | **0.0278** | 0.0540 | M4 (Better Calibrated) |
| **Mean Confidence** | 0.6144 | 0.5380 | M4 |
| **High-Confidence Cases ($> 0.70$)** | 1,755 | 665 | M4 |
| **High-Confidence Error Rate** | 23.53% | **16.39%** | M8 (Safer when over-confident) |

- **Key Takeaway:** Model M4 exhibits superior probabilistic sharpness and lower Expected Calibration Error (0.0278 vs 0.0540). M8 flattens probability distributions toward the mean, which reduces extreme misclassifications (16.39% vs 23.53% high-confidence errors) but slightly increases overall log loss.

---

## 10. Final Model Comparison & Scientific Recommendation

### Comparative Trade-off Matrix:

| Dimension | Model M4 (Section-Aware TF-IDF) | Model M8 (Hybrid Fusion Ensemble) | Evaluator Assessment |
|---|---|---|---|
| **Predictive Power** | 58.66% Acc, 0.5898 Macro-F1 | 59.20% Acc, 0.5919 Macro-F1 | Comparable (Difference not significant, $p=0.267$) |
| **Probabilistic Quality** | Log Loss: 0.8630, ECE: 0.0278 | Log Loss: 0.8689, ECE: 0.0540 | **M4 is superior** |
| **Inference Latency** | **15.03 ms / doc** | **>130 ms / doc** | **M4 is ~8.7x faster** |
| **System Complexity** | **Single Model** (50k features) | **Ensemble of 4 Models** | **M4 is substantially simpler** |
| **Storage Footprint** | **3.01 MB** | **7.45 MB** | **M4 is smaller** |

### Final Engineering & Research Recommendation:
**MODEL M4 (Section-Aware TF-IDF Classifier) IS RECOMMENDED AS THE PRIMARY PRODUCTION SYSTEM.**

**Rationale:**
1. **Statistical Parity:** McNemar's paired test ($p = 0.2670$) and bootstrap confidence intervals confirm that M8's $+0.54\%$ accuracy margin is statistically indistinguishable from random variation.
2. **Operational Efficiency:** M4 requires only a single classifier invocation, running in **15.03 ms/doc** compared to $>130\text{ ms/doc}$ for M8 (which must execute character n-grams, text n-grams, section segmentations, and dense feature extractions).
3. **Probabilistic Integrity:** M4 provides sharper, better-calibrated probabilities ($\text{ECE} = 0.0278$) and lower cross-entropy log loss ($0.8630$).

---

## 11. Unresolved Research Limitations & Future Roadmap

1. **Proxy Transformer Limitation:** End-to-end InLegalBERT fine-tuning and hierarchical BiLSTM models could not be trained due to GPU resource constraints. When GPU infrastructure is provisioned, transformer fine-tuning should be revisited.
2. **Balanced vs Natural Class Weighting:** M1-W was trained on a pre-balanced subsample ($N=6,000$), nullifying class weighting. In future work, training on the natural imbalanced corpus ($N=29,395$) will allow proper testing of class weighting for minority "Other" decisions.
3. **Cross-Partition Near-Duplicates:** The 3 identified cross-partition duplicate pairs should be globally purged from the master Jud-IPL corpus.
