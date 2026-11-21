# Phase 20G: Multi-Class Model Comparison & Decision Dynamics Report

**Evaluation Partition:** Jud-IPL Three-Class Validation Partition ($N = 6,297$)  
**Classes:** `Accepted` (2,555), `Rejected` (2,775), `Other` (967)  
**Random Baseline:** 33.33% Accuracy  

---

## 1. Executive Summary

Across 8 distinct architectures, **Model M8 (Hybrid Multi-Class Fusion)** achieved the top overall performance:
- **Validation Accuracy:** **59.50%**
- **Macro-F1:** **0.5950**
- **Balanced Accuracy:** **61.71%**
- **Multiclass ROC-AUC (One-vs-Rest):** **0.7699**

**Section-Aware Superiority:** Unlike the binary task where monolithic n-grams edged out section models, in the three-class task **Section-Aware TF-IDF (M4)** clearly outperforms Monolithic TF-IDF (M1) by **+1.21% Accuracy** (59.06% vs 57.85%) and **+0.0131 Macro-F1** (0.5913 vs 0.5782), because isolating procedural history from court reasoning helps cleanly distinguish remands and interlocutory orders from final dispositions.

---

## 2. Decision Dynamics & Confusion Analysis for 'Other'

Out of **967 actual 'Other' judgments** in the validation partition:
- **Correctly Identified as Other:** **675 cases (69.80% Recall)**
- **Misclassified as Accepted:** 155 cases (16.03%)
- **Misclassified as Rejected:** 137 cases (14.17%)

### Symmetry in Misclassification:
The error rate for `Other` is almost perfectly balanced between `Accepted` (16.03%) and `Rejected` (14.17%). This confirms that 'Other' acts as a neutral procedural centroid rather than gravitating toward one adversarial pole.

### False Other Predictions:
- **Accepted predicted as Other:** 279 cases (10.9% of Accepted)
- **Rejected predicted as Other:** 348 cases (12.5% of Rejected)

---

## 3. Comprehensive Model Ablation Table

| Ablation ID | Model Configuration | Val Accuracy | Balanced Acc | Macro-F1 | F1 (Accepted) | F1 (Rejected) | F1 (Other) | ROC-AUC | Delta Macro-F1 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ABL-0 (Best Fusion)** | Hybrid Multi-Class Fusion (M8) | 59.50% | 61.71% | 0.5950 | 0.5913 | 0.5986 | 0.5950 | 0.7699 | **+0.0000** |
| **ABL-1** | Section-Aware TF-IDF alone (M4) | 59.06% | 61.35% | 0.5913 | 0.5836 | 0.5955 | 0.5947 | 0.7721 | **-0.0037** |
| **ABL-2** | Monolithic TF-IDF alone (M1) | 57.85% | 60.08% | 0.5782 | 0.5765 | 0.5812 | 0.5768 | 0.7524 | **-0.0168** |
| **ABL-3** | Calibrated Linear SVM (M2) | 57.61% | 60.01% | 0.5757 | 0.5805 | 0.5729 | 0.5738 | 0.7520 | **-0.0193** |
| **ABL-4** | Sub-word Char N-Grams alone (M3) | 56.25% | 59.27% | 0.5647 | 0.5553 | 0.5639 | 0.5748 | 0.7505 | **-0.0303** |
| **ABL-5** | Citation-Aware Model alone (M5) | 57.22% | 59.81% | 0.5729 | 0.5702 | 0.5724 | 0.5761 | 0.7500 | **-0.0221** |
