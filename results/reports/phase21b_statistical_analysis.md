# Phase 21B-5: Comparative Statistical Analysis Report (M8 vs M4)

**Evaluation Date:** 2026-11-04 11:06:34
**Project:** Jud-IPL Standalone Legal Intelligence Research System
**Evaluation Corpus:** Held-out Three-Class Test Partition ($N = 6,311$ eligible cases)

---

## 1. Executive Summary

A paired statistical comparison was conducted between **Model M8 (Hybrid Multi-Class Fusion Ensemble)** and **Model M4 (Section-Aware TF-IDF Classifier)** across identical eligible test documents.

### Key Comparative Metrics:

| Metric | M4 (Section-Aware) | M8 (Hybrid Fusion) | Difference ($\Delta$) | 95% Bootstrap CI | Statistical Significance ($p < 0.05$) |
|---|---|---|---|---|---|
| **Accuracy** | 58.66% | 59.20% | **+0.54%** | [-0.40%, +1.43%] | Not Significant ($p = 0.2670$) |
| **Macro-F1** | 0.5898 | 0.5919 | **+0.0021** | [-0.0068, +0.0106] | Inconclusive ($p = 0.3190$) |
| **Balanced Acc** | 61.03% | 61.26% | **+0.22%** | [-0.70%, +1.08%] | Inconclusive |
| **Weighted-F1** | 0.5864 | 0.5922 | **+0.0058** | N/A | N/A |

---

## 2. McNemar's Paired Error Analysis

Contingency table of paired predictions across all 6,311 eligible documents:

| Contingency Cell | Definition | Count |
|---|---|---|
| **$n_{11}$** | Both models correct | 3,277 (51.93%) |
| **$n_{00}$** | Both models incorrect | 2,150 (34.07%) |
| **$n_{01}$** | M4 incorrect, M8 correct | **459** (7.27%) |
| **$n_{10}$** | M4 correct, M8 incorrect | **425** (6.73%) |

- **Net Error Reduction by M8:** $n_{01} - n_{10} = +34$ judgments.
- **McNemar Chi-Square Statistic ($\chi^2$):** 1.2319 ($df = 1$)
- **P-Value:** 0.267038

---

## 3. Class-Wise F1 Comparison

| Class | M4 F1-Score | M8 F1-Score | F1 Delta ($\Delta$) | Support ($N$) |
|---|---|---|---|---|
| **Accepted** | 0.5709 | 0.5823 | **+0.0113** | 2,459 |
| **Other** | 0.6055 | 0.5930 | **-0.0125** | 980 |
| **Rejected** | 0.5931 | 0.6005 | **+0.0074** | 2,872 |

---

## 4. Statistical Conclusion

- Model M8 achieves a **+0.54%** gain in overall test accuracy and **+0.0021** gain in Macro-F1 over M4.
- The 95% paired bootstrap confidence interval for Macro-F1 difference is **[-0.0068, +0.0106]**.
- McNemar's test yielded $p = 0.267038$.
