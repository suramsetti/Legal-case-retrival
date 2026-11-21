# Phase 21A: Comprehensive Model Verification, Prediction Integrity & Leakage Audit Report

**Project:** Jud-IPL Standalone Legal Intelligence Research System  
**Audit Conducted:** 2026-11-04  
**Audit Scope:** Forensic verification of 9 Phase 20 multi-class models, checkpoints, prediction outputs, evaluation metrics, preprocessing pipelines, and dataset partitions.

---

## 1. Executive Summary & Audit Matrix

This audit was conducted under strict scientific constraints: 0% IL-TUR dependency, zero test label contamination, zero retraining, zero checkpoint alteration, and rigorous empirical reproducibility. Every reported claim from Phase 20 was verified against frozen artifacts.

| Model ID | Architecture Family | Reported Acc | Recalc Acc | Disagreements vs Reported | Checkpoint Integrity | Forensic Audit Verdict |
|---|---|---|---|---|---|---|
| **M1** | Monolithic Lexical (TF-IDF + LR) | 57.85% | 57.85% | 0 / 6,297 | Verified (25k params) | **VERIFIED** |
| **M1-W** | Monolithic Lexical (Balanced LR) | 57.85% | 57.85% | 0 / 6,297 | Verified (25k params) | **VERIFIED (Identical to M1)** |
| **M2** | Support Vector Machine (Linear SVM) | 57.61% | 57.61% | 0 / 6,297 | Verified (3 folds) | **VERIFIED** |
| **M3** | Sub-word Lexical (Char 3-5 Gram) | 56.25% | 56.25% | 0 / 6,297 | Verified (25k params) | **VERIFIED** |
| **M4** | Section-Aware Lexical (5 Sections) | 59.06% | 59.06% | 0 / 6,297 | Verified (50k params) | **VERIFIED (Best Standalone)** |
| **M5** | Feature Augmented (Citation + 15 Dense) | 57.22% | 57.22% | 0 / 6,297 | Verified (25,015 params) | **VERIFIED** |
| **M6** | Transformer (InLegalBERT Section) | 57.85% | 57.85% | 0 / 6,297 | Head Only (100k params) | **FAILED (Proxy Evaluated)** |
| **M7** | Hierarchical DL (BiLSTM + Attention) | 59.06% | 59.06% | 0 / 6,297 | Head Only (952k params) | **FAILED (Proxy Evaluated)** |
| **M8** | Hybrid Late Fusion (M1+M4+M5+M3) | 59.50% | 59.50% | 0 / 6,297 | Verified Bundle | **VERIFIED (Best Overall)** |

---

## 2. Checkpoint & Artifact Inventory (Phase 21A-1)

All checkpoints located in `models/checkpoints/phase20/` were inventoried, hashed via SHA-256, and structurally validated:
- `m1_tfidf_lr_unweighted.joblib`: 1.52 MB (SHA-256: `26c62266ad9a2d59...`) — 25,000 features.
- `m1w_tfidf_lr_balanced.joblib`: 1.52 MB (SHA-256: `6b1688e2638448df...`) — 25,000 features.
- `m2_tfidf_calibrated_svm.joblib`: 2.67 MB (SHA-256: `c13224805aa4e93e...`) — 3-fold Platt sigmoid calibrators.
- `m3_char_ngram_lr.joblib`: 1.39 MB (SHA-256: `1ce55ad05ceddde1...`) — 25,000 char n-grams.
- `m4_section_aware_tfidf.joblib`: 3.01 MB (SHA-256: `1d5963eec3d40021...`) — 50,000 section features.
- `m5_citation_aware_lr.joblib`: 1.52 MB (SHA-256: `53b592c95352fea1...`) — 25,015 features.
- `m6_section_transformer.pt`: 0.39 MB (SHA-256: `6d447910d7d531b8...`) — 100,867 head parameters.
- `m7_hierarchical_model.pt`: 3.64 MB (SHA-256: `d6319ce226341690...`) — 952,707 head parameters.
- `m8_hybrid_fusion.joblib`: 130 bytes (SHA-256: `552e0819080d560c...`) — Late fusion dictionary bundle.

---

## 3. Validation Prediction Reproducibility (Phase 21A-2)

Predictions across all 6,297 validation cases in `data/processed/judipl/phase18c/three_class/val_masked.parquet` were reproduced independently and saved in `results/predictions/phase21a/`.
- Every prediction file contains exact columns: `id`, `true_label`, `prob_accepted`, `prob_other`, `prob_rejected`, `predicted_label`.
- Zero NaNs, zero infinite values, and zero probability bounding violations ($p \in [0, 1]$).
- Exact probability normalization confirmed ($\sum_{k} p_k = 1.000000 \pm 10^{-5}$).

---

## 4. Deep Investigation of Identical Metrics (Phase 21A-3)

Forensic analysis solved the root causes behind the three metric-identical model pairs:

### Case A: Model M1 vs Model M1-W
- **Observation:** Both models achieved exactly 57.85% Accuracy, 0.5782 Macro-F1, 60.08% Balanced Accuracy, and identical confusion matrices.
- **Root Cause:** In `scripts/phase20d_multiclass_training.py`, models were trained on a balanced subsample with 2,000 instances per class ($N=6,000$, $K=3$). Scikit-learn computes balanced class weights as $w_k = \frac{N}{K \cdot N_k} = \frac{6000}{3 \cdot 2000} = 1.000000$. Hence, passing `class_weight='balanced'` resulted in weights of exactly $1.0$ for all classes, reducing the balanced loss function identically to unweighted loss.
- **Verification:** Coefficients differ by $< 10^{-7}$. M1-W is bitwise identical to M1 by algebraic necessity.

### Case B: Model M4 vs Model M7
- **Observation:** Both models achieved exactly 59.06% Accuracy, 0.5913 Macro-F1, 61.35% Balanced Accuracy, and identical confusion matrices.
- **Root Cause:** In `scripts/phase20d_multiclass_training.py`, M7 validation probabilities were calculated using a convex combination with a uniform distribution:
  $$P(M7) = 0.70 \cdot [1/3, 1/3, 1/3] + 0.30 \cdot P(M4)$$
  Adding a constant scalar ($0.70/3$) does not alter class rank ordering: $\text{argmax}_k(P(M7)_k) \equiv \text{argmax}_k(P(M4)_k)$. Consequently, discrete predictions and confusion matrices were 100% identical (0 disagreements out of 6,297 cases).
- **Audit Verdict:** M7 is a proxy implementation. It did not run full hierarchical BiLSTM forward inference during evaluation.

### Case C: Model M1 vs Model M6
- **Observation:** Both models achieved exactly 57.85% Accuracy, 0.5782 Macro-F1, 60.08% Balanced Accuracy, and identical confusion matrices.
- **Root Cause:** In `scripts/phase20d_multiclass_training.py`, M6 validation probabilities were calculated as:
  $$P(M6) = 0.80 \cdot [1/3, 1/3, 1/3] + 0.20 \cdot P(M1)$$
  Adding a uniform vector preserved the argmax ranking of M1 across all 6,297 documents.
- **Audit Verdict:** M6 is a proxy implementation. It did not run full transformer forward inference.

---

## 5. Training Configuration Audit (Phase 21A-4)

- **Training Subsample:** 6,000 cases (2,000 Accepted, 2,000 Rejected, 2,000 Other) sampled with `random_state=42`.
- **Full Partition Size:** 29,395 cases (11,669 Accepted, 13,188 Rejected, 4,538 Other).
- **Natural Class Weights on Full Data:** Accepted: 0.8397, Rejected: 0.7430, Other: 2.1592.
- **Hyperparameters:** All TF-IDF models used `sublinear_tf=True`, L-BFGS optimizer, $C=1.0$, `max_iter=1000`.

---

## 6. Comprehensive Leakage Audit (Phase 21A-5)

- **Document ID Overlap:**
  - Train vs Val: **0** (100% disjoint)
  - Train vs Test: **0** (100% disjoint)
  - Val vs Test: **0** (100% disjoint)
- **Raw ID Overlap:** **0** across all split pairs.
- **Duplicate Group Overlap:** **0** cluster leakage across all split pairs.
- **Masked Text Hash Overlap:** Exactly 3 near-duplicates identified across 42,005 judgments (0.007% leakage rate) resulting from disparate source IDs.
- **Proof Sentence Neutralization:** 100% enforced (`proof_masked == True`). Dispositive ruling regex cleanliness: 82.20% in Train, 81.45% in Val, 82.23% in Test (residual matches are factual trial court citations in procedural history).
- **IL-TUR Dependency:** **0.00%**. 100.0% of records originate from Jud-IPL.

---

## 7. Recalculated Validation Metrics (Phase 21A-7)

| Model ID | Recalc Accuracy | Balanced Acc | Macro-F1 | Weighted-F1 | ROC-AUC OvR | Log Loss | F1 Accepted | F1 Other | F1 Rejected |
|---|---|---|---|---|---|---|---|---|---|
| **M1** | 57.85% | 60.08% | 0.5782 | 0.5786 | 0.7524 | 0.8868 | 0.5765 | 0.5768 | 0.5812 |
| **M1-W** | 57.85% | 60.08% | 0.5782 | 0.5786 | 0.7524 | 0.8868 | 0.5765 | 0.5768 | 0.5812 |
| **M2** | 57.61% | 60.01% | 0.5757 | 0.5761 | 0.7520 | 0.8800 | 0.5805 | 0.5738 | 0.5729 |
| **M3** | 56.25% | 59.27% | 0.5647 | 0.5621 | 0.7505 | 0.8828 | 0.5553 | 0.5748 | 0.5639 |
| **M4** | 59.06% | 61.35% | 0.5913 | 0.5905 | 0.7721 | 0.8572 | 0.5836 | 0.5947 | 0.5955 |
| **M5** | 57.22% | 59.81% | 0.5729 | 0.5720 | 0.7500 | 0.8887 | 0.5702 | 0.5761 | 0.5724 |
| **M6** | 57.85% | 60.08% | 0.5782 | 0.5786 | 0.7524 | 1.0380 | 0.5765 | 0.5768 | 0.5812 |
| **M7** | 59.06% | 61.35% | 0.5913 | 0.5905 | 0.7721 | 0.9808 | 0.5836 | 0.5947 | 0.5955 |
| **M8** | **59.50%** | **61.71%** | **0.5950** | **0.5951** | **0.7699** | **0.8637** | **0.5913** | **0.5950** | **0.5986** |

---

## 8. Final Audit Verdicts & Recommendations

1. **M1, M2, M3, M4, M5, M8:** **VERIFIED**
   - Solid, reproducible, leakage-free implementations.
   - M4 confirms the substantial predictive value of legal section decomposition (+1.21% accuracy over monolithic TF-IDF).
   - M8 confirms that multi-view late fusion yields optimal three-class performance (59.50% Acc, 0.5950 Macro-F1).
2. **M1-W:** **VERIFIED (MATHEMATICALLY IDENTICAL TO M1)**
   - Re-train on the full imbalanced training corpus ($N=29,395$) in Phase 21B to evaluate genuine class-weight impact.
3. **M6 & M7:** **FAILED (PROXY IMPLEMENTATION DETECTED)**
   - Checkpoints contain only head modules without full InLegalBERT weights.
   - Predictions were simulated via uniform probability blending.
   - Recommendation: Full GPU end-to-end fine-tuning should be scheduled when CUDA resources are provisioned.
