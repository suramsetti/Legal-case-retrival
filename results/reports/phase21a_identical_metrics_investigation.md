# Phase 21A-3: Identical Metrics Investigation & Prediction Integrity Audit

**Audit Date:** 2026-11-04 10:52:15
**Project:** Jud-IPL Standalone Legal Intelligence Research System
**Scope:** Forensic diagnosis and root-cause analysis of identical metrics across Phase 20 models.

---

## 1. Executive Summary & Forensic Findings

In Phase 20, three pairs of models exhibited identical validation metrics (Accuracy, Macro-F1, Balanced Accuracy, and Confusion Matrices). This forensic investigation evaluated all $\binom{9}{2} = 36$ pairwise model combinations and audited the underlying training scripts and checkpoints.

The three investigated anomalies are:
1. **Case A (M1 vs M1-W):** Identical metrics across all dimensions (Accuracy: 57.85%, Macro-F1: 0.5782).
   - **Root Cause:** In `scripts/phase20d_multiclass_training.py`, models were trained on a perfectly balanced subsample ($N_k = 2,000$ per class). Under scikit-learn's weighting formula $w_k = N / (K \cdot N_k)$, when all classes have identical sample counts, $w_k = 6000 / (3 \cdot 2000) = 1.0000$ for all classes. Consequently, `class_weight='balanced'` was algebraically equivalent to `class_weight=None`.
   - **Parameter Parity:** Max coefficient difference is **0.00e+00**, confirming coefficients are bitwise identical.
2. **Case B (M4 vs M7):** Identical discrete metrics (Accuracy: 59.06%, Macro-F1: 0.5913, Confusion Matrix identical).
   - **Root Cause:** Due to CPU computational limits during Phase 20, M7 validation probabilities were generated via proxy formula:
     $$P(M7) = 0.70 \cdot [1/3, 1/3, 1/3] + 0.30 \cdot P(M4)$$
     Because the addition of a constant scalar ($0.70 / 3$) to each class logit/probability is strictly order-preserving, $\text{argmax}_k(P(M7)_k) \equiv \text{argmax}_k(P(M4)_k)$ for 100% of cases.
   - **Empirical Check:** Disagreements = **0 / 6,297** (100.0% agreement), Probability Correlation = **1.0000**, Probability MAE = **0.1391**.
3. **Case C (M1 vs M6):** Identical discrete metrics (Accuracy: 57.85%, Macro-F1: 0.5782, Confusion Matrix identical).
   - **Root Cause:** Similarly, M6 validation probabilities were generated via proxy formula:
     $$P(M6) = 0.80 \cdot [1/3, 1/3, 1/3] + 0.20 \cdot P(M1)$$
     This strictly preserves the argmax ranking of M1, resulting in 100.0% discrete label agreement and identical confusion matrices.
   - **Empirical Check:** Disagreements = **0 / 6,297** (100.0% agreement), Probability Correlation = **1.0000**, Probability MAE = **0.1113**.

---

## 2. Target Investigation Pairs Summary

| Metric / Dimension | Case A: M1 vs M1-W | Case B: M4 vs M7 | Case C: M1 vs M6 |
|---|---|---|---|
| **Model Architectures** | Unweighted LR vs Balanced LR | Section TF-IDF vs Hierarchical Model | TF-IDF LR vs Section Transformer |
| **Discrete Agreement (%)** | **100.00%** | **100.00%** | **100.00%** |
| **Disagreement Count** | **0 / 6,297** | **0 / 6,297** | **0 / 6,297** |
| **Probability Correlation ($r$)** | **1.000000** | **1.000000** | **1.000000** |
| **Probability MAE** | **0.000000** | **0.139063** | **0.111314** |
| **Probability Max Diff** | **0.000000** | **0.464719** | **0.509114** |
| **Exact Probability Match?** | **True** | **False** | **False** |
| **Exact Discrete Match?** | **True** | **True** | **True** |
| **Model Weights Identical?** | **YES (diff < 1e-7)** | **NO (M7 has distinct PyTorch weights)** | **NO (M6 has distinct PyTorch weights)** |

---

## 3. Mathematical & Empirical Proofs

### Case A: Why M1 and M1-W Produced Bitwise Identical Classifiers
In scikit-learn's `LogisticRegression(class_weight='balanced')`, the class weight $w_k$ for class $k \in \{1, \dots, K\}$ is defined as:
$$w_k = \frac{N}{K \cdot N_k}$$
Where $N = \sum_{k=1}^K N_k$ is the total number of training instances, and $N_k$ is the number of instances in class $k$.

In `scripts/phase20d_multiclass_training.py` lines 134–139:
```python
sample_dfs = [
    tr_df[tr_df["label"] == c].sample(n=min(2000, len(tr_df[tr_df["label"] == c])), random_state=RANDOM_SEED)
    for c in CANONICAL_CLASSES
]
train_sub = pd.concat(sample_dfs).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
```
Here $K = 3$, and $N_1 = N_2 = N_3 = 2,000$. Thus $N = 6,000$.
$$w_k = \frac{6,000}{3 \cdot 2,000} = \frac{6,000}{6,000} = 1.000000 \quad \forall k \in \{1, 2, 3\}$$
When every sample has weight $1.0$, the weighted loss function reduces identically to the standard unweighted loss function:
$$\mathcal{L}_{\text{balanced}}(\theta) = \sum_{i=1}^N 1.0 \cdot \ell(y_i, f(x_i; \theta)) = \sum_{i=1}^N \ell(y_i, f(x_i; \theta)) = \mathcal{L}_{\text{unweighted}}(\theta)$$
Given identical random seeds and solver configurations (L-BFGS), the optimization trajectories and final parameters are mathematically identical.
- **Coefficient maximum absolute difference:** `0.00e+00`
- **Intercept maximum absolute difference:** `0.00e+00`
- **Conclusion:** M1-W was not overwritten; it naturally solved to the exact same weights as M1 because the training subset was pre-balanced.

---

### Cases B & C: Why M6 and M7 Replicated Discrete Predictions of M1 and M4
In Phase 20D, proxy probability formulations were used during the benchmark generation:
1. For M6: $p_k^{(M6)} = 0.80 \cdot \frac{1}{3} + 0.20 \cdot p_k^{(M1)}$
2. For M7: $p_k^{(M7)} = 0.70 \cdot \frac{1}{3} + 0.30 \cdot p_k^{(M4)}$

Let $p = (p_1, p_2, p_3)$ be a probability vector such that $p_k \ge 0$ and $\sum p_k = 1$. Let $\lambda \in (0, 1)$ and $u = (1/3, 1/3, 1/3)$.
The convex mixture is:
$$q_k = (1 - \lambda) u_k + \lambda p_k = \frac{1 - \lambda}{3} + \lambda p_k$$
For any two classes $j$ and $k$:
$$q_j - q_k = \left(\frac{1 - \lambda}{3} + \lambda p_j\right) - \left(\frac{1 - \lambda}{3} + \lambda p_k\right) = \lambda (p_j - p_k)$$
Since $\lambda > 0$:
$$q_j > q_k \iff p_j > p_k$$
Therefore:
$$\text{argmax}_{k \in \{1, 2, 3\}} q_k \equiv \text{argmax}_{k \in \{1, 2, 3\}} p_k$$
The predicted discrete class is completely invariant to any uniform blend with positive weight $\lambda > 0$.
Consequently:
- Every discrete prediction $\hat{y}^{(M6)}_i$ matches $\hat{y}^{(M1)}_i$ exactly.
- Every discrete prediction $\hat{y}^{(M7)}_i$ matches $\hat{y}^{(M4)}_i$ exactly.
- All discrete evaluation metrics (Accuracy, Balanced Accuracy, Precision, Recall, Macro-F1, Weighted-F1, and Confusion Matrix) are 100.0% identical.
- However, continuous loss (Cross-Entropy / Log Loss) is drastically affected:
  - M1 Log Loss: 0.8868 vs M6 Log Loss: 1.0380
  - M4 Log Loss: 0.8572 vs M7 Log Loss: 0.9808
  This demonstrates that the probability distributions were shrunk toward maximum uncertainty ($1/3$), which degraded probabilistic sharpness while maintaining the same discrete decision boundary.

---

## 4. Complete Pairwise Prediction Similarity Matrix (36 Model Pairs)

| Model A | Model B | Agreement (%) | Disagreements | Prob Corr ($r$) | Prob MAE | Max Diff |
|---|---|---|---|---|---|---|
| M1 | M1-W | 100.00% | 0 | 1.0000 | 0.0000 | 0.0000 |
| M1 | M2 | 90.63% | 590 | 0.9726 | 0.0346 | 0.2133 |
| M1 | M3 | 79.83% | 1270 | 0.9207 | 0.0542 | 0.4142 |
| M1 | M4 | 73.69% | 1657 | 0.8433 | 0.1021 | 0.6410 |
| M1 | M5 | 92.22% | 490 | 0.9884 | 0.0182 | 0.4851 |
| M1 | M6 | 100.00% | 0 | 1.0000 | 0.1113 | 0.5091 |
| M1 | M7 | 73.69% | 1657 | 0.8433 | 0.0957 | 0.4852 |
| M1 | M8 | 89.80% | 642 | 0.9830 | 0.0263 | 0.1895 |
| M1-W | M2 | 90.63% | 590 | 0.9726 | 0.0346 | 0.2133 |
| M1-W | M3 | 79.83% | 1270 | 0.9207 | 0.0542 | 0.4142 |
| M1-W | M4 | 73.69% | 1657 | 0.8433 | 0.1021 | 0.6410 |
| M1-W | M5 | 92.22% | 490 | 0.9884 | 0.0182 | 0.4851 |
| M1-W | M6 | 100.00% | 0 | 1.0000 | 0.1113 | 0.5091 |
| M1-W | M7 | 73.69% | 1657 | 0.8433 | 0.0957 | 0.4852 |
| M1-W | M8 | 89.80% | 642 | 0.9830 | 0.0263 | 0.1895 |
| M2 | M3 | 76.24% | 1496 | 0.8827 | 0.0688 | 0.4348 |
| M2 | M4 | 72.08% | 1758 | 0.8260 | 0.1044 | 0.6175 |
| M2 | M5 | 87.64% | 778 | 0.9613 | 0.0396 | 0.4683 |
| M2 | M6 | 90.63% | 590 | 0.9726 | 0.1254 | 0.4610 |
| M2 | M7 | 72.08% | 1758 | 0.8260 | 0.1091 | 0.4724 |
| M2 | M8 | 85.56% | 909 | 0.9567 | 0.0426 | 0.2491 |
| M3 | M4 | 67.95% | 2018 | 0.7819 | 0.1147 | 0.6775 |
| M3 | M5 | 78.50% | 1354 | 0.9097 | 0.0578 | 0.4629 |
| M3 | M6 | 79.83% | 1270 | 0.9207 | 0.1209 | 0.5315 |
| M3 | M7 | 67.95% | 2018 | 0.7819 | 0.1068 | 0.5080 |
| M3 | M8 | 79.72% | 1277 | 0.9219 | 0.0550 | 0.3977 |
| M4 | M5 | 73.35% | 1678 | 0.8385 | 0.1026 | 0.7057 |
| M4 | M6 | 73.69% | 1657 | 0.8433 | 0.1758 | 0.6270 |
| M4 | M7 | 100.00% | 0 | 1.0000 | 0.1391 | 0.4647 |
| M4 | M8 | 83.18% | 1059 | 0.9246 | 0.0767 | 0.4836 |
| M5 | M6 | 92.22% | 490 | 0.9884 | 0.1140 | 0.5085 |
| M5 | M7 | 73.35% | 1678 | 0.8385 | 0.0981 | 0.5790 |
| M5 | M8 | 87.90% | 762 | 0.9766 | 0.0296 | 0.3681 |
| M6 | M7 | 73.69% | 1657 | 0.8433 | 0.0384 | 0.1891 |
| M6 | M8 | 89.80% | 642 | 0.9830 | 0.1208 | 0.5152 |
| M7 | M8 | 83.18% | 1059 | 0.9246 | 0.0961 | 0.4443 |


---

## 5. Audit Verdict & Remediation Guidance

1. **Model M1 vs M1-W Verdict:** **VERIFIED (MATHEMATICAL IDENTITY)**
   - Not an error or fraud; a direct mathematical consequence of training with `class_weight='balanced'` on a balanced sample ($N_k = 2,000$).
   - Remediation: To evaluate genuine class-weight impact, train on the natural imbalanced training set ($N = 29,395$, where Accepted=11,669, Rejected=13,188, Other=4,538).

2. **Model M6 & M7 Verdict:** **FAILED / PROXY ARTIFACT DETECTED**
   - Checkpoints `m6_section_transformer.pt` and `m7_hierarchical_model.pt` exist and contain valid PyTorch module weights (100k and 952k params), but the validation predictions were generated using a uniform lexical blend proxy rather than full forward-pass transformer inference.
   - Remediation: These models must be clearly designated as **Proxy Baseline Artifacts** and should not be claimed as fully trained end-to-end deep transformers until full GPU fine-tuning is completed.

3. **Model M8 Verdict:** **VERIFIED (PROBABILITY LATE FUSION ENSEMBLE)**
   - M8 genuinely combines probability vectors from M1, M4, M5, and M3 with weights $[0.50, 0.25, 0.15, 0.10]$. Its distinct predictions and improved accuracy (59.50%) and Macro-F1 (0.5950) are mathematically reproducible.
