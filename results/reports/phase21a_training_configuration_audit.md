# Phase 21A-4: Training Configuration & Class Weighting Audit Report

**Audit Date:** 2026-11-04 10:49:42
**Project:** Jud-IPL Standalone Legal Intelligence Research System
**Scope:** Forensic verification of training subset sampling, class weighting algorithms, feature space dimensions, and solver hyperparameter configurations.

---

## 1. Dataset Partition & Subsampling Audit

In Phase 20, the full three-class training dataset comprised **29,395 judgments**.
To optimize training turnaround on CPU, a stratified subsampling procedure was utilized in `scripts/phase20d_multiclass_training.py`.

### Class Distribution Comparison:

| Class | Full Train Partition ($N = 29,395$) | Full Share (%) | Training Subsample ($N = 6,000$) | Subsample Share (%) |
|---|---|---|---|---|
| **Accepted** | 11,669 | 39.70% | 2,000 | 33.33% |
| **Rejected** | 13,188 | 44.86% | 2,000 | 33.33% |
| **Other** | 4,538 | 15.44% | 2,000 | 33.33% |
| **Total** | **29,395** | **100.0%** | **6,000** | **100.0%** |

---

## 2. Forensic Analysis of Class Weighting (M1 vs M1-W)

Scikit-learn implements automated balanced class weighting using the heuristic formula:
$$w_k = \frac{N}{K \cdot N_k}$$
where $N$ is total samples, $K$ is the number of classes, and $N_k$ is the class sample count.

### Calculated Weights Comparison:

| Target Class | Sampled Training Weights ($N=6,000$) | Full Training Partition Weights ($N=29,395$) |
|---|---|---|
| **Accepted** | **1.000000** (Unity) | **0.839689** (-16.0% downweight) |
| **Rejected** | **1.000000** (Unity) | **0.742973** (-25.7% downweight) |
| **Other** | **1.000000** (Unity) | **2.159174** (+115.9% upweight) |

### Key Audit Finding:
- Because the training subset was strictly balanced ($2,000$ per class), every class received a weight of exactly **$1.000000$**.
- As a direct mathematical consequence, passing `class_weight='balanced'` in Model M1-W exerted **zero weighting adjustment** relative to `class_weight=None` in Model M1.
- This fully explains why M1 and M1-W achieved identical loss functions, identical convergence trajectories, and identical model coefficients.
- **Auditor Note:** If `class_weight='balanced'` is to be meaningfully evaluated in future research, models must be trained on the natural imbalanced corpus ($N=29,395$), where the minority class ("Other") will receive a 2.16x multiplier.

---

## 3. Comprehensive Model Hyperparameter Matrix

| Model ID | Model Architecture | Feature Dimensions | Classifier / Solver | Key Hyperparameters | Random Seed |
|---|---|---|---|---|---|
| **M1** | TF-IDF + Logistic Regression (Unweighted) | 25,000 (word 1-2 gram) | LogisticRegression (L-BFGS) | $C=1.0$, `class_weight=None`, `max_iter=1000`, `sublinear_tf=True` | 42 |
| **M1-W** | TF-IDF + Logistic Regression (Balanced) | 25,000 (word 1-2 gram) | LogisticRegression (L-BFGS) | $C=1.0$, `class_weight='balanced'`, `max_iter=1000`, `sublinear_tf=True` | 42 |
| **M2** | TF-IDF + Calibrated Linear SVM | 25,000 (word 1-2 gram) | LinearSVC + CalibratedClassifierCV | $C=1.0$, `method='sigmoid'`, `cv=3` | 42 |
| **M3** | Sub-word Character N-Gram LR | 25,000 (char 3-5 gram) | LogisticRegression (L-BFGS) | $C=1.0$, `sublinear_tf=True`, `max_iter=1000` | 42 |
| **M4** | Section-Aware TF-IDF Classifier | 50,000 (5 sections combined) | LogisticRegression (L-BFGS) | Preamble (5k), Facts (15k), Args (10k), Reasoning (15k), Disp (5k) | 42 |
| **M5** | Legal Citation-Aware Classifier | 25,015 (25k TF-IDF + 15 Dense) | LogisticRegression (L-BFGS) | Standardized dense legal & procedural features + text | 42 |
| **M6** | InLegalBERT Section-Aware Transformer | 768-d InLegalBERT + Multihead Attn | PyTorch Classifier Head | 3-class attention pooling, 100k head params (proxy eval) | 42 |
| **M7** | Hierarchical Legal Document Model | 768-d InLegalBERT + BiLSTM + Attn | PyTorch Hierarchical Module | 2-layer BiLSTM ($h=128$) + Attention, 952k head params (proxy eval) | 42 |
| **M8** | Hybrid Multi-Class Fusion Ensemble | Ensemble Blend (75k features) | Late Probability Fusion | $w_{M1}=0.50, w_{M4}=0.25, w_{M5}=0.15, w_{M3}=0.10$ | 42 |

---

## 4. Audit Verdict: Training Configuration

- **Configuration Integrity:** **VERIFIED**
- Training pipelines, random seeds, solvers, and hyperparameters conform to documented design.
- The mathematical equivalence of M1 and M1-W is confirmed as an expected consequence of sample-level balance rather than code corruption.
