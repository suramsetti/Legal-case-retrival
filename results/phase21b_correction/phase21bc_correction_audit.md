# Phase 21B-C Frozen Prediction Numerical Consistency Audit

## Scope and safeguards

This correction audit uses only Jud-IPL, the existing Phase 21B duplicate exclusion manifest, the existing eligible test labels, and the two frozen prediction CSVs. It does not retrain, refit, tune thresholds, alter partitions, inspect labels to alter predictions, or overwrite Phase 21B artifacts.

Eligible population: **6,311** of the original **6,313** test rows; exclusions: **2**. Historical Phase 18 binary overlap is **812** and the remaining eligible subset is **5,499**. The latter is not an independently collected external test set and higher performance on it does not prove zero contamination.

## Corrected metrics

model_id  accuracy  balanced_accuracy  macro_precision  macro_recall  macro_f1  weighted_f1  roc_auc_ovr  roc_auc_ovo  pr_auc_macro  log_loss  brier_score_mean
      M4  0.586595           0.610330         0.578046      0.610330   0.58984     0.586385     0.768505     0.768505      0.650021  0.862997          0.520927
      M8  0.591982           0.612561         0.580512      0.612561   0.59191     0.592229     0.765085     0.765085      0.652587  0.868916          0.521871

ROC-AUC OvR and OvO are independently computed with macro averaging. Brier is the mean per-document sum of squared errors across the three one-hot class probabilities. ECE uses 10 equal-width bins, `(left, right]`, top-class confidence, and top-class correctness.

## Confusion matrices

Rows are actual and columns predicted in `['Accepted', 'Other', 'Rejected']`. The corrected M4 diagonal is **3,702**; this resolves the matrix itself. The old McNemar claim (3,366 both correct plus 334 M4-only correct = 3,700) is inconsistent with the frozen predictions.

## Paired correctness and McNemar

`{
  "n11_both_correct": 3277,
  "n00_both_incorrect": 2150,
  "n01_m4_incorrect_m8_correct": 459,
  "n10_m4_correct_m8_incorrect": 425,
  "n": 6311,
  "m4_correct": 3702,
  "m8_correct": 3736,
  "net_m8_minus_m4": 34,
  "chi2_continuity_corrected": 1.2319004524886878,
  "p_value_continuity_corrected": 0.26703776344580454,
  "p_value_exact_binomial": 0.2670273666195135,
  "discordant": 884
}`

The continuity-corrected chi-square and exact binomial test answer related but different questions. The former uses a chi-square approximation with a continuity correction; the exact test conditions on the discordant total and uses the exact binomial tail, so their p-values need not match.

## Bootstrap

The corrected bootstrap table reports M8 minus M4 for accuracy, balanced accuracy, Macro-F1, and weighted-F1 using 1,000 paired document-index resamples, seed 42, and percentile 95% intervals. A CI containing zero does not establish equivalence; it means this evaluation does not demonstrate statistically significant superiority.

## Original versus corrected

See `original_vs_corrected_metrics.csv` and `phase21bc_discrepancy_register.json`. Original Phase 21B files remain preserved in their original directories.

## Research decision

The recommendation is not forced by this audit. M4 remains the operationally preferred primary model only if the broader Phase 21B evidence (comparable predictive metrics, lower log loss/ECE, latency, and simplicity) is retained; the corrected frozen-evidence comparison does not demonstrate statistically significant M8 superiority.
