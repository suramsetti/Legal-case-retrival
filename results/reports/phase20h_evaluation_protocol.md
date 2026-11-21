# Phase 20H: Multi-Class Evaluation Integrity & Test Protocol

**Date:** 2026-11-03 23:50:00  
**Target Task:** Three-Class Legal Outcome Prediction (`Accepted`, `Rejected`, `Other`)  
**Status:** Pre-registered, Audited, Frozen  

---

## 1. Prior-Exposure Audit & Methodological Integrity

A critical prerequisite of empirical legal NLP is the transparent disclosure of prior data exposure:
1. **Shared History:** During Phase 18, binary judgments were partitioned and evaluated. In Phase 20, the full Jud-IPL corpus (including 6,485 'Other' judgments) was partitioned into a unified three-class split:
   - Train: 29,395 cases
   - Validation: 6,297 cases
   - Test: 6,313 cases
2. **Exposure Overlap:** An exact ID-matching audit reveals that **812 cases (12.86%)** in the Phase 20 Three-Class Test partition were previously part of the Phase 18 Binary Test partition.
3. **Scientific Inadmissibility as 'Virgin Blind Test':** Because hyperparameter selection in earlier phases was exposed to these cases in binary form, claiming that the Phase 20 test set constitutes a completely virgin independent evaluation would violate research integrity.
4. **Prohibition of Label Fabrication:** All valid judgments in Jud-IPL have been audited. No unused documents with ground-truth binary or three-class labels exist. Therefore, per Constraint 4 ('Do not fabricate or synthetically generate outcome labels'), **no artificial holdout is manufactured**.

---

## 2. Requirements for Future True Independent Evaluation

For future verification of genuine unseen generalization:
- **External Temporal Extension:** Ingestion of newly published Supreme Court of India judgments from 2024–2026.
- **Zero Contamination:** Raw documents must be hashed and cross-checked against all 42,006 Jud-IPL document hashes.
- **Sanitization:** Apply the exact `LegalSectionSegmenter` and dispositive neutralization mask prior to running the frozen `m8_hybrid_fusion.joblib` model.

---

## 3. Frozen Checkpoints & Cryptographic Manifest

| Checkpoint Name | Size | SHA-256 Hash |
|:---|---:|:---|
| `m1_tfidf_lr_unweighted.joblib` | 1.52 MB | `26c62266ad9a2d59...` |
| `m1w_tfidf_lr_balanced.joblib` | 1.52 MB | `6b1688e2638448df...` |
| `m2_tfidf_calibrated_svm.joblib` | 2.67 MB | `c13224805aa4e93e...` |
| `m3_char_ngram_lr.joblib` | 1.39 MB | `1ce55ad05ceddde1...` |
| `m4_section_aware_tfidf.joblib` | 3.01 MB | `1d5963eec3d40021...` |
| `m5_citation_aware_lr.joblib` | 1.52 MB | `53b592c95352fea1...` |
| `m6_section_transformer.pt` | 0.39 MB | `6d447910d7d531b8...` |
| `m7_hierarchical_model.pt` | 3.64 MB | `d6319ce226341690...` |
| `m8_hybrid_fusion.joblib` | 0.00 MB | `552e0819080d560c...` |
