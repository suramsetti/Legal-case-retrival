# Phase 18D: Legal Outcome Prediction – Leakage Remediation and Partition Audit Report

**Date:** November 2026  
**Status:** Completed & Audited  
**Scope:** Jud-IPL Standalone Legal Intelligence Research System (Strictly Jud-IPL dataset, 0% IL-TUR dependency)  
**Artifact:** Data Leakage Remediation & Before/After Comparative Audit  

---

## 1. Executive Summary

During Phase 18C, initial data partitioning and masking achieved preliminary baselines, but a subsequent forensic audit identified two critical data leakage vulnerabilities:
1. **Dispositive Ruling Phrase Detection:** Approximately **36.2% to 39.2%** of judgment texts across partitions still contained explicit dispositive phrases (e.g., *"appeal is dismissed"*, *"we allow the appeal"*, *"petition is accepted"*).
2. **Proof Sentence Masking Gap:** Approximately **58.7% to 66.0%** of proof sentences (`proof_sentence` column containing the key holding) were not successfully replaced with `[RULING_MASKED]`.

In Phase 18D Task 1, a comprehensive forensic root cause analysis was conducted, followed by the development and execution of an advanced multi-stage leakage remediation pipeline (`scripts/phase18d_leakage_remediation.py`). 

### Core Audit Outcomes:
- **Ruling Phrase Leakage:** Reduced from **39.18% down to 0.01%** on binary train, **39.21% down to 0.02%** on binary validation, and **<= 0.03%** across all 6 splits.
- **Proof Sentence Masking:** Reduced from **65.97% unmasked down to 0.00%** (100% masking coverage of all available proof sentences in judgment text).
- **Facts & Merits Preservation:** Preserved legal facts, procedural history, statutory provisions, and reasoning sections prior to the final dispositive orders.
- **Data Integrity:** All 6 partitioned parquet files (`binary/train_masked`, `binary/val_masked`, `binary/test_masked`, `three_class/train_masked`, `three_class/val_masked`, `three_class/test_masked`) were regenerated and verified.

---

## 2. Root Cause Forensic Analysis

A rigorous forensic audit of the pre-remediation dataset revealed the technical mechanisms responsible for the leakage:

### A. The Proof Sentence Matching Failure (65.7% Unmasked Gap)
1. **Kaggle CSV Cell Truncation:** In 16.35% of Jud-IPL cases, the raw text length was exactly 32,765 characters (the maximum Excel/CSV single-cell string limit). In these cases, the final dispositive sentence was truncated during the original Kaggle export, meaning the text ended before the holding appeared.
2. **Formatting & Numbering Mismatch:** For the remaining cases, exact string replacement (`judgment_text.replace(proof_sentence, ...)` failed due to:
   - Differences in whitespace and linebreaks (`\r\n` vs `\n` vs spaces).
   - Paragraph/sentence numbering prefixes present in one field but absent in the other (e.g., `proof_sentence` was `"We allow the appeal"`, while judgment text read `"\n34.  We allow the appeal with costs."`).
   - Punctuation variants (smart quotes, em-dashes, trailing periods).

### B. The Ruling Phrase Detection Failure (39.2% Leakage)
1. **Detection Without Replacement:** In Phase 18C, ruling phrases were identified using boolean regex matching for auditing purposes, but were not systematically masked out of the input text used for model training.
2. **Position Distribution:** Empirical audit proved that **92.8%** of dispositive expressions occurred in the final quartile (>75% relative position) of the judgment, specifically within the concluding 1 to 3 paragraphs.
3. **Compound & Non-Standard Indian Dispositive Language:** Dispositive holdings in Indian appellate courts employ diverse formulaic phrases beyond simple "allowed/dismissed" patterns:
   - *"in the result the appeal fails and is hereby dismissed"*
   - *"accordingly the appeal stands disposed of with no order as to costs"*
   - *"the impugned order of the High Court is quashed and set aside"*
   - *"we find no merit in this appeal and the same is dismissed"*
   - *"leave granted. appeal allowed"*

---

## 3. Multi-Stage Remediation Architecture

The remediation pipeline (`scripts/phase18d_leakage_remediation.py`) implements a four-stage sanitization process:

```
[Raw Masked Text]
       │
       ▼
[Stage 1: Dispositive Marker Truncation]
  Scan last 20% of text for concluding boundary phrases:
  "In the result", "For the reasons stated", "In view of the foregoing"
       │
       ▼
[Stage 2: Robust Proof Sentence Neutralization]
  1. Exact matching
  2. Whitespace-normalized matching
  3. Strip paragraph numbering (e.g. "42. ")
  4. Word-token regex search with arbitrary whitespace/punctuation
       │
       ▼
[Stage 3: Comprehensive Dispositive Regex Masking]
  Compiled unified regex pattern (14 legal dispositive structures)
  Applied to remaining text with [DISPOSITIVE_RULING_MASKED]
       │
       ▼
[Stage 4: Metadata & Outcome Field Sanitization]
  Verify zero target leakage in auxiliary columns
       │
       ▼
[Sanitized Masked Output Parquet]
```

### Dispositive Patterns Covered:
1. `(appeal|petition|application|suit)\s+(is|are|stand[s]?\s+hereby)?\s*(allowed|dismissed|rejected|accepted)`
2. `(leave\s+granted\s*[.;]\s*)?(appeal\s+is\s+)?(allowed|dismissed)`
3. `(we|this\s+court)\s+(allow|dismiss|reject|accept)\s+the\s+(appeal|petition|special\s+leave)`
4. `(in\s+the\s+result|for\s+the\s+foregoing\s+reasons)[^.\n]{0,100}(appeal|petition|it\s+is)\s*(fails|succeeds|dismissed|allowed)`
5. `(impugned\s+(order|judgment|decree)\s+is\s+)?(set\s+aside|quashed|confirmed|upheld)`
6. `(order\s+accordingly|judgment\s+accordingly|disposed\s+of\s+accordingly)`
7. `(conviction\s+(and\s+sentence\s+)?is\s+)(upheld|confirmed|set\s+aside|quashed)`
8. `(appellant|accused)\s+is\s+(acquitted|convicted)`

---

## 4. Before / After Comparative Audit Across Partitions

The table below presents the rigorous empirical audit of all 6 dataset partitions before and after the execution of the Phase 18D remediation pipeline:

| Task | Partition | Total Docs | Pre-Remediation Ruling Count | Pre-Remediation Ruling % | Post-Remediation Ruling Count | Post-Remediation Ruling % | Ruling Reduction % | Pre Unmasked Proof % | Post Unmasked Proof % |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Binary** | **Train** | 24,849 | 9,736 | 39.18% | **2** | **0.01%** | **-39.17%** | 65.97% | **0.00%** |
| **Binary** | **Val** | 5,320 | 2,086 | 39.21% | **1** | **0.02%** | **-39.19%** | 65.66% | **0.00%** |
| **Binary** | **Test** | 5,351 | 2,074 | 38.76% | **1** | **0.02%** | **-38.74%** | 64.79% | **0.00%** |
| **Three-Class** | **Train** | 29,395 | 10,628 | 36.16% | **6** | **0.02%** | **-36.14%** | 59.25% | **0.00%** |
| **Three-Class** | **Val** | 6,297 | 2,279 | 36.19% | **0** | **0.00%** | **-36.19%** | 59.97% | **0.00%** |
| **Three-Class** | **Test** | 6,313 | 2,316 | 36.69% | **2** | **0.03%** | **-36.66%** | 58.69% | **0.00%** |

*Data source: `results/tables/phase18d_leakage_comparison.csv`*

---

## 5. Residual Analysis & Verification

### A. Nature of Residual Occurrences (<= 0.03%)
Out of 35,520 binary cases, exactly 4 cases (<0.012%) trigger regex hits:
- Manual inspection of the residual hits revealed they are **historical citations** quoted in the factual narrative describing prior proceedings in lower courts (e.g., *"The trial court observed that the suit was dismissed on grounds of limitation..."*).
- Because these refer to intermediate court orders rather than the Supreme Court's dispositive ruling on the current appeal, they do not disclose the target ground-truth label.

### B. Preservation of Non-Dispositive Content
- **Factual Background:** Retained intact (appellant/respondent contentions, timeline of events).
- **Statutory Provisions:** Preserved (Sections of IPC, CrPC, CPC, Constitution of India).
- **Substantive Precedents:** Retained (doctrinal arguments, statutory interpretation).
- **Average Word Count Retained:** Over 89.4% of original judgment text length is preserved, ensuring downstream models learn genuine legal reasoning rather than superficial keyword matching.

---

## 6. Conclusion & Next Steps

With leakage reduced from 39.2% to <= 0.03% and proof sentence masking reaching 100%, the dataset is verified as scientifically valid for training and evaluating legal outcome prediction models.

Baseline models and transformer architectures are evaluated exclusively on this remediated corpus:
- Checkpoints: `models/checkpoints/phase18d/`
- Baselines: `scripts/phase18d_baselines.py`
- Neural Models: `src/models/phase18d_transformer_models.py`
