# Phase 19B: Legal Section Segmentation & Chunking Audit

**Audit Population:** Jud-IPL Validation Partition ($N = 5,320$ judgments)  
**Status:** Completed, Verified  

---

## 1. Executive Summary

Standard legal NLP pipelines naively chunk text from the top of the judgment down, resulting in severe **silent truncation** where the judicial reasoning located in the second half of the document is systematically omitted.
Phase 19B introduces a two-tier segmentation and rhetorical chunking system (`LegalSectionSegmenter` and `LegalSectionChunker`) that deconstructs Indian Supreme Court judgments into canonical rhetorical sections:
1. Facts & Background
2. Procedural History & Lower Court Records
3. Issues Framed
4. Submissions & Counsel Arguments
5. Legal Provisions & Statutory Citations
6. Court Reasoning & Precedent Analysis

Dispositive rulings (`[MASKED_CONCLUDING_DISPOSITION]` and `[MASKED_PROOF_SENTENCE]`) are explicitly stripped to prevent target leakage.

---

## 2. Validation Section Coverage & Distribution

| Canonical Section | Document Coverage (%) | Mean Words | Median Words | Max Words |
|:---|---:|---:|---:|---:|
| **facts** | 99.77% | 541.6 | 448.0 | 3556 |
| **procedural_history** | 99.62% | 444.2 | 362.0 | 1684 |
| **issues** | 2.46% | 1.3 | 0.0 | 271 |
| **arguments** | 99.85% | 557.6 | 457.0 | 2063 |
| **legal_provisions** | 81.79% | 424.7 | 198.0 | 4400 |
| **court_reasoning** | 99.74% | 620.0 | 506.0 | 2398 |

---

## 3. Naive Head-Chunking vs Section-Aware Chunking

| Chunking Paradigm | Reasoning Coverage | Facts Coverage | Procedural History Noise | Truncation Bias |
|:---|---:|---:|:---|:---||
| **Naive Head-Chunking (Phase 18)** | **67.9%** | 100.0% | Heavy (captures 100% of lower court text) | Severe head truncation (drops judicial ratio) |
| **Section-Aware Chunking (Phase 19)** | **99.7%** | 100.0% | Controlled (budgeted at 1 chunk max) | Balanced across rhetorical zones |

### Key Empirical Insights:
- Under naive head chunking, only **67.9%** of long judgments capture the court's core reasoning.
- Under section-aware chunking, **99.7%** of documents represent the court's reasoning.
- By isolating `court_reasoning` and `legal_provisions`, downstream models can directly weigh the bench's legal analysis over adversarial submissions and lower court history.
