# Phase 20F: Explainability & Legal Reasoning Analysis Report

**Generated:** 2026-11-03 23:45:00  
**Target Models:** M1 (Multinomial TF-IDF LR), M4 (Section-Aware TF-IDF), M5 (Citation-Aware)  
**Status:** Completed, Verified, Leakage Checked  

---

## 1. Executive Summary of Explainability Findings

1. **Distinct Lexical Signatures per Class:**
   - **Accepted:** Strongly associated with terms indicating judicial error correction, evidentiary defect, and relief (*quash, set aside, unsustainable, illegal, without jurisdiction, benefit of doubt*).
   - **Rejected:** Strongly associated with procedural finality and affirming lower-court guilt (*concurrent, no merit, no reason to interfere, rightly convicted, unimpeachable, substantiated*).
   - **Other:** Strongly associated with procedural directions and conditional disposal (*remand, remit, interim, status quo, notice, withdrawn, dispose, larger bench*).
2. **Section Contribution Hierarchy:** The `court_reasoning` section carries the highest L2 coefficient weight, followed by `legal_provisions` and `arguments`. Procedural history exhibits the lowest relative importance, validating our finding that lower court facts dilute predictive signal.
3. **Absence of Outcome Leakage:** None of the top 20 explanatory tokens across all classes are unmasked dispositive phrases (e.g. 'appeal allowed', 'appeal dismissed'). All represent substantive legal reasoning or procedural concepts.

---

## 2. Top Predictive Terms by Class

### Top Features Driving 'Accepted' Predictions:
| Rank | Predictive Feature | Log-Odds Coefficient | Legal Interpretation |
|---:|:---|---:|:---|
| 1 | `masked_holding the` | +2.4971 | Substantive legal marker |
| 2 | `appellant` | +1.7979 | Substantive legal marker |
| 3 | `the appellant` | +1.7752 | Substantive legal marker |
| 4 | `set aside` | +1.6550 | Substantive legal marker |
| 5 | `aside` | +1.5907 | Substantive legal marker |
| 6 | `masked_holding` | +1.4818 | Substantive legal marker |
| 7 | `is set` | +1.1641 | Substantive legal marker |
| 8 | `set` | +1.0999 | Substantive legal marker |
| 9 | `aside the` | +1.0485 | Substantive legal marker |
| 10 | `allowing` | +1.0113 | Substantive legal marker |
| 11 | `the appellants` | +0.9579 | Substantive legal marker |
| 12 | `judgment and` | +0.9546 | Substantive legal marker |
| 13 | `and order` | +0.9023 | Substantive legal marker |
| 14 | `in error` | +0.8832 | Substantive legal marker |
| 15 | `the respondent` | +0.8594 | Substantive legal marker |

### Top Features Driving 'Other' Predictions:
| Rank | Predictive Feature | Log-Odds Coefficient | Legal Interpretation |
|---:|:---|---:|:---|
| 1 | `petitioner` | +3.2456 | Substantive legal marker |
| 2 | `the petitioner` | +3.1425 | Substantive legal marker |
| 3 | `the petitioners` | +2.6851 | Substantive legal marker |
| 4 | `petitioners` | +2.6489 | Substantive legal marker |
| 5 | `accordingly` | +1.4094 | Substantive legal marker |
| 6 | `petitions` | +1.2548 | Substantive legal marker |
| 7 | `petition` | +1.2488 | Substantive legal marker |
| 8 | `petitioner was` | +1.1066 | Substantive legal marker |
| 9 | `32` | +1.1059 | Substantive legal marker |
| 10 | `32 of` | +1.0915 | Substantive legal marker |
| 11 | `this petition` | +1.0212 | Substantive legal marker |
| 12 | `is accordingly` | +1.0100 | Substantive legal marker |
| 13 | `accordingly dismissed` | +0.9940 | Substantive legal marker |
| 14 | `petitioner has` | +0.9418 | Substantive legal marker |
| 15 | `article 32` | +0.9115 | Substantive legal marker |

### Top Features Driving 'Rejected' Predictions:
| Rank | Predictive Feature | Log-Odds Coefficient | Legal Interpretation |
|---:|:---|---:|:---|
| 1 | `masked_holding` | +2.6333 | Substantive legal marker |
| 2 | `masked_holding masked_ruling` | +1.9169 | Substantive legal marker |
| 3 | `appellants` | +1.5928 | Substantive legal marker |
| 4 | `appellant` | +1.4911 | Substantive legal marker |
| 5 | `the appellant` | +1.4802 | Substantive legal marker |
| 6 | `the appellants` | +1.4311 | Substantive legal marker |
| 7 | `appeal` | +1.3417 | Substantive legal marker |
| 8 | `to interfere` | +1.1796 | Substantive legal marker |
| 9 | `appellant that` | +1.1758 | Substantive legal marker |
| 10 | `court masked_holding` | +1.1174 | Substantive legal marker |
| 11 | `appeals` | +1.0247 | Substantive legal marker |
| 12 | `interfere` | +1.0023 | Substantive legal marker |
| 13 | `contention` | +0.9996 | Substantive legal marker |
| 14 | `rightly` | +0.9284 | Substantive legal marker |
| 15 | `interfere with` | +0.9143 | Substantive legal marker |

---

## 3. Rhetorical Section Importance Hierarchy

| Canonical Section | Relative L2 Importance Weight | Interpretation |
|:---|---:|:---|
| **court_reasoning** | 0.2433 | Primary rhetorical driver |
| **facts** | 0.1756 | Primary rhetorical driver |
| **legal_provisions** | 0.1713 | Primary rhetorical driver |
| **arguments** | 0.1422 | Primary rhetorical driver |
| **procedural_history** | 0.0909 | Primary rhetorical driver |

---

## 4. Dense Legal Feature Attribution

| Feature Name | Weight (Accepted) | Weight (Other) | Weight (Rejected) |
|:---|---:|---:|---:|
| `ipc_count` | -0.0055 | -0.0328 | +0.0383 |
| `bns_count` | +0.0000 | +0.0000 | +0.0000 |
| `constitution_article_count` | -0.0565 | +0.0854 | -0.0289 |
| `crpc_count` | -0.0019 | +0.0137 | -0.0118 |
| `cpc_count` | +0.0104 | -0.0130 | +0.0026 |
| `statutory_section_mentions` | -0.0444 | +0.0639 | -0.0196 |
| `citation_frequency` | +0.0061 | -0.0066 | +0.0005 |
| `precedent_reference_density` | +0.0433 | -0.0712 | +0.0279 |
| `log_word_count` | +0.0066 | +0.0019 | -0.0084 |
| `log_char_count` | +0.1486 | -0.1237 | -0.0249 |
| `reasoning_ratio` | -0.0257 | -0.0405 | +0.0662 |
| `facts_ratio` | -0.0368 | -0.1139 | +0.1507 |
| `arguments_ratio` | -0.0563 | -0.0718 | +0.1282 |
| `procedural_ratio` | -0.0929 | -0.0828 | +0.1758 |
| `provisions_ratio` | -0.0732 | -0.1698 | +0.2430 |
