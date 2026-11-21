# Phase 20A: Three-Class Dataset Forensic Audit

**Audit Timestamp:** 2026-11-03 23:05:00  
**Target Corpus:** Jud-IPL Three-Class Partitions ($N = 42,005$)  
**Status:** Completed, Verified, Cryptographically Audited  

---

## 1. Executive Summary

The Jud-IPL corpus contains 42,006 judgments. After accounting for 1 corrupt record, all **42,005 judgments** are partitioned into the three-class task:
- **Train:** 29,395 (70.0%)
- **Validation:** 6,297 (15.0%)
- **Test:** 6,313 (15.0%)

### Critical Methodological Findings:
1. **Zero Partition Leakage:** 0 overlapping document IDs, 0 raw IDs, and 0 duplicate groups between Train, Val, and Test.
2. **Prior-Exposure Disclosure:** **812 documents** (12.86%) of the Three-Class Test partition originate from the Phase 18 binary test partition. Therefore, the Three-Class Test partition **cannot be characterized as a completely virgin blind benchmark**; all evaluations on it must include this formal disclaimer.
3. **Masking Coverage:** Dispositive ruling phrases and proof sentences are neutralized across all three partitions.
4. **Class Imbalance:** The corpus is moderately imbalanced: Rejected (44.8%), Accepted (39.7%), and Other (15.4%).

---

## 2. Three-Class Partition Distribution Table

| Partition | Label | Cases ($N$) | Proportion of Partition | Total Partition Size |
|:---|:---|---:|---:|---:|
| **Train** | Rejected | 13,188 | 44.86% | 29,395 |
| **Train** | Accepted | 11,669 | 39.70% | 29,395 |
| **Train** | Other | 4,538 | 15.44% | 29,395 |
| **Validation** | Rejected | 2,775 | 44.07% | 6,297 |
| **Validation** | Accepted | 2,555 | 40.57% | 6,297 |
| **Validation** | Other | 967 | 15.36% | 6,297 |
| **Test** | Rejected | 2,873 | 45.51% | 6,313 |
| **Test** | Accepted | 2,460 | 38.97% | 6,313 |
| **Test** | Other | 980 | 15.52% | 6,313 |

---

## 3. Partition Isolation Audit

| Partition Pair | Overlapping Document IDs | Overlapping Duplicate Groups | Isolation Status |
|:---|---:|---:|:---|---:|
| Train vs Validation | 0 | 0 | **VERIFIED ISOLATED** |
| Train vs Test | 0 | 0 | **VERIFIED ISOLATED** |
| Validation vs Test | 0 | 0 | **VERIFIED ISOLATED** |

---

## 4. Prior-Exposure Audit

- Phase 18 Binary Test Size: **5,351**
- Phase 20 Three-Class Test Size: **6,313**
- Number of Shared Cases: **812**
- Prior-Exposure Proportion: **12.86%**
- **Methodological Rule:** This partition cannot be claimed as an untouched independent held-out benchmark. Results will be labeled strictly as *Held-Out Partition with Documented Prior Exposure*.
