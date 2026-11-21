# Phase 18C: Jud-IPL Dataset Preparation Report

**Generated:** 2026-11-03 17:21:27

## 1. Dataset Inventory

- **Total Documents:** 42,006
- **Expected Count:** 42,006
- **Verification:** [PASS]

## 2. Label Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| Rejected | 18,836 | 44.84% |
| Accepted | 16,684 | 39.72% |
| Other | 6,485 | 15.44% |
|  | 1 | 0.00% |

- **Binary-labeled (Accepted + Rejected):** 35,520
- **Other cases:** 6,485
- **Unlabeled:** 1

## 3. Duplicate Analysis

- **Exact full_text duplicates:** 1,028 (2.45%)
- **Documents in duplicate groups:** 2,041

## 4. Date Coverage

- **Records with judgment date:** 16,845 (40.1%)
- **Records missing date:** 25,161 (59.9%)
- **Unique date values:** 6,319

## 5. Proof Sentence Coverage

- **Records with proof_sentence:** 42,005 (100.00%)

## 6. Text Statistics

- **Character length:** median=13504, mean=15727.8
- **Word count:** median=2309, mean=2688.2
- **Sentence count:** median=82, mean=125.7

## 7. Notes

- Only 40.1% of documents have known judgment dates, limiting temporal split options.
- 1,028 exact duplicate documents must be kept together in same partition.
- All 42,006 documents have proof_sentence annotations.
- One unlabeled record (empty string label) requires exclusion from supervised tasks.

