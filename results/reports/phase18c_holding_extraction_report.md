# Phase 18C: Holding Extraction Dataset Preparation

**Generated:** 2026-11-03 18:36:30

## Dataset Description

The holding extraction task is a secondary task complementary to outcome prediction.

- **Input:** Judgment text (proof_sentence will be masked during training)
- **Target:** Original proof_sentence (dispositive holding)
- **Purpose:** Extract the dispositive holding sentence from the judgment

## Partition Statistics

| Partition | Total | Accepted | Rejected | Groups |
|-----------|-------|----------|----------|--------|
| train | 24,849 | 11,722 | 13,127 | 24,262 |
| val | 5,320 | 2,432 | 2,888 | 5,199 |
| test | 5,351 | 2,530 | 2,821 | 5,200 |

## Notes

- Uses the same group-aware partitions as the binary outcome prediction task
- Prevents duplicate-group leakage across partitions
- Input text has proof_sentence available for masking during training
- Target is the original proof_sentence for supervised training

