# Legal Document Preprocessing and Sentence-Aware Chunking Report

**Project**: Enhanced Legal Document Similarity and Case Retrieval Using Hybrid BERT, Legal Knowledge, and Ensemble Learning  
**Dataset**: `Exploration-Lab/IL-TUR` (Configuration: `pcr`)  
**Processed Storage**: [`data/processed/IL-TUR-PCR/`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/data/processed/IL-TUR-PCR)  
**Summary Statistics**: [`results/tables/preprocessing_summary.json`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/preprocessing_summary.json)  
**Distribution Figures**: [`results/figures/chunking_distributions.png`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/figures/chunking_distributions.png)  
**Demonstration Notebook**: [`notebooks/02_text_preprocessing.ipynb`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/notebooks/02_text_preprocessing.ipynb)  
**Date**: November 2026  

---

## 1. Executive Summary

Phase 2 establishes a reproducible, deterministic preprocessing and chunking pipeline tailored specifically for Indian legal texts. 

### Core Challenges in Legal Document Processing
1. **Extreme Document Length**: Indian Supreme Court judgments average ~6,000–8,500 words (~50–75 sentences), with extreme documents exceeding 100,000 words. Because standard transformer models (such as BERT or Legal-BERT) enforce a strict 512-token sequence limit, documents cannot be ingested as single sequences without chunking.
2. **Domain-Specific Entity Preservation**: Legal judgments depend critically on statutory sections (e.g., `Section 302 IPC`, `Article 21`), precedent reporter citations (e.g., `AIR 1965 SC 123`, `(2008) 4 SCC 594`), and case identifiers (`Criminal Appeal No. 1234 of 2018`). Naive text cleaners frequently destroy or corrupt these entities.
3. **Retrieval Integrity & Boundary Preservation**: Chunks must respect judicial sentence boundaries to preserve rhetorical context and legal ratios. Furthermore, queries and candidate pools across `train`, `dev`, and `test` partitions must remain strictly isolated.

---

## 2. Pipeline Architecture & Modular Components

The pipeline is organized into four production modules under `src/preprocessing/`:

### A. [`LegalTextCleaner`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/src/preprocessing/text_cleaner.py)
- **Unicode Normalization**: Applies `NFKC` normalization to standardize typographic variants while retaining essential character semantics.
- **Whitespace & Invisible Character Sanitization**: Replaces non-breaking spaces (`\u00a0`), thin spaces, zero-width spaces (`\u200b`), and non-standard spacing with standard ASCII space.
- **Control Character Stripping**: Removes non-printable control characters (`\x00`–`\x1f`, except standard newlines and tabs) that often appear in raw scanned PDF conversions.
- **Quote & Dash Standardization**: Converts curly quotes (`“`, `”`, `‘`, `’`) to standard quotes and em/en dashes (`—`, `–`) to hyphens without altering hyphenated section numbers (e.g., `498-A`).
- **Entity & Stopword Preservation**: Strictly retains all statutory sections, case citations, dates, party indicators (`v.`, `vs.`, `Ors.`, `Anr.`), and legal stopwords.

### B. [`DocumentReconstructor`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/src/preprocessing/document_reconstructor.py)
- **Sentence Ordering**: Assembles ordered sentence arrays (`text` field) into cohesive judicial texts without altering judicial chronology.
- **Safe Null Handling**: Gracefully strips null entries, whitespace-only artifacts, and unprintable byte noise without hallucinating missing text.
- **Metric Extraction**: Computes clean sentence count, word count, character count, and tracking of empty sentences removed.

### C. [`SentenceAwareChunker`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/src/preprocessing/chunker.py)
- **Sentence Boundary Preservation**: Assembles chunks using complete sentences. Sentences are never bisected unless a single sentence exceeds the token budget.
- **Token Budget & Special Token Reservation**: Operates with configurable `max_tokens=512` and reserves space (`special_tokens_reserve=2`) for transformer `[CLS]` and `[SEP]` tokens (effective budget = 510 tokens).
- **Configurable Overlap**: Implements backward sentence traversal targeting 64 tokens of overlap to maintain semantic continuity across chunk boundaries.
- **Metadata Tracking**: Every chunk retains `doc_id`, `chunk_id` (`{doc_id}_chk_{idx:04d}`), `chunk_idx`, `start_sentence_idx`, `end_sentence_idx`, and exact `token_count`.
- **Oversized Sentence Fallback**: Sentences exceeding 510 tokens are chunked using word-level sliding windows to maintain natural language boundaries without truncation.

### D. [`PreprocessingPipeline`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/src/preprocessing/preprocessing_pipeline.py)
- **Batch Processing**: Ingests all 6 splits from `data/external/IL-TUR-PCR/` and writes processed datasets:
  - `documents.parquet`: Document-level full text, clean sentence lists, and retrieval metadata.
  - `chunks.parquet`: Chunk-level token-bounded segments for dense embedding indexing.
- **Retrieval Integrity & Flags**: Cross-checks `relevant_candidates` against candidate pools and sets boolean `has_zero_relevant_in_pool` flags for queries.

---

## 3. Recommended Chunking Configuration

Based on empirical token-length distributions across 479,000 sentences from 8,252 Supreme Court cases:

| Parameter | Recommended Value | Engineering Justification |
| :--- | :---: | :--- |
| **Max Tokens** (`max_tokens`) | **512** | Standard maximum positional capacity for BERT, Legal-BERT, and InLegalBERT. |
| **Overlap Tokens** (`overlap_tokens`) | **64** | ~1–2 sentences of context window overlap; prevents boundary-straddling ratio statements from being missed. |
| **Special Token Reserve** | **2** | Guarantees room for `[CLS]` and `[SEP]` without requiring runtime truncation. |
| **Tokenizer** | `bert-base-uncased` | Exact alignment with downstream transformer vocabulary (cached locally in `models/bert/tokenizer.json`). |
| **Aggregation Strategy** | **Max-Pooling / Mean-Pooling over Top-K Chunks** | For document-level retrieval, max-sim or score aggregation across top-ranked chunks yields the best precedent matching accuracy. |

---

## 4. Retrieval Integrity & Safeguards

1. **Partition Isolation**:
   - `train_queries` $\cap$ `dev_queries` = **0**
   - `train_queries` $\cap$ `test_queries` = **0**
   - `dev_queries` $\cap$ `test_queries` = **0**
   - Strictly preserved across both `documents.parquet` and `chunks.parquet`.
2. **Evaluation Safeguard (Query Self-ID Exclusion)**:
   - All query judgments exist within their candidate pools.
   - During future evaluation (`src/evaluation/`), the retrieval engine must explicitly mask or drop `query_id == candidate_id` from the top-$K$ rankings to prevent trivial self-retrieval.
   - The query ID is preserved intact in the dataset itself for structural completeness.
3. **Zero-Relevant Query Flagging**:
   - Queries with zero resolvable precedents in their candidate pool (10 in train, 0 in dev, 3 in test) are flagged with `has_zero_relevant_in_pool: True` in `documents.parquet`.

---

## 5. Automated Unit Test Verification

The unit test suite in [`tests/unit/test_preprocessing.py`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/tests/unit/test_preprocessing.py) validates all core components:

| Test Case | Description | Result |
| :--- | :--- | :---: |
| `test_unicode_and_whitespace_normalization` | Verifies non-breaking spaces, zero-width characters, and control characters are cleanly stripped | **PASSED** |
| `test_legal_citation_preservation` | Confirms AIR, SCC, SCR citations, statutory sections (IPC, CPC), and case numbers are preserved | **PASSED** |
| `test_document_reconstruction_preserves_order` | Validates sentence order preservation and accurate full-text assembly | **PASSED** |
| `test_empty_and_malformed_document_reconstruction`| Checks graceful handling of nulls, empty strings, and empty documents | **PASSED** |
| `test_chunking_ordering_and_id_preservation` | Asserts chunk indices, chunk IDs, sentence spans, and token limits | **PASSED** |
| `test_chunk_overlap` | Verifies consecutive chunks share overlapping sentence context | **PASSED** |
| `test_split_isolation_logic` | Asserts disjoint sets across training, development, and test partitions | **PASSED** |

**Test Execution**: `Ran 7 tests in 0.468s - OK`

---

## 6. Empirical Processing & Chunking Statistics

The complete preprocessing and chunking pipeline processed all 6 partitions (8,252 documents) and yielded 265,592 sentence-bounded chunks.

### A. Partition-Level Statistics Summary

| Split | Documents | Total Chunks | Avg Chunks/Doc | Median Chunks/Doc | Max Chunks/Doc | Avg Clean Words | Median Words | Max Words | Avg Sentences | Extremely Long Docs (>10k Words) | Zero-Rel in Pool |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train_queries`** | 827 | 26,663 | 32.24 | 21.0 | 555 | 6,807.48 | 4,508.0 | 119,513 | 58.86 | 131 | 10 |
| **`train_candidates`**| 4,320 | 128,297 | 29.70 | 17.0 | 1,842 | 6,293.89 | 3,632.5 | 410,597 | 52.81 | 596 | N/A |
| **`dev_queries`** | 118 | 4,760 | 40.34 | 22.0 | 495 | 8,398.24 | 4,619.5 | 106,169 | 72.85 | 20 | 0 |
| **`dev_candidates`** | 1,023 | 41,376 | 40.45 | 19.0 | 1,842 | 8,609.63 | 4,036.0 | 410,597 | 70.97 | 170 | N/A |
| **`test_queries`** | 237 | 7,541 | 31.82 | 22.0 | 451 | 6,699.88 | 4,653.0 | 91,838 | 59.37 | 42 | 3 |
| **`test_candidates`**| 1,727 | 56,955 | 32.98 | 19.0 | 1,842 | 6,980.26 | 3,950.0 | 410,597 | 60.19 | 275 | N/A |
| **TOTAL / OVERALL** | **8,252** | **265,592** | **32.19** | **18.0** | **1,842** | **6,825.21** | **3,824.0** | **410,597** | **57.44** | **1,234** | **13** |

*Note: Zero empty documents were detected across all splits.*

### B. Global Chunk Token Distribution (`bert-base-uncased`)

Across all 265,592 generated chunks:
- **Mean Token Length**: **367.9** tokens
- **Median Token Length**: **420.0** tokens
- **25th Percentile ($P_{25}$)**: **285.0** tokens
- **75th Percentile ($P_{75}$)**: **474.0** tokens
- **95th Percentile ($P_{95}$)**: **507.0** tokens
- **Configured Upper Bound**: 512 tokens (510 budget + 2 reserved special tokens `[CLS]`/`[SEP]`)

### C. Generated Artifacts & File Inventory

```
legal-case-retrieval/
|-- data/
|   `-- processed/
|       `-- IL-TUR-PCR/
|           |-- train_queries/
|           |   |-- documents.parquet       (29.74 MB, 827 docs)
|           |   `-- chunks.parquet          (16.48 MB, 26,663 chunks)
|           |-- train_candidates/
|           |   |-- documents.parquet       (143.75 MB, 4,320 docs)
|           |   `-- chunks.parquet          (79.31 MB, 128,297 chunks)
|           |-- dev_queries/
|           |   |-- documents.parquet       (5.27 MB, 118 docs)
|           |   `-- chunks.parquet          (2.94 MB, 4,760 chunks)
|           |-- dev_candidates/
|           |   |-- documents.parquet       (46.79 MB, 1,023 docs)
|           |   `-- chunks.parquet          (25.90 MB, 41,376 chunks)
|           |-- test_queries/
|           |   |-- documents.parquet       (8.32 MB, 237 docs)
|           |   `-- chunks.parquet          (4.60 MB, 7,541 chunks)
|           `-- test_candidates/
|               |-- documents.parquet       (63.94 MB, 1,727 docs)
|               `-- chunks.parquet          (35.41 MB, 56,955 chunks)
|-- models/
|   `-- bert/
|       `-- tokenizer.json                 (466 KB, bert-base-uncased)
|-- notebooks/
|   `-- 02_text_preprocessing.ipynb        (Interactive demonstration)
|-- results/
|   |-- figures/
|   |   `-- chunking_distributions.png     (4-panel distribution visualization)
|   `-- tables/
|       `-- preprocessing_summary.json     (Complete empirical statistics)
|-- src/
|   `-- preprocessing/
|       |-- __init__.py
|       |-- text_cleaner.py                (Unicode, whitespace, legal preservation)
|       |-- document_reconstructor.py      (Sentence ordering and metadata assembly)
|       |-- chunker.py                     (Sentence-aware sliding window chunker)
|       `-- preprocessing_pipeline.py      (End-to-end multi-split batch processor)
`-- tests/
    `-- unit/
        `-- test_preprocessing.py          (7 passed unit tests)
```

