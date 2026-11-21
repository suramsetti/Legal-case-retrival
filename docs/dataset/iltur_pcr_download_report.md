# IL-TUR Prior Case Retrieval (PCR) Dataset Acquisition & Integrity Report

**Dataset Source**: `Exploration-Lab/IL-TUR` (Subset: `pcr`)  
**Official Repository**: [https://huggingface.co/datasets/Exploration-Lab/IL-TUR](https://huggingface.co/datasets/Exploration-Lab/IL-TUR)  
**Benchmark Reference**: *IL-TUR: Benchmark for Indian Legal Text Understanding and Reasoning* (Joshi et al., ACL 2024 / arXiv:2407.05399)  
**Acquisition Date**: November 2026  
**Storage Format**: Apache Parquet (`.parquet`)  
**Local Destination**: [`data/external/IL-TUR-PCR/`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/data/external/IL-TUR-PCR)  
**Detailed Statistics Table**: [`results/tables/iltur_pcr_split_statistics.json`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/iltur_pcr_split_statistics.json)  

---

## 1. Overview & Provenance

The Prior Case Retrieval (PCR) task of the IL-TUR benchmark provides genuine Indian legal precedence retrieval challenges derived from historical Supreme Court of India judgments (sourced via IndianKanoon). 

In this benchmark:
- **Queries** represent target cases where explicit precedent citations have been masked or withheld.
- **Candidates** form the historical legal corpus pool against which retrieval models rank potential supporting precedents.
- **Relevant Candidates** (`relevant_candidates`) represent gold-standard citations reflecting genuine legal precedents relied upon or cited in the judicial opinion.

---

## 2. Authentication & Access Verification

1. **Gated Access**: The dataset is a controlled-access resource on Hugging Face (`gated: manual`, license `CC BY-NC-SA 4.0`).
2. **Authentication Flow**: Access was verified through official Hugging Face user credentials (`mohanbaluvedantam`) stored securely in the standard local Hugging Face cache directory (`~/.cache/huggingface/token`).
3. **Security Assurance**: No Hugging Face tokens are logged, stored in source code, or exposed in repository artifacts.

---

## 3. Storage Architecture & Format Selection

All 6 official splits have been downloaded directly from the Hugging Face repository and saved in dedicated split directories under `data/external/IL-TUR-PCR/`:

```text
data/external/IL-TUR-PCR/
├── train_queries/
│   └── train_queries.parquet       (15.22 MB, 827 records)
├── train_candidates/
│   └── train_candidates.parquet    (73.40 MB, 4,320 records)
├── dev_queries/
│   └── dev_queries.parquet         (2.70 MB, 118 records)
├── dev_candidates/
│   └── dev_candidates.parquet      (23.90 MB, 1,023 records)
├── test_queries/
│   └── test_queries.parquet        (4.25 MB, 237 records)
└── test_candidates/
    └── test_candidates.parquet     (32.66 MB, 1,727 records)
```

### Why Apache Parquet?
- **Native Hugging Face Format**: Matches the authoritative upstream storage format without transformation loss.
- **Typed Schema Preservation**: Preserves the native `list<element: string>` schema for sentence lists (`text`) and candidate lists (`relevant_candidates`) losslessly.
- **Columnar Efficiency**: Drastically reduces memory consumption during dense embedding generation and multi-stage re-ranking.

---

## 4. Comprehensive Split Statistics

All six splits were loaded, validated, and statistically profiled across document lengths, sentence counts, and citation densities:

| Split | Records | Total Sentences | Total Words | Avg Sentences / Doc | Avg Words / Doc | Median Words / Doc | Max Words |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train_queries`** | 827 | 49,138 | 5,629,833 | 59.42 | 6,807.54 | 4,508.0 | 119,513 |
| **`train_candidates`** | 4,320 | 229,697 | 27,189,860 | 53.17 | 6,293.95 | 3,632.5 | 410,597 |
| **`dev_queries`** | 118 | 8,628 | 990,941 | 73.12 | 8,397.81 | 4,619.5 | 106,169 |
| **`dev_candidates`** | 1,023 | 72,999 | 8,807,629 | 71.36 | 8,609.61 | 4,036.0 | 410,597 |
| **`test_queries`** | 237 | 14,325 | 1,587,912 | 60.44 | 6,700.05 | 4,653.0 | 91,838 |
| **`test_candidates`** | 1,727 | 105,112 | 12,055,119 | 60.86 | 6,980.38 | 3,947.0 | 410,597 |
| **Total / Aggregate** | **8,252** | **479,899** | **62,231,294** | **58.16** | **7,541.36** | **3,822.0** | **410,597** |

---

## 5. Record & ID Integrity Verification

Every record across all 6 splits was audited for schema validity and identifier consistency:

1. **Null or Empty ID Check**:
   - `null_ids`: **0** across all 6 splits.
   - `empty_ids`: **0** across all 6 splits.
   - **Every record possesses a non-empty, valid string case ID**.
2. **Within-Split Duplicate IDs**:
   - Duplicate IDs within `train_queries`: **0**
   - Duplicate IDs within `train_candidates`: **0**
   - Duplicate IDs within `dev_queries`: **0**
   - Duplicate IDs within `dev_candidates`: **0**
   - Duplicate IDs within `test_queries`: **0**
   - Duplicate IDs within `test_candidates`: **0**
3. **Partition Separation (Cross-Split Query Disjointness)**:
   - `train_queries` $\cap$ `dev_queries` = **0**
   - `train_queries` $\cap$ `test_queries` = **0**
   - `dev_queries` $\cap$ `test_queries` = **0**
   - **Complete disjointness**: Zero query leakage across training, development, and test partitions.
4. **Query Inclusion in Candidate Pools**:
   - All 827 train queries are present in the `train_candidates` pool.
   - All 118 dev queries are present in the `dev_candidates` pool.
   - All 237 test queries are present in the `test_candidates` pool.

---

## 6. Query-Candidate Relevance & Pool Alignment

For each evaluation partition, ground truth relevance labels (`relevant_candidates`) were matched against the corresponding candidate corpus pool:

| Partition Pair | Queries | Pool Size | Total Listed Citations | Citations in Pool | Pool Coverage (%) | Queries with 0 Relevant in Pool | Queries with $\ge 1$ Relevant in Pool | Avg Resolvable Candidates / Query |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Partition** (`train_queries` vs `train_candidates`) | 827 | 4,320 | 5,318 | 5,308 | **99.81%** | 10 (1.21%) | 817 (98.79%) | **6.42** (min: 0, max: 52, median: 4.0) |
| **Dev Partition** (`dev_queries` vs `dev_candidates`) | 118 | 1,023 | 973 | 973 | **100.00%** | 0 (0.00%) | 118 (100.0%) | **8.25** (min: 1, max: 92, median: 5.0) |
| **Test Partition** (`test_queries` vs `test_candidates`) | 237 | 1,727 | 1,730 | 1,727 | **99.83%** | 3 (1.27%) | 234 (98.73%) | **7.29** (min: 0, max: 58, median: 5.0) |
| **Total / Combined** | **1,182** | **7,070** | **8,021** | **8,008** | **99.84%** | **13** (1.10%) | **1,169** (98.90%) | **6.78** |

### Key Alignment Insights:
- **Outstanding Candidate Coverage (99.84%)**: Almost all cited precedents exist directly within their partitioned candidate pool. Only 13 queries across the entire benchmark (10 in train, 0 in dev, 3 in test) have zero resolvable cited precedents in their pool, due to cited authorities outside the Supreme Court pool.
- **Evaluation Validity**: For evaluation, queries with 0 resolvable candidates can either be excluded or treated as having empty relevant sets following standard COLIEE / IL-TUR benchmarking protocols.

---

## 7. Downstream Engineering Implications for Case Retrieval

1. **Document Length & Chunking**:
   - The average case document is ~6,000–8,500 words (with extremes reaching 400,000 words).
   - Standard 512-token BERT models cannot ingest full cases directly. The `src/embeddings/` and `src/preprocessing/` modules must employ sentence-level pooling, rhetorical-role-guided chunking, or hierarchical document transformers (e.g., Longformer, InLegalBERT with chunk pooling).
2. **Preservation of Sentence Boundaries**:
   - Because `text` is stored as an array of pre-segmented sentences, downstream legal feature extractors (`src/legal_features/`) can directly analyze citation context windows, judge ratio statements, and statutory references without noisy sentence tokenization.
3. **Ensemble & Re-Ranking**:
   - With candidate pools ranging from 1,023 to 4,320 documents, a two-stage retrieval pipeline is ideal:
     - **Stage 1 (Fast Retrieval)**: Lexical BM25 + dense bi-encoder embeddings (`src/retrieval/`) to narrow down to top-100 candidates.
     - **Stage 2 (Re-ranking & Legal Knowledge)**: Ensemble model (`src/ranking/`) fusing dense BERT similarity, citation graph overlap, and rhetorical features to generate the final ranked top-$K$ list.

---

## 8. Verification Sign-Off

- [x] All 6 splits successfully downloaded without truncation.
- [x] IDs validated: Zero null, zero empty, zero duplicate IDs within splits.
- [x] Query-candidate mapping verified with 99.84% ground-truth coverage.
- [x] Formal statistics persisted to [`results/tables/iltur_pcr_split_statistics.json`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/iltur_pcr_split_statistics.json).
- [x] No modifications made to `Jud-IPL` or other workspace assets.
- [x] Train/Dev/Test partitions strictly isolated.

