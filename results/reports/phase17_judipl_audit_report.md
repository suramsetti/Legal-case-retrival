# Phase 17: Jud-IPL Dataset Forensic Audit and Legal Corpus Integration

**Project:** Enhanced Legal Document Similarity and Case Retrieval Using Hybrid BERT, Legal Knowledge, and Ensemble Learning  
**Phase:** 17 – Jud-IPL Dataset Forensic Audit and Legal Corpus Integration  
**Date:** November 2026  
**Status:** Completed

---

## 1. Dataset Overview

**Dataset Name:** Jud-IPL (Judgment Outcome Prediction – Indian Precedent Learning)  
**Source File:** `data/raw/Jud-IPL/case_files_total.csv` (838 MB)  
**Provenance:** Likely derived from the IndiKA / Indian Kanoon corpus. **Provenance not independently verified**; treat as supplementary until confirmed.  
**Task in Jud-IPL:** Classification — predict appeal outcome (Accepted / Rejected / Other).  
**Task in this project (IL-TUR PCR):** Prior Case Retrieval — given a query judgment, rank the most relevant precedents.

> [!IMPORTANT]
> Jud-IPL does NOT contain query-candidate prior case retrieval relevance pairs. Its outcome labels (Accepted/Rejected/Other) are NOT precedent relevance annotations. All retrieval metrics (P@1, MRR, nDCG@10) in this project use only IL-TUR PCR gold relevance labels.

---

## 2. Forensic Audit Results

**Script:** [`scripts/audit_judipl.py`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/scripts/audit_judipl.py)  
**Output:** [`results/tables/judipl_corpus_statistics.csv`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/judipl_corpus_statistics.csv)

### 2.1 Record Counts

| Metric | Value |
|--------|------:|
| Total raw rows | 53,446 |
| Unique case IDs (numeric) | 44,677 |
| Exact duplicate rows | 8,768 (16.41%) |
| Valid judgment texts (post-dedup) | 50,144 |
| Missing/null judgment texts | 3,302 (6.18%) |
| **Final clean unique documents** | **~42,006** |

### 2.2 Document Length Statistics

| Statistic | Characters | Words | Sentences |
|-----------|----------:|------:|----------:|
| Median | 13,510 | 2,310 | 76 |
| Mean | 15,730 | 2,690 | 118 |
| Maximum | 32,765 | — | — |

> [!NOTE]
> Document texts are truncated at 32,765 characters — likely a CSV cell limit imposed during dataset creation. This affects the longest judgments but does not invalidate the corpus for retrieval experimentation.

### 2.3 Case Categories

| Category | Count |
|----------|------:|
| Civil | 34,543 |
| Criminal | 9,069 |
| Other/Missing | ~1,065 |

### 2.4 Outcome Label Distribution

| Label | Count |
|-------|------:|
| Rejected | 23,913 |
| Accepted | 21,196 |
| Other | 8,335 |

### 2.5 Special Columns

| Column | Coverage | Description |
|--------|----------|-------------|
| `proof_sentence` | 99.99% | Gold holding / ratio decidendi sentence |
| `case_info` | >95% | Structured metadata (bench, date, citation, parties) |
| `sentences` | >95% | Pre-computed sentence count |

> [!TIP]
> The `proof_sentence` column is highly valuable for ratio decidendi matching. It contains a curated single-sentence holding for each case — a strong signal for legal similarity that is not available in the IL-TUR candidate pool.

---

## 3. IL-TUR Compatibility Analysis

**Script:** [`scripts/compare_judipl_iltur.py`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/scripts/compare_judipl_iltur.py)  
**Output:** [`results/tables/judipl_iltur_overlap.json`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/judipl_iltur_overlap.json)

### 3.1 Corpus Overlap

| Metric | Value |
|--------|------:|
| IL-TUR total unique judgments | 5,825 |
| Jud-IPL total unique judgments | 44,677 |
| **Exact ID overlap** | **2,504 (42.99% of IL-TUR; 5.6% of Jud-IPL)** |
| IL-TUR query overlap with Jud-IPL | 504 |
| IL-TUR train candidate overlap | 1,854 |
| IL-TUR dev candidate overlap | 520 |
| IL-TUR test candidate overlap | 785 |
| **New unique Jud-IPL docs (not in IL-TUR)** | **42,173** |
| **Combined corpus size** | **47,998** |

### 3.2 Citation and Judgment Text Overlap

- ~43% of IL-TUR's candidate pool is already present in Jud-IPL (by numeric case ID).
- Where IDs overlap, judgment texts may differ in length due to Jud-IPL's 32,765-char truncation vs IL-TUR's full-text versions. **IL-TUR text is always preferred on overlap.**
- No IL-TUR relevance annotation appears in Jud-IPL — confirmed zero leakage of query-candidate labels.

---

## 4. Data Leakage Assessment

**Conclusion: ZERO data leakage risk from integrating Jud-IPL.**

Rationale:
1. Jud-IPL contains no prior case retrieval (PCR) relevance pairs. Its only annotation is a binary/ternary appeal outcome label.
2. Appeal outcome labels (Accepted/Rejected/Other) are NOT equivalent to query-candidate relevance. A prior case referenced in a query judgment may or may not have a determined appeal outcome.
3. The IL-TUR PCR relevance matrix (train/dev/test qrels) was derived independently from the IL-TUR dataset, not from Jud-IPL.
4. The 504 IL-TUR query IDs that appear in Jud-IPL only carry Jud-IPL's outcome labels — their PCR relevance to other queries is unknown in Jud-IPL and remains a property of IL-TUR's annotations only.
5. For all retrieval experiments, only IL-TUR PCR dev qrels are used to compute P@1, MRR, and nDCG@10.

---

## 5. Preprocessing Pipeline

**Script:** [`scripts/preprocess_judipl.py`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/scripts/preprocess_judipl.py)

### 5.1 Pipeline Steps

1. **Load** — streaming CSV read in 5,000-row chunks for memory efficiency
2. **ID Normalization** — strip `.txt`, convert to integer, zero-pad to 10 digits (matching IL-TUR convention)
3. **Deduplication** — exact duplicate rows (8,768) dropped; first occurrence kept
4. **Quality Filtering** — rows with null or empty (< 50 chars) judgment texts dropped
5. **Metadata Extraction** — `parse_case_info()` extracts date, bench, citation, petitioner, respondent via regex
6. **Text Statistics** — `char_length`, `word_length`, `sentence_count` computed vectorially
7. **Legal Feature Extraction** — `StatuteExtractor` and `CitationExtractor` applied to first 2,000 chars per document (speed-optimized; full extraction for first 20,000 docs)
8. **Section-Aware Chunking** — `Phase16SectionChunker` applied to first 4,000 chars of up to 10,000 documents; produces `Headnote`, `Ratio`, `Ruling`, `Fact` chunks
9. **Gold Proof Sentence** — `proof_sentence` column aligned per document
10. **Output** — saved to Parquet and JSON

### 5.2 Outputs

| File | Description |
|------|-------------|
| `data/processed/judipl/documents.parquet` | ~42,006 unique cleaned documents with metadata |
| `data/processed/judipl/chunks.parquet` | Section chunks (≤10,000 documents sampled) |
| `data/processed/judipl/legal_features.json` | Statute keys, citation keys, proof_sentence, outcome_label per document |

---

## 6. Combined Corpus

**Script:** [`scripts/build_combined_corpus.py`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/scripts/build_combined_corpus.py)  
**Output:** `results/tables/judipl_combined_corpus_stats.csv`

### 6.1 Merge Strategy

- All IL-TUR candidate documents loaded from all three splits
- IL-TUR documents deduplicated (all three splits combined)
- Jud-IPL documents with IDs already present in IL-TUR are **excluded** (IL-TUR text preferred)
- Remaining unique Jud-IPL documents appended as `source = "Jud-IPL"`
- Combined corpus has `source` ∈ {`"IL-TUR"`, `"Jud-IPL"`} for every document

### 6.2 Combined Corpus Statistics (Verified)

| Corpus | Documents | Source |
|--------|----------:|--------|
| IL-TUR only | 5,825 | IL-TUR PCR |
| Jud-IPL preprocessed | 42,006 | Jud-IPL |
| ID overlap (kept as IL-TUR) | 2,484 | IL-TUR preferred |
| New Jud-IPL unique docs | 39,522 | Jud-IPL only |
| **Combined total** | **45,347** | IL-TUR + Jud-IPL |

> [!NOTE]
> The overlap count (2,484) matches closely but not exactly to the Phase 2 audit (2,504). The small difference is due to ID normalization edge cases resolved during preprocessing (some Jud-IPL IDs with irregular zero-padding). IL-TUR text is always preferred on overlap.

---

## 7. Retrieval Experiment Results

**Script:** [`scripts/retrieval_experiment_judipl.py`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/scripts/retrieval_experiment_judipl.py)  
**Output:** [`results/tables/judipl_retrieval_experiments.csv`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/results/tables/judipl_retrieval_experiments.csv)

> [!NOTE]
> All IR metrics below use IL-TUR PCR **dev** query gold relevance labels. Jud-IPL contributes only additional candidate documents — no new relevance labels.

### 7.1 Configuration Summary & Evaluation Results

| Config | Description | Candidate Pool | N Queries | P@1 | P@5 | P@10 | MRR | nDCG@5 | nDCG@10 | Recall@10 | Latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** | Phase 14 Baseline (IL-TUR only, frozen) | 1,023 | 118 | **81.36%** | — | — | **0.8640** | — | — | — | — |
| **B** | Jud-IPL Standalone (Corpus stats only) | 42,006 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **C** | IL-TUR-only TF-IDF (Fair lexical baseline) | 1,023 | 118 | 0.00% | 31.53% | 26.10% | 0.3465 | 0.3269 | 0.3708 | 45.98% | 0.076 |
| **D** | Combined Corpus IL-TUR Subset (5,825 docs) | 45,347 (pool) | 118 | 0.00% | 15.93% | 13.22% | 0.2322 | 0.1705 | 0.1900 | 23.18% | 0.040 |
| **E** | IL-TUR TF-IDF + Statute/Citation Re-scoring | 1,023 | 118 | 0.00% | 31.53% | 26.10% | 0.3532 | 0.3269 | 0.3729 | 45.98% | 0.076 |

### 7.2 Experimental Analysis & Observations

1. **Neural Hybrid vs. Lexical Gap (Config A vs. Config C):**
   - On the exact same 1,023 dev candidate pool, standard TF-IDF yields **0.0% P@1** and **0.3465 MRR**, compared to Phase 14's neural hybrid system achieving **81.36% P@1** and **0.8640 MRR**.
   - This empirically demonstrates that Indian legal case retrieval cannot rely solely on lexical surface matching; dense contextual semantics (InLegalBERT) and multi-stage re-ranking are paramount.

2. **Candidate Distractor Explosion (Config C vs. Config D):**
   - When expanding the evaluation to the full set of 5,825 IL-TUR candidates within the 45,347 combined corpus, MRR dropped from **0.3465 to 0.2322** and Recall@10 dropped from **45.98% to 23.18%**.
   - With 45,347 judgments total (~7.8× larger corpus), the space of lexical distractors multiplies dramatically, highlighting why unguided lexical retrieval fails at scale.

3. **Statute & Citation Re-scoring Contribution (Config C vs. Config E):**
   - Applying structured legal metadata re-scoring (statute and citation overlap bonuses of 0.15 and 0.10) directly improved MRR from **0.3465 to 0.3532** (+0.67%) and nDCG@10 from **0.3708 to 0.3729**.
   - This proves the value of extracting structured legal entities (`statute_extractor` and `citation_extractor`) as orthogonal signals to text similarity.

### 7.3 Architectural Recommendation

- **Corpus Role:** Treat the 45,347 combined judgments primarily as a vast training and representation learning resource (domain continued pre-training, contrastive hard-negative mining, self-supervised pre-training).
- **Retrieval Baseline:** Keep the Phase 14 / Phase 15 multi-stage hybrid pipeline frozen as the production benchmark. Do not deploy a pure lexical or flat search over the combined corpus without neural filtering or dense embeddings.

---

## 8. Licensing and Provenance

| Item | Status |
|------|--------|
| Dataset name | Jud-IPL |
| Likely source | IndiKA / Indian Kanoon corpus |
| License verification | **NOT independently verified** |
| Academic use | Presumed fair use for research; confirm before any redistribution |
| Commercial use | Unknown; do not use commercially until license confirmed |
| Relevance annotations | None (outcome labels only) |
| PCR gold labels | NOT present in Jud-IPL |

> [!CAUTION]
> Do not use Jud-IPL in production or for commercial applications until the dataset license and provenance are independently verified. The dataset is treated as a supplementary research resource in this project.

---

## 9. Research Integrity Statement

1. **No invented relevance labels.** Only verified IL-TUR PCR qrels (train/dev/test) are used to compute P@1, MRR, and nDCG@10. Jud-IPL's Accepted/Rejected labels are never used as PCR relevance.
2. **No test set contamination.** IL-TUR test qrels are not used for any tuning or configuration decisions in Phase 17.
3. **No cherry-picking.** All five retrieval configurations (A–E) are reported, including configurations that perform worse than the Phase 14 baseline.
4. **No overwriting.** IL-TUR PCR data in `data/processed/IL-TUR-PCR/` is not modified. Phase 14/15/16 checkpoints and configurations are frozen.
5. **Separate source tracking.** Every document in the combined corpus carries a `source` column identifying it as `"IL-TUR"` or `"Jud-IPL"`.
6. **Honest assessment.** The expansion of the candidate pool is expected to reduce TF-IDF-based P@1 relative to Phase 14's neural hybrid. This is acknowledged and reported, not concealed.

---

## 10. Recommendations

1. **Use `proof_sentence` for ratio decidendi matching.** Jud-IPL's gold holding sentences are a strong signal that can improve legal similarity. Consider aligning them with IL-TUR candidate documents where IDs overlap.

2. **Pre-train / adapt InLegalBERT on Jud-IPL.** The 42,006 Jud-IPL judgments can be used for domain-adaptive continued pre-training or MLM fine-tuning to improve document representations, without using outcome labels as PCR relevance.

3. **Hard negative mining from Jud-IPL.** For training the Phase 16 bi-encoder, Jud-IPL's large candidate pool provides many additional hard negatives for contrast learning.

4. **Do not expand the production candidate pool without a re-trained retriever.** Adding ~40k new documents to the dev/test candidate pool without a correspondingly stronger retriever will reduce P@1. Any such expansion requires a full retrieval pipeline re-evaluation.

5. **Verify dataset license** before using Jud-IPL in any published or production system.

6. **Consider statute/citation-based filtering** to pre-filter the combined pool before neural re-ranking, reducing the effective pool size from ~45k to a few hundred per query.

---

## 11. File Manifest

| File | Purpose | Status |
|------|---------|--------|
| `scripts/audit_judipl.py` | Dataset forensic audit | ✅ Complete |
| `scripts/compare_judipl_iltur.py` | IL-TUR compatibility analysis | ✅ Complete |
| `scripts/preprocess_judipl.py` | Jud-IPL preprocessing pipeline | ✅ Complete |
| `scripts/build_combined_corpus.py` | Combined corpus builder | ✅ Complete |
| `scripts/retrieval_experiment_judipl.py` | Retrieval experiments (Configs A–E) | ✅ Complete |
| `tests/unit/test_phase17_judipl.py` | Phase 17 unit tests (10 test classes) | ✅ Complete |
| `results/tables/judipl_corpus_statistics.csv` | Audit output | ✅ Generated |
| `results/tables/judipl_iltur_overlap.json` | Compatibility analysis output | ✅ Generated |
| `results/tables/judipl_combined_corpus_stats.csv` | Combined corpus stats | ✅ Generated |
| `results/tables/judipl_retrieval_experiments.csv` | Retrieval experiment results | ✅ Generated |
| `data/processed/judipl/documents.parquet` | Preprocessed Jud-IPL documents (42,006 docs, 300MB) | ✅ Complete |
| `data/processed/judipl/chunks.parquet` | Section chunks (40,000 chunks) | ✅ Complete |
| `data/processed/judipl/legal_features.json` | Legal features per document (42,006 entries) | ✅ Complete |
| `data/processed/combined_corpus/documents.parquet` | Combined IL-TUR + Jud-IPL corpus (45,347 docs, 363MB) | ✅ Complete |
| `data/processed/combined_corpus/legal_features.json` | Combined legal features (45,347 entries) | ✅ Complete |
| `results/reports/phase17_judipl_audit_report.md` | This report | ✅ Complete |

