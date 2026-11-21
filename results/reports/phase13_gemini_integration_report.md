# Phase 13: Technical Report – Google Gemini API Integration for Indian Legal Case Retrieval

**Project Title:** Enhanced Legal Document Similarity and Case Retrieval Using Hybrid BERT, Legal Knowledge, and Ensemble Learning  
**Benchmark Split:** IL-TUR PCR (Indian Legal Text Understanding and Reasoning - Prior Case Retrieval)  
**Document Type:** Architecture & Implementation Technical Report  
**Date:** November 2026  
**Status:** Completed & Validated  

---

## 1. Executive Summary

Phase 13 integrates Google Gemini API into the Indian Legal Case Retrieval system. The core design principle of Phase 13 is **defensive architectural augmentation**: rather than replacing the established Stage-1 hybrid search and Stage-2 query-adaptive reranking pipeline with an opaque, uncalibrated generative model, Gemini is deployed as a modular **cognitive co-pilot and judicial precedent analyst**.

### Key Architectural Highlights:
1. **Preservation of Phase 11 Production Baseline:** The frozen Stage 1 + margin-gated neural reranker ($\tau=0.07$, $\alpha=0.12$, $\gamma=0.04$) serves as the deterministic anchor. When Gemini is toggled off or unavailable, the system executes 100% on the Phase 11 baseline with zero behavioral or accuracy regression.
2. **Four Specialized Cognitive Modules:**
   - **Module A (Legal Query Understanding & Expansion):** Transforms raw or lay legal descriptions into structured controversies, statutory provisions (with IPC/BNS cross-mapping), and Boolean search strings.
   - **Module B (Ratio Decidendi & Legal Reasoning Extractor):** Extracts judicial holdings, distinguishing binding *ratio decidendi* from *obiter dicta*, with grounded chunk pointer citations.
   - **Module C (Gemini Precedent Re-Ranking):** Evaluates candidate relevance across 5 legal dimensions (issue alignment, factual congruency, statutory overlap, ratio guidance, reasoning compatibility) for low-margin queries.
   - **Module D (Evidence-Based Grounded Explanations):** Generates transparent, citation-backed legal justifications with distinction analysis and mandatory Indian legal practice disclaimers.
3. **Enterprise Resilience & Quota Engineering:**
   - Disk-backed deterministic MD5 request caching (`data/cache/gemini/`) preventing duplicate billing and latency spikes.
   - Automatic HTTP 429 rate limit backoff parsing exact `retry_delay` seconds.
   - Strict JSON-schema validation and sanitized Markdown code block stripping.
   - Zero quota consumption during automated unit and integration tests via comprehensive mocking fixtures.

---

## 2. Integrated Architecture Overview

```
                      +---------------------------------------+
                      |   Client Search Query (Text or ID)   |
                      +---------------------------------------+
                                          |
                                          v
         +-----------------------------------------------------------------+
         |      Module A: Legal Query Understanding & Expansion (Gemini)   |
         |  - Structured Legal Issue Identification                        |
         |  - Statutory Mapping (e.g., Sec 302 IPC / Sec 101 BNS)         |
         |  - Contextual Legal Concepts & Synonyms                         |
         +-----------------------------------------------------------------+
                                          |
                                          v
         +-----------------------------------------------------------------+
         |            Stage 1: Multi-Modal Hybrid Candidate Pool           |
         |  - TF-IDF Lexical Retrieval (Weight: 0.65)                      |
         |  - InLegalBERT Dense Semantic Embeddings (Weight: 0.25)        |
         |  - Composite Legal Knowledge Features (Weight: 0.10)            |
         |  - Self-exclusion guarantee (query excluded from candidates)    |
         +-----------------------------------------------------------------+
                                          |
                                          v
         +-----------------------------------------------------------------+
         |       Margin Calculation: margin(q) = S_1(top1) - S_1(top2)     |
         +-----------------------------------------------------------------+
                         /                                 \
           margin >= tau (0.07)               margin < tau (0.07)
         [High Confidence Query]             [Low Confidence / Ambiguous]
                 |                                         |
                 v                                         v
         +--------------------+              +-----------------------------+
         | Preserve Stage 1   |              | Apply Margin-Gated Reranker |
         | Ordering Untouched |              | (Phase 11 Neural or Gemini) |
         +--------------------+              +-----------------------------+
                 \                                         /
                  \                                       /
                   v                                     v
         +-----------------------------------------------------------------+
         |    Module B & D: Precedent Deep Analysis & Evidence Grounding   |
         |  - Grounded Ratio Decidendi Extraction (Pointers to Chunks)     |
         |  - 5-Dimension Precedent Alignment Breakdown                   |
         |  - Divergence / Distinction Analysis                            |
         |  - Statutory Verification & Legal Disclaimers                   |
         +-----------------------------------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |    Interactive Single-Page UI (React) |
                      |    and REST API (/api/search)         |
                      +---------------------------------------+
```

---

## 3. Module Specifications & Implementations

### 3.1 Module A: Legal Query Expansion (`backend/services/legal_query_expansion.py`)
- **System Prompt Design:** Configured as a judicial research clerk for the Supreme Court of India. Extracts framed legal issues, legal taxonomy concepts, relevant enactments, and boolean search query variants.
- **IPC to BNS Transition Awareness:** Prompts specifically instruct Gemini to distinguish legacy Indian Penal Code (IPC 1860) sections from Bharatiya Nyaya Sanhita (BNS 2023) sections to maintain statutory currency.
- **Fail-Safe Fallback:** If Gemini API is offline or returns invalid JSON, `fallback_expansion()` generates standard lexical queries from the input string with zero interruption to downstream search.

### 3.2 Module B: Ratio Decidendi Extractor (`backend/services/legal_reasoning_extractor.py`)
- **Doctrinal Distinction:** Explicitly guides the model to differentiate binding judicial principles (*ratio decidendi*) from incidental judicial observations (*obiter dicta*).
- **Grounded Verification:** Requires each extracted point of law to cite `source_chunk_id` and include exact verbatim `quoted_text` from the candidate judgment chunks.
- **Fallback Guarantee:** Falls back to extracting prominent opening paragraphs and legal features if generative extraction is unavailable.

### 3.3 Module C: Precedent Re-Ranker (`backend/services/gemini_precedent_reranker.py`)
- **Multi-Dimensional Relevance Scoring:**
  1. `legal_issue_alignment` (0.0 - 1.0)
  2. `factual_alignment` (0.0 - 1.0)
  3. `statutory_alignment` (0.0 - 1.0)
  4. `ratio_alignment` (0.0 - 1.0)
  5. `reasoning_alignment` (0.0 - 1.0)
- **Candidate Pool Integrity:** Restricted strictly to Top-5 Stage-1 candidates for low-margin queries ($\text{margin} < 0.07$). High-margin queries retain Stage-1 ordering untouched. Un-retrieved cases are never injected.
- **Linear Fusion:** $\text{Score}_{\text{final}} = (1 - w_g) \cdot \text{Score}_{\text{Stage1}} + w_g \cdot \text{Score}_{\text{Gemini}}$ (default $w_g=0.25$).

### 3.4 Module D: Evidence-Based Legal Explanation (`backend/services/legal_explanation_service.py`)
- **Structured Explanations:** Emits structured 8-field analysis including summary of relevance, key matching facts, statutory overlap, ratio application, contradictory evidence / points of distinction, confidence score, and operational disclaimers.
- **Strict Grounding:** Emphasizes factual divergences (e.g., different recovery circumstances, lack of sanction under Section 197 CrPC) to prevent judicial hallucination.

---

## 4. API Endpoints & Interfaces

The Flask backend (`backend/api/routes.py`) provides the following Phase 13 endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/api/gemini/status` | `GET` | Health check for Gemini API configuration, active model, and cache status |
| `/api/gemini/expand-query` | `POST` | Executes Module A query expansion on raw user input |
| `/api/gemini/extract-reasoning/<case_id>` | `GET` | Extracts ratio decidendi and obiter dicta for a candidate judgment |
| `/api/gemini/explain/<query_id>/<case_id>` | `POST` | Generates 5-dimension grounded explanation between query and candidate |
| `/api/gemini/metrics` | `GET` | Live telemetry: total requests, cache hits, token usage, and latency |
| `/api/search` | `POST` | Enhanced unified search with `enable_gemini`, `enable_expansion`, `enable_reranking` toggles |

---

## 5. Test Suite & Reliability

All 116 tests across the repository pass cleanly:
- `tests/unit/test_gemini_service.py`: 8 tests covering config validation, retry logic, caching, JSON parsing, error handling.
- `tests/unit/test_legal_query_expansion.py`: 3 tests covering query expansion schema, fallback, and validation.
- `tests/unit/test_legal_reasoning_extractor.py`: 3 tests covering ratio/obiter distinction, grounded citations, and fallback.
- `tests/unit/test_gemini_reranker.py`: 4 tests covering multi-dimensional scoring, candidate pool integrity, margin preservation, and fallback.
- `tests/integration/test_phase13_pipeline.py`: 5 tests covering all REST endpoints, pure Phase 11 baseline preservation, and end-to-end mocked execution.
- Existing units (Phase 1–12): 93 unit tests covering indexing, vectorization, LightGBM ranker, Pairwise neural ranker, and adaptive margin gating.
- **Total Test Result:** `111 passed unit tests`, `5 passed integration tests`, `0 failed`.

