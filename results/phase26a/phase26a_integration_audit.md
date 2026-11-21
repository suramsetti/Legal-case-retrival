# Phase 26A IL-TUR PCR Integration Audit

## Existing implementation

- API route: `POST /api/search`
- Service: `RetrievalService`
- Candidate storage: `data/processed/IL-TUR-PCR/test_candidates/documents.parquet`
- Retrieval feature index: `data/processed/IL-TUR-PCR/test/hybrid_features.parquet`
- Frozen configuration artifact: `models/checkpoints/weighted_fusion_weights.json`
- Frozen fusion weights: `{'tfidf_score': 0.65, 'inlegalbert_sim': 0.25, 'composite_legal_score': 0.1, 'bm25_score': 0.0}`

The existing service uses its precomputed IL-TUR hybrid feature index for benchmark queries and its existing ad-hoc TF-IDF/legal-feature scoring path for free-text queries. No Jud-IPL service or checkpoint is accessed by this phase.

## End-to-end result

The valid legal query returned **5** ranked judgments through the unified Flask route. Candidate IDs, scores, ranks, and excerpts were returned from the existing IL-TUR candidate repository. Scores are genuine outputs of the existing service; no retrieval result was fabricated.

The benchmark query `0000021652` returned **3** results with self-match exclusion verified as `True`.

## Dataset integrity

Actual counts and integrity checks are recorded in `phase26a_dataset_integrity.json`. The repository contains 827 train queries, 118 dev queries, and 237 test queries—not 8,274, 1,181, and 2,371. Candidate counts are 4,320, 1,023, and 1,727 respectively.

## API validation

Empty, whitespace-only, invalid-type, invalid-`top_k`, and out-of-range `top_k` requests are recorded in `phase26a_api_test_results.json` with JSON error responses. Internal paths and stack traces are not returned by the route.

## Isolation

IL-TUR and Jud-IPL remain independent. This audit used only IL-TUR data, retrieval service state, and IL-TUR route calls.
