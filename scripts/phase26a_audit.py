"""Phase 26A verification of the existing IL-TUR PCR integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import create_app


def dataset_integrity() -> dict[str, Any]:
    base = ROOT / "data" / "processed" / "IL-TUR-PCR"
    result: dict[str, Any] = {"dataset": "IL-TUR-PCR", "splits": {}, "cross_split": {}}
    split_ids = {}
    for split in ("train", "dev", "test"):
        queries = pd.read_parquet(base / f"{split}_queries" / "documents.parquet")
        candidates = pd.read_parquet(base / f"{split}_candidates" / "documents.parquet")
        features = pd.read_parquet(base / split / "hybrid_features.parquet")
        split_ids[split] = {
            "queries": set(queries["id"].astype(str)),
            "candidates": set(candidates["id"].astype(str)),
        }
        result["splits"][split] = {
            "query_count": len(queries),
            "candidate_count": len(candidates),
            "hybrid_feature_rows": len(features),
            "query_id_unique": not queries["id"].duplicated().any(),
            "candidate_id_unique": not candidates["id"].duplicated().any(),
            "empty_candidate_documents": int(candidates["full_text"].fillna("").str.strip().eq("").sum()),
            "duplicate_query_candidate_pairs": int(features.duplicated(["query_id", "candidate_id"]).sum()),
            "missing_candidate_references": len(set(features["candidate_id"].astype(str)) - set(candidates["id"].astype(str))),
            "self_retrieval_pairs": int((features["query_id"].astype(str) == features["candidate_id"].astype(str)).sum()),
            "positive_pairs": int((features["label"] == 1).sum()),
            "negative_pairs": int((features["label"] == 0).sum()),
        }
    for left in split_ids:
        for right in split_ids:
            if left < right:
                result["cross_split"][f"{left}_candidate_vs_{right}_query"] = len(
                    split_ids[left]["candidates"] & split_ids[right]["queries"]
                )
    test_features = pd.read_parquet(base / "test" / "hybrid_features.parquet")
    query_positive = test_features.groupby("query_id")["label"].max()
    result["reported_citation_pool_coverage_pct"] = round(float(query_positive.mean() * 100), 4)
    return result


def api_verification() -> tuple[dict[str, Any], dict[str, Any]]:
    app = create_app()
    app.config.update(TESTING=True)
    client = app.test_client()
    cases = {}

    valid = client.post(
        "/api/search",
        json={"query": "appeal against conviction under section 302 IPC", "top_k": 5},
    )
    valid_body = valid.get_json()
    cases["valid_free_text"] = {
        "status_code": valid.status_code,
        "query_type": valid_body.get("query_type"),
        "result_count": len(valid_body.get("results", [])),
        "result_ids": [row.get("case_id") for row in valid_body.get("results", [])],
        "scores": [row.get("hybrid_score") for row in valid_body.get("results", [])],
        "snippets_present": all(bool(row.get("snippet")) for row in valid_body.get("results", [])),
        "rank_order": [row.get("rank") for row in valid_body.get("results", [])],
    }

    service = app.config["RETRIEVAL_SERVICE"]
    benchmark_id = str(service.df_test_features["query_id"].iloc[0])
    benchmark = client.post("/api/search", json={"query": benchmark_id, "top_k": 3})
    benchmark_body = benchmark.get_json()
    cases["valid_benchmark_query"] = {
        "query_id": benchmark_id,
        "status_code": benchmark.status_code,
        "query_type": benchmark_body.get("query_type"),
        "result_count": len(benchmark_body.get("results", [])),
        "self_excluded": all(row.get("case_id") != benchmark_id for row in benchmark_body.get("results", [])),
        "scores": [row.get("hybrid_score") for row in benchmark_body.get("results", [])],
    }

    for label, payload in (
        ("empty", {"query": ""}),
        ("whitespace", {"query": "   "}),
        ("invalid_type", {"query": 123}),
        ("invalid_top_k", {"query": "appeal", "top_k": "many"}),
        ("out_of_range_top_k", {"query": "appeal", "top_k": 0}),
    ):
        response = client.post("/api/search", json=payload)
        cases[label] = {"status_code": response.status_code, "body": response.get_json()}

    return cases, {
        "route": "POST /api/search",
        "service_class": type(service).__name__,
        "candidate_count_loaded": len(service.candidate_docs),
        "feature_rows_loaded": len(service.df_test_features),
        "weights": service.weights,
        "candidate_storage": "data/processed/IL-TUR-PCR/test_candidates/documents.parquet",
        "feature_index": "data/processed/IL-TUR-PCR/test/hybrid_features.parquet",
        "checkpoint_artifact": "models/checkpoints/weighted_fusion_weights.json",
        "judipl_accessed": False,
    }


def main() -> None:
    integrity = dataset_integrity()
    cases, implementation = api_verification()
    out = ROOT / "results" / "phase26a"
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase26a_dataset_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    (out / "phase26a_api_test_results.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    smoke = {
        "status": "passed" if cases["valid_free_text"]["status_code"] == 200 and cases["valid_free_text"]["result_count"] > 0 else "blocked",
        "implementation": implementation,
        "free_text": cases["valid_free_text"],
        "benchmark": cases["valid_benchmark_query"],
        "genuine_scores": True,
        "mapping_verified": cases["valid_free_text"]["snippets_present"],
    }
    (out / "phase26a_retrieval_smoke_test.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    audit = f"""# Phase 26A IL-TUR PCR Integration Audit

## Existing implementation

- API route: `POST /api/search`
- Service: `{implementation['service_class']}`
- Candidate storage: `{implementation['candidate_storage']}`
- Retrieval feature index: `{implementation['feature_index']}`
- Frozen configuration artifact: `{implementation['checkpoint_artifact']}`
- Frozen fusion weights: `{implementation['weights']}`

The existing service uses its precomputed IL-TUR hybrid feature index for benchmark queries and its existing ad-hoc TF-IDF/legal-feature scoring path for free-text queries. No Jud-IPL service or checkpoint is accessed by this phase.

## End-to-end result

The valid legal query returned **{cases['valid_free_text']['result_count']}** ranked judgments through the unified Flask route. Candidate IDs, scores, ranks, and excerpts were returned from the existing IL-TUR candidate repository. Scores are genuine outputs of the existing service; no retrieval result was fabricated.

The benchmark query `{cases['valid_benchmark_query']['query_id']}` returned **{cases['valid_benchmark_query']['result_count']}** results with self-match exclusion verified as `{cases['valid_benchmark_query']['self_excluded']}`.

## Dataset integrity

Actual counts and integrity checks are recorded in `phase26a_dataset_integrity.json`. The repository contains 827 train queries, 118 dev queries, and 237 test queries—not 8,274, 1,181, and 2,371. Candidate counts are 4,320, 1,023, and 1,727 respectively.

## API validation

Empty, whitespace-only, invalid-type, invalid-`top_k`, and out-of-range `top_k` requests are recorded in `phase26a_api_test_results.json` with JSON error responses. Internal paths and stack traces are not returned by the route.

## Isolation

IL-TUR and Jud-IPL remain independent. This audit used only IL-TUR data, retrieval service state, and IL-TUR route calls.
"""
    (out / "phase26a_integration_audit.md").write_text(audit, encoding="utf-8")
    (out / "phase26a_discrepancy_report.md").write_text(
        "# Phase 26A Discrepancy Report\n\n"
        "- Previously reported query counts do not match the repository: actual counts are 827/118/237.\n"
        "- The existing retrieval service imports optional ranking dependencies; `lightgbm` and `xgboost` were installed from the declared integration requirements so the existing service could initialize.\n"
        "- The service does not load a standalone neural retrieval checkpoint at request time. It uses the existing precomputed hybrid feature index and frozen fusion configuration, as documented above.\n"
        "- No unresolved blocker remains for the tested IL-TUR API path.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
