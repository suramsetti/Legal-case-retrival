import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_pcr_api_returns_ranked_mapped_candidates(client):
    response = client.post("/api/search", json={"query": "appeal against conviction under section 302 IPC", "top_k": 5})
    assert response.status_code == 200
    body = response.get_json()
    assert body["query_type"] == "free_text"
    assert 0 < len(body["results"]) <= 5
    assert [row["rank"] for row in body["results"]] == list(range(1, len(body["results"]) + 1))
    assert all(row["case_id"] and row["snippet"] for row in body["results"])
    assert all(body["results"][i]["hybrid_score"] >= body["results"][i + 1]["hybrid_score"] for i in range(len(body["results"]) - 1))


def test_benchmark_query_excludes_self_match(client):
    service = client.application.config["RETRIEVAL_SERVICE"]
    query_id = str(service.df_test_features["query_id"].iloc[0])
    response = client.post("/api/search", json={"query": query_id, "top_k": 5})
    assert response.status_code == 200
    assert all(row["case_id"] != query_id for row in response.get_json()["results"])


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "   "}, {"query": 123}, {"query": "appeal", "top_k": "bad"}, {"query": "appeal", "top_k": 101}])
def test_invalid_queries_return_json_errors(client, payload):
    response = client.post("/api/search", json=payload)
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_dataset_isolation(client):
    response = client.get("/api/datasets")
    data = response.get_json()
    il_tur = next(item for item in data["datasets"] if item["id"] == "il-tur")
    assert il_tur["metadata"]["candidate_pool_size"] == 1727
    assert "model" not in il_tur["metadata"]
