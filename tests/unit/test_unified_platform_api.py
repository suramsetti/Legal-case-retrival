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


def test_datasets_expose_independent_workspaces(client):
    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.get_json()
    assert {item["id"] for item in data["datasets"]} == {"jud-ipl", "il-tur"}
    assert data["datasets"][0]["task"]


def test_judipl_metrics_are_served_from_correction_artifacts(client):
    response = client.get("/api/judipl/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert data["dataset"]["eligible_test_size"] == 6311
    assert {row["model_id"] for row in data["models"]} >= {"M4", "M8"}


def test_judipl_analysis_validates_input(client):
    response = client.post("/api/judipl/analyze", json={"text": ""})
    assert response.status_code == 400
    assert "cannot be empty" in response.get_json()["error"]


def test_judipl_analysis_rejects_unknown_model(client):
    response = client.post("/api/judipl/analyze", json={"text": "A judgment text.", "model_id": "M9"})
    assert response.status_code == 400


@pytest.mark.parametrize("dataset,expected_field", [
    ("jud-ipl", "label"),
    ("il-tur", "candidate_id"),
])
def test_dataset_explorer_returns_real_bounded_metadata(client, dataset, expected_field):
    response = client.get(f"/api/explorer?dataset={dataset}&page_size=2")
    assert response.status_code == 200
    body = response.get_json()
    assert body["dataset"] == dataset
    assert body["total"] > 0
    assert len(body["rows"]) == 2
    assert expected_field in body["available_fields"]
    assert expected_field in body["rows"][0]


def test_dataset_explorer_rejects_unknown_dataset(client):
    response = client.get("/api/explorer?dataset=unknown")
    assert response.status_code == 400
