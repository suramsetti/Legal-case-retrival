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


def test_iltur_metrics_endpoint(client):
    response = client.get("/api/iltur/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert "model_comparison" in data
    models = data["model_comparison"]
    assert "Weighted_Linear_Fusion" in models
    wlf = models["Weighted_Linear_Fusion"]
    assert "MRR" in wlf
    assert "nDCG@5" in wlf
    assert "P@1" in wlf
    assert wlf["P@1"] > 0.60


def test_explorer_document_detail_judipl(client):
    # First get an ID from explorer
    exp_res = client.get("/api/explorer?dataset=jud-ipl&page=1&page_size=1")
    assert exp_res.status_code == 200
    rows = exp_res.get_json()["rows"]
    assert len(rows) > 0
    doc_id = rows[0]["id"]

    # Request document detail
    detail_res = client.get(f"/api/explorer/document?dataset=jud-ipl&id={doc_id}")
    assert detail_res.status_code == 200
    detail = detail_res.get_json()
    assert detail["dataset"] == "jud-ipl"
    assert "document" in detail
    assert detail["document"]["id"] == str(doc_id)


def test_explorer_document_detail_iltur(client):
    # First get an ID from explorer
    exp_res = client.get("/api/explorer?dataset=il-tur&page=1&page_size=1")
    assert exp_res.status_code == 200
    rows = exp_res.get_json()["rows"]
    assert len(rows) > 0
    cand_id = rows[0]["candidate_id"]

    # Request document detail
    detail_res = client.get(f"/api/explorer/document?dataset=il-tur&id={cand_id}")
    assert detail_res.status_code == 200
    detail = detail_res.get_json()
    assert detail["dataset"] == "il-tur"
    assert "document" in detail
    assert detail["document"]["candidate_id"] == str(cand_id)
    assert "snippet" in detail["document"]
    assert "statutory_provisions" in detail["document"]


def test_explorer_document_missing_id(client):
    response = client.get("/api/explorer/document?dataset=jud-ipl")
    assert response.status_code == 400
    assert "required" in response.get_json()["error"].lower()


def test_explorer_document_not_found(client):
    response = client.get("/api/explorer/document?dataset=jud-ipl&id=NON_EXISTENT_ID_999999")
    assert response.status_code == 404


def test_datasets_metadata_enriched(client):
    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.get_json()
    datasets = {d["id"]: d for d in data["datasets"]}
    assert "jud-ipl" in datasets
    assert "il-tur" in datasets
    iltur_meta = datasets["il-tur"]["metadata"]
    assert iltur_meta["candidate_pool_size"] == 1727
    assert iltur_meta["test_queries_count"] == 237
    assert iltur_meta["model_name"] == "Weighted Linear Score Fusion"


def test_judipl_m4_and_m8_inference(client):
    sample_text = (
        "The appellant filed this criminal appeal challenging the judgment and order of conviction "
        "passed by the High Court under Section 302 read with Section 34 of the Indian Penal Code. "
        "The prosecution examined multiple eyewitnesses. We have perused the record and heard the counsel."
    )
    for model_id in ["M4", "M8"]:
        res = client.post("/api/judipl/analyze", json={"text": sample_text, "model_id": model_id})
        assert res.status_code == 200
        body = res.get_json()
        assert body["model_id"] == model_id
        assert body["predicted_label"] in ["Accepted", "Other", "Rejected"]
        assert "probabilities" in body
        assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4

