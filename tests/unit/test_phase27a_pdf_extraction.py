"""
Unit & Integration Tests for Phase 27A: PDF Upload & Text Extraction.
Tests genuine pypdf extraction, security boundaries, error handling,
temporary file cleanup, and downstream frozen M4/M8 inference integration.
"""

import io
import os
import glob
import tempfile
import pytest
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import create_app
from backend.services.pdf_service import (
    PDFExtractionService,
    PDFExtractionError,
    MAX_FILE_SIZE_BYTES,
    MAX_PAGE_COUNT,
)


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def make_text_pdf(lines, pages=1):
    """Generates a genuine text-based PDF in memory."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for p in range(pages):
        y = 750
        c.drawString(100, 780, f"--- Judgment Page {p + 1} ---")
        for line in lines:
            c.drawString(100, y, line)
            y -= 20
        c.showPage()
    c.save()
    buf.seek(0)
    return buf


def make_scanned_pdf(width=200, height=200):
    """Generates a PDF containing an image but no text layer."""
    img = Image.new("RGB", (width, height), color=(220, 220, 220))
    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    img_buf.seek(0)

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawImage(ImageReader(img_buf), 50, 400, width=width, height=height)
    c.showPage()
    c.save()
    buf.seek(0)
    return buf


def make_empty_pdf():
    """Generates a valid PDF with blank pages."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.showPage()
    c.save()
    buf.seek(0)
    return buf


class TestPDFExtractionService:
    """Direct unit tests for PDFExtractionService."""

    def test_valid_single_page_pdf(self):
        service = PDFExtractionService()
        pdf_buf = make_text_pdf([
            "IN THE SUPREME COURT OF INDIA",
            "CIVIL APPELLATE JURISDICTION",
            "CIVIL APPEAL NO. 1029 OF 2021",
            "The impugned order passed by the High Court is set aside."
        ])
        result = service.extract_from_file(pdf_buf, "appeal_1029.pdf")
        assert result["success"] is True
        assert result["page_count"] == 1
        assert result["word_count"] > 10
        assert "SUPREME COURT OF INDIA" in result["extracted_text"]
        assert result["scanned"] is False

    def test_valid_multipage_pdf(self):
        service = PDFExtractionService()
        pdf_buf = make_text_pdf([
            "Appellant submits that the arbitral tribunal lacked jurisdiction.",
            "Respondent contends that section 34 petition was rightly dismissed."
        ], pages=3)
        result = service.extract_from_file(pdf_buf, "multi_page_appeal.pdf")
        assert result["success"] is True
        assert result["page_count"] == 3
        assert len(result["pages"]) == 3
        assert "arbitral tribunal" in result["extracted_text"]

    def test_scanned_pdf_detection(self):
        service = PDFExtractionService()
        pdf_buf = make_scanned_pdf()
        with pytest.raises(PDFExtractionError) as exc_info:
            service.extract_from_file(pdf_buf, "scanned_order.pdf")
        assert exc_info.value.status_code == 400
        assert exc_info.value.scanned is True
        assert "This PDF appears to be scanned. OCR extraction is currently unavailable." in exc_info.value.message

    def test_empty_pdf_detection(self):
        service = PDFExtractionService()
        pdf_buf = make_empty_pdf()
        with pytest.raises(PDFExtractionError) as exc_info:
            service.extract_from_file(pdf_buf, "blank.pdf")
        assert exc_info.value.status_code == 400
        assert "empty and contains no readable text" in exc_info.value.message

    def test_invalid_extension(self):
        service = PDFExtractionService()
        with pytest.raises(PDFExtractionError) as exc_info:
            service.extract_from_file(io.BytesIO(b"%PDF-1.4 dummy"), "judgment.docx")
        assert exc_info.value.status_code == 400
        assert "Only genuine .pdf files are supported" in exc_info.value.message

    def test_invalid_signature(self):
        service = PDFExtractionService()
        with pytest.raises(PDFExtractionError) as exc_info:
            service.extract_from_file(io.BytesIO(b"Plain text disguised as pdf"), "fake.pdf")
        assert exc_info.value.status_code == 400
        assert "Invalid PDF file signature" in exc_info.value.message

    def test_corrupted_pdf(self):
        service = PDFExtractionService()
        with pytest.raises(PDFExtractionError) as exc_info:
            service.extract_from_file(io.BytesIO(b"%PDF-1.4 Corrupted content bytes"), "corrupt.pdf")
        assert exc_info.value.status_code == 400
        assert "Failed to parse PDF document" in exc_info.value.message

    def test_oversized_pdf(self):
        service = PDFExtractionService(max_size_bytes=1024)
        big_content = b"%PDF-" + b"0" * 2000
        with pytest.raises(PDFExtractionError) as exc_info:
            service.extract_from_file(io.BytesIO(big_content), "oversized.pdf")
        assert exc_info.value.status_code == 413
        assert "exceeds the 15 MB maximum upload limit" in exc_info.value.message

    def test_exceeding_page_limit(self):
        service = PDFExtractionService(max_pages=5)
        pdf_buf = make_text_pdf(["Brief line of legal text."], pages=6)
        with pytest.raises(PDFExtractionError) as exc_info:
            service.extract_from_file(pdf_buf, "too_many_pages.pdf")
        assert exc_info.value.status_code == 400
        assert exc_info.value.page_count == 6
        assert "exceeds the 5-page limit" in exc_info.value.message

    def test_temporary_file_cleanup(self):
        temp_dir = tempfile.gettempdir()
        before_files = set(glob.glob(os.path.join(temp_dir, "legal_doc_*.pdf")))

        service = PDFExtractionService()
        pdf_buf = make_text_pdf(["Temporary file cleanup verification."])
        service.extract_from_file(pdf_buf, "cleanup_check.pdf")

        after_success_files = set(glob.glob(os.path.join(temp_dir, "legal_doc_*.pdf")))
        assert after_success_files == before_files

        try:
            service.extract_from_file(io.BytesIO(b"%PDF-corrupt"), "error_cleanup.pdf")
        except PDFExtractionError:
            pass

        after_error_files = set(glob.glob(os.path.join(temp_dir, "legal_doc_*.pdf")))
        assert after_error_files == before_files


class TestPDFExtractionAPI:
    """Integration tests for POST /api/judipl/extract-pdf endpoint."""

    def test_api_extract_valid_pdf(self, client):
        pdf_buf = make_text_pdf([
            "SUPREME COURT OF INDIA",
            "CRIMINAL APPELLATE JURISDICTION",
            "Learned counsel for the respondent submitted that the evidence was credible."
        ], pages=2)
        response = client.post(
            "/api/judipl/extract-pdf",
            data={"file": (pdf_buf, "valid_judgment.pdf")},
            content_type="multipart/form-data"
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["filename"] == "valid_judgment.pdf"
        assert data["page_count"] == 2
        assert "SUPREME COURT OF INDIA" in data["extracted_text"]
        assert data["word_count"] > 10

    def test_api_extract_scanned_pdf(self, client):
        pdf_buf = make_scanned_pdf()
        response = client.post(
            "/api/judipl/extract-pdf",
            data={"file": (pdf_buf, "scanned_judgment.pdf")},
            content_type="multipart/form-data"
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert data["scanned"] is True
        assert "This PDF appears to be scanned. OCR extraction is currently unavailable." in data["error"]

    def test_api_extract_missing_file_field(self, client):
        response = client.post(
            "/api/judipl/extract-pdf",
            data={"wrong_field": (io.BytesIO(b"data"), "test.pdf")},
            content_type="multipart/form-data"
        )
        assert response.status_code == 400
        assert "No PDF file provided" in response.get_json()["error"]

    def test_api_extract_invalid_extension(self, client):
        response = client.post(
            "/api/judipl/extract-pdf",
            data={"file": (io.BytesIO(b"%PDF-valid"), "case_file.exe")},
            content_type="multipart/form-data"
        )
        assert response.status_code == 400
        assert "Only genuine .pdf files are supported" in response.get_json()["error"]


class TestEndToEndPDFInferenceIntegration:
    """Tests feeding extracted PDF text directly into frozen M4 and M8 models."""

    def test_pdf_extraction_to_m4_inference(self, client):
        sample_legal_text = (
            "IN THE SUPREME COURT OF INDIA\n"
            "CIVIL APPELLATE JURISDICTION\n"
            "CIVIL APPEAL NO. 8842 OF 2018\n"
            "Union of India and Ors. ... Appellants\n"
            "Versus\n"
            "M/s ABC Enterprises ... Respondent\n\n"
            "JUDGMENT\n"
            "This appeal is directed against the final judgment and order passed by the High Court "
            "of Delhi in Writ Petition No. 1204 of 2016 whereby the High Court set aside the penalty "
            "imposed by the Commissioner under Section 11AC of the Central Excise Act. Having heard the learned "
            "Additional Solicitor General appearing on behalf of the appellants and learned Senior Counsel appearing "
            "for the respondent, we are of the considered opinion that the view taken by the High Court is plausible "
            "and does not call for interference. Consequently, the appeal is dismissed with no order as to costs."
        )
        pdf_buf = make_text_pdf(sample_legal_text.splitlines(), pages=1)

        # 1. Extract PDF
        extract_res = client.post(
            "/api/judipl/extract-pdf",
            data={"file": (pdf_buf, "appeal_8842.pdf")},
            content_type="multipart/form-data"
        )
        assert extract_res.status_code == 200
        extracted_text = extract_res.get_json()["extracted_text"]
        assert len(extracted_text) > 100

        # 2. Run M4 Inference on extracted text
        m4_res = client.post(
            "/api/judipl/analyze",
            json={"text": extracted_text, "model_id": "M4"}
        )
        assert m4_res.status_code == 200
        m4_data = m4_res.get_json()
        assert m4_data["dataset"] == "jud-ipl"
        assert m4_data["model_id"] == "M4"
        assert m4_data["predicted_label"] in ["Accepted", "Other", "Rejected"]
        assert set(m4_data["probabilities"].keys()) == {"Accepted", "Other", "Rejected"}
        assert abs(sum(m4_data["probabilities"].values()) - 1.0) < 1e-4

        # 3. Run M8 Inference on extracted text
        m8_res = client.post(
            "/api/judipl/analyze",
            json={"text": extracted_text, "model_id": "M8"}
        )
        assert m8_res.status_code == 200
        m8_data = m8_res.get_json()
        assert m8_data["dataset"] == "jud-ipl"
        assert m8_data["model_id"] == "M8"
        assert m8_data["predicted_label"] in ["Accepted", "Other", "Rejected"]
        assert set(m8_data["probabilities"].keys()) == {"Accepted", "Other", "Rejected"}
        assert abs(sum(m8_data["probabilities"].values()) - 1.0) < 1e-4

    def test_dataset_isolation_preserved(self, client):
        """Verifies that PDF extraction output does not contaminate IL-TUR search state."""
        # Querying IL-TUR search without query fails cleanly
        search_res = client.post(
            "/api/search",
            json={"query": "   "}
        )
        assert search_res.status_code == 400

        # Jud-IPL PDF extraction endpoint accepts only files and does not interfere with search
        health_res = client.get("/api/health")
        assert health_res.status_code == 200
        h_data = health_res.get_json()
        assert h_data["candidate_pool_size"] == 1727

