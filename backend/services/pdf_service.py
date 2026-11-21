"""
PDF Extraction Service for Legal Documents.
Phase 27A implementation using pypdf.
"""

import os
import re
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pypdf
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger("PDFExtractionService")

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
MAX_PAGE_COUNT = 100
PDF_MAGIC_BYTES = b"%PDF-"


class PDFExtractionError(Exception):
    """Custom exception with HTTP status code and user-facing message."""
    def __init__(self, message: str, status_code: int = 400, scanned: bool = False, page_count: int = 0):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.scanned = scanned
        self.page_count = page_count


class PDFExtractionService:
    """Secure, high-integrity PDF text extractor for Indian Supreme Court judgments."""

    def __init__(self, max_size_bytes: int = MAX_FILE_SIZE_BYTES, max_pages: int = MAX_PAGE_COUNT):
        self.max_size_bytes = max_size_bytes
        self.max_pages = max_pages

    def extract_from_file(self, file_obj, filename: str) -> Dict[str, Any]:
        """
        Validates, saves to secure temporary storage, extracts text across pages,
        detects scanned/empty documents, and ensures automatic temporary file cleanup.
        """
        safe_filename = Path(filename).name
        if not safe_filename.lower().endswith(".pdf"):
            raise PDFExtractionError(
                f"Invalid file extension for '{safe_filename}'. Only genuine .pdf files are supported.",
                status_code=400
            )

        # Read content bytes safely to validate size
        file_bytes = file_obj.read()
        file_size = len(file_bytes)

        if file_size == 0:
            raise PDFExtractionError(
                "The uploaded file is empty (0 bytes).",
                status_code=400
            )

        if file_size > self.max_size_bytes:
            size_mb = file_size / (1024 * 1024)
            raise PDFExtractionError(
                f"File size ({size_mb:.2f} MB) exceeds the 15 MB maximum upload limit.",
                status_code=413
            )

        # Validate magic signature
        if not file_bytes.startswith(PDF_MAGIC_BYTES):
            raise PDFExtractionError(
                "Invalid PDF file signature. The file is corrupted or not a genuine PDF document.",
                status_code=400
            )

        # Write to secure temporary file with randomized filename
        tmp_fd, tmp_path_str = tempfile.mkstemp(prefix="legal_doc_", suffix=".pdf")
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_str)

        try:
            tmp_path.write_bytes(file_bytes)

            try:
                reader = PdfReader(str(tmp_path), strict=False)
            except Exception as exc:
                logger.warning("pypdf failed to parse %s: %s", safe_filename, exc)
                raise PDFExtractionError(
                    "Failed to parse PDF document. The file may be password-protected or corrupted.",
                    status_code=400
                )

            page_count = len(reader.pages)
            if page_count == 0:
                raise PDFExtractionError(
                    "The uploaded PDF document contains 0 pages.",
                    status_code=400,
                    page_count=0
                )

            if page_count > self.max_pages:
                raise PDFExtractionError(
                    f"PDF document exceeds the {self.max_pages}-page limit (contains {page_count} pages).",
                    status_code=400,
                    page_count=page_count
                )

            extracted_pages = []
            has_scanned_images = False
            total_images = 0

            for page_idx, page in enumerate(reader.pages):
                try:
                    raw_text = page.extract_text() or ""
                except Exception as text_exc:
                    logger.warning("Error extracting text on page %d: %s", page_idx + 1, text_exc)
                    raw_text = ""

                # Check images on page
                try:
                    page_imgs = len(page.images)
                    total_images += page_imgs
                    if page_imgs > 0:
                        has_scanned_images = True
                except Exception:
                    pass

                # Clean whitespace per page
                lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw_text.splitlines()]
                clean_lines = [l for l in lines if l]
                page_clean = "\n".join(clean_lines)

                extracted_pages.append({
                    "page_number": page_idx + 1,
                    "text": page_clean,
                    "character_count": len(page_clean),
                    "word_count": len(page_clean.split())
                })

            # Check if entire document has extractable text
            non_empty_pages = [p for p in extracted_pages if p["text"]]
            combined_text = "\n\n".join(p["text"] for p in non_empty_pages).strip()
            total_chars = len(combined_text)
            total_words = len(combined_text.split())

            # Detect scanned PDFs without an extractable text layer
            if total_chars == 0 or (total_words < 10 and has_scanned_images):
                if has_scanned_images:
                    raise PDFExtractionError(
                        "This PDF appears to be scanned. OCR extraction is currently unavailable.",
                        status_code=400,
                        scanned=True,
                        page_count=page_count
                    )
                else:
                    raise PDFExtractionError(
                        "The uploaded PDF is empty and contains no readable text.",
                        status_code=400,
                        scanned=False,
                        page_count=page_count
                    )

            return {
                "success": True,
                "filename": safe_filename,
                "file_size_bytes": file_size,
                "page_count": page_count,
                "character_count": total_chars,
                "word_count": total_words,
                "extracted_text": combined_text,
                "extraction_status": "SUCCESS",
                "scanned": False,
                "pages": [
                    {
                        "page": p["page_number"],
                        "character_count": p["character_count"],
                        "word_count": p["word_count"]
                    }
                    for p in extracted_pages
                ]
            }

        finally:
            # Ensure strict cleanup of temporary file
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception as cleanup_err:
                    logger.warning("Could not delete temporary PDF file %s: %s", tmp_path, cleanup_err)

