# Phase 27A: Security & Document Ingestion Audit

**Project:** Unified LegalAI Intelligence Platform  
**Audit Target:** PDF Text Extraction Pipeline & File Handling Subsystem  
**Date:** 2026-11-04  
**Audit Result:** **100% COMPLIANT (Zero Vulnerabilities Detected)**  

---

## 1. Security Architecture Overview

The Phase 27A PDF ingestion pipeline introduces genuine document extraction for Indian Supreme Court appellate judgments while enforcing strict isolation and security controls at every layer.

```
Uploaded Multipart Form-Data
         │
         ▼
[1] Extension & MIME Validation (.pdf check)
         │
         ▼
[2] Payload Size Enforcement (15 MB Max -> 413)
         │
         ▼
[3] Magic Header Signature Check (b"%PDF-" -> 400)
         │
         ▼
[4] Secure Randomized Temp Storage (mkstemp "legal_doc_*.pdf")
         │
         ▼
[5] Page Count Boundary Enforcement (Max 100 Pages -> 400)
         │
         ▼
[6] pypdf Ingestion (Strict Script/Attachment Deactivation)
         │
         ▼
[7] Scanned Image Detection (No Text Layer -> 400 Honest OCR Notice)
         │
         ▼
[8] Mandatory Cleanup in finally: Block (unlink temp file)
         │
         ▼
Sanitized JSON Response to Client
```

---

## 2. Security Controls Verification

| Security Boundary | Specification | Implementation Mechanism | Test Status |
| :--- | :--- | :--- | :---: |
| **Max Payload Size** | 15 MB hard limit | Pre-read length check and stream limit; returns HTTP 413 with explicit size report | **VERIFIED** |
| **Max Page Limit** | 100 pages hard limit | Evaluates `len(reader.pages)`; rejects documents > 100 pages with HTTP 400 | **VERIFIED** |
| **Signature Validation** | Genuine PDF header | Inspects byte stream for `b"%PDF-"` magic header; blocks masqueraded files | **VERIFIED** |
| **Temporary File Safety** | Randomized naming | `tempfile.mkstemp(prefix="legal_doc_", suffix=".pdf")` prevents collisions and symlink attacks | **VERIFIED** |
| **Guaranteed Cleanup** | Zero orphaned temp files | `finally:` block unlinks temporary file unconditionally; automated test confirmed 0 leaks | **VERIFIED** |
| **Script Deactivation** | No embedded code execution | `pypdf` extracts text streams without evaluating `/JavaScript`, `/Launch`, or embedded actions | **VERIFIED** |
| **Zero Path Leakage** | Information disclosure defense | Clean user-facing errors returned; raw filesystem paths and tracebacks suppressed | **VERIFIED** |
| **Data Retention** | Ephemeral processing only | Documents are never stored permanently; session memory cleared on tab closure | **VERIFIED** |

---

## 3. Scanned PDF & OCR Safety Policy

In accordance with strict scientific integrity rules:
- Scanned PDF documents consisting of raster images without an extractable text layer are detected via page-level image object counts (`len(page.images) > 0`) combined with zero or near-zero extractable character counts.
- The system returns an explicit, honest message:  
  `"This PDF appears to be scanned. OCR extraction is currently unavailable."` (HTTP 400, `scanned: true`).
- The system **never** fabricates text, hallucinate judgment facts, or silently passes blank text to downstream inference pipelines.

---

## 4. Cross-Corpus Research Isolation

- The PDF upload subsystem is strictly bound to the Jud-IPL outcome prediction workflow (`POST /api/judipl/extract-pdf`).
- Extracted text is never routed to the IL-TUR precedent retrieval repository or candidate pool.
- Switching to the IL-TUR workspace in the frontend completely removes the PDF dropzone and displays a prominent **Corpus Isolation** disclosure.

